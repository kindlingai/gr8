//! How the KV tier copies (kv_tier.cuh KvTiers::run) are cut into batches (was engine/runtime/s2/kv_plan.h). A batch
//! is one kind of copy (0 spill, 1 move, 2 restore), at most `batch` blocks, each in one host slot, and at most
//! `items_cap` page copies. A batch's store reads go out at once, then its writes, so a batch never holds two copies
//! whose order matters.

/// One copy the pool queued (kvman-rs g6kv_io, the same layout).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Io {
    pub kind: u32,
    pub page: u32,
    pub tier: u32,
    pub to: u32,
    pub slot: u64,
    pub to_slot: u64,
    pub part: u32,
    pub sub: u32,
}

/// One copy of a batch: page, part, sub, and the host slot it goes to or comes from.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Item {
    pub page: i32,
    pub part: i32,
    pub sub: i32,
    pub hslot: i32,
}

/// Per host slot j: block `slots[j]` of tier `from[j]` (a move: into block `to_slots[j]` of tier `to[j]`). `items`:
/// the spills' or restores' page copies.
#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct Batch {
    pub kind: u32,
    pub items: Vec<Item>,
    pub from: Vec<u32>,
    pub to: Vec<u32>,
    pub slots: Vec<u64>,
    pub to_slots: Vec<u64>,
}

/// The batch from `io[*i]` on; advances `*i` past it. `per_slot`: the replicated parts a block holds (0: no binding is
/// CP). None when no copy fits, which a caller treats as an error.
pub fn next_batch(io: &[Io], i: &mut usize, batch: i32, per_slot: u32, items_cap: i32) -> Option<Batch> {
    let mut b = Batch { kind: io.get(*i)?.kind, ..Default::default() };
    let need = if per_slot != 0 { per_slot as i32 } else { 1 }; // a block's most copies: a page, a shard, or per_slot parts
    while *i < io.len() && io[*i].kind == b.kind {
        let x = io[*i];
        if b.kind == 1 {
            // a move that reads a block an earlier move of the batch writes (the batch reads first), or writes a block an
            // earlier move writes (one write_many has no order): the next batch
            let dep = (0..b.slots.len()).any(|j| {
                (b.to[j] == x.tier && b.to_slots[j] == x.slot) || (b.to[j] == x.to && b.to_slots[j] == x.to_slot)
            });
            if dep || b.slots.len() as i32 == batch {
                break;
            }
            b.from.push(x.tier);
            b.to.push(x.to);
            b.slots.push(x.slot);
            b.to_slots.push(x.to_slot);
            *i += 1;
            continue;
        }
        // a block's copies are next to each other: one host slot for them all. A spill's block starts at part 0, part 1
        // or part 2 sub 0, so only its later replicated parts join it; a block spilled twice in one call (a slot freed and
        // taken again) is two writes, in order: the second starts the next batch. Restores of one block share its host
        // slot (read once, copied to each page)
        let same = !b.slots.is_empty()
            && *b.from.last().unwrap() == x.tier
            && *b.slots.last().unwrap() == x.slot
            && (b.kind == 2 || (x.part == 2 && x.sub > 0));
        if same {
            if b.items.len() as i32 == items_cap {
                break; // a restore only: the next batch reads the block again
            }
        } else {
            if b.slots.len() as i32 == batch || b.items.len() as i32 + need > items_cap {
                break;
            }
            if b.kind == 0 && (0..b.slots.len()).any(|j| b.from[j] == x.tier && b.slots[j] == x.slot) {
                break;
            }
            b.from.push(x.tier);
            b.slots.push(x.slot);
        }
        b.items.push(Item {
            page: x.page as i32,
            part: x.part as i32,
            sub: x.sub as i32,
            hslot: b.slots.len() as i32 - 1,
        });
        *i += 1;
    }
    if b.slots.is_empty() {
        None
    } else {
        Some(b)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn spill(page: u32, tier: u32, slot: u64, part: u32, sub: u32) -> Io {
        Io { kind: 0, page, tier, slot, part, sub, ..Default::default() }
    }

    #[test]
    fn spills_share_a_slot_only_for_later_parts() {
        let io = [spill(1, 0, 5, 2, 0), spill(1, 0, 5, 2, 1), spill(2, 0, 6, 0, 0), spill(3, 0, 5, 0, 0)];
        let mut i = 0;
        let b = next_batch(&io, &mut i, 8, 2, 16).unwrap();
        // the block at slot 5 again (part 0) is a second write: the next batch
        assert_eq!((i, b.slots.clone(), b.items.len()), (3, vec![5, 6], 3));
        assert_eq!(b.items[1].hslot, 0);
        let b2 = next_batch(&io, &mut i, 8, 2, 16).unwrap();
        assert_eq!((i, b2.slots), (4, vec![5]));
    }

    #[test]
    fn moves_stop_at_a_dependency_and_the_batch_size() {
        let mv = |tier, slot, to, to_slot| Io { kind: 1, tier, slot, to, to_slot, ..Default::default() };
        let io = [mv(0, 1, 1, 9), mv(1, 9, 2, 3), mv(0, 2, 1, 8)];
        let mut i = 0;
        assert_eq!(next_batch(&io, &mut i, 8, 0, 16).unwrap().slots, vec![1]); // the second reads what the first writes
        assert_eq!(next_batch(&io, &mut i, 1, 0, 16).unwrap().slots, vec![9]); // batch 1
        assert_eq!(i, 2);
    }

    #[test]
    fn no_copy_fits() {
        let io = [spill(1, 0, 5, 0, 0)];
        let mut i = 0;
        assert!(next_batch(&io, &mut i, 8, 4, 2).is_none()); // a block needs 4 copies, the cap is 2
        assert!(next_batch(&[], &mut 0, 8, 0, 2).is_none());
    }
}
