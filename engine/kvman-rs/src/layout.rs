//! A page block's layout as data: what one pool page (pool::PAGE tokens) of one rank holds, binding by binding, in the
//! order every rank shares (main model, then draft). The engine describes its attention bindings this way and every
//! size below follows, so nothing here knows a model: the row formats stay in engine/attn/kvfmt.h.
//!
//! A page block is every binding's page, one after the other: its latent rows, then its indexer keys when it has any.
//! A store keeps it in a slot, the block rounded up to 4 KiB. Past R a CP binding holds only this rank's shard of a
//! run's tail, so the tail goes to a store in two parts (stash.rs Part): the shard pages in the page block's layout, and
//! each logical page's replicated record (the page of every binding that is not CP), `per_slot` records to a slot.

use crate::tier::ALIGN;

/// One binding's bytes of one page.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Binding {
    /// its latent cache
    pub latent: u32,
    /// its indexer keys, 0 for a binding without any
    pub index: u32,
    /// not 0: sharded across the ranks past R
    pub cp: u32,
}

impl Binding {
    pub fn bytes(&self) -> u64 {
        self.latent as u64 + self.index as u64
    }
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Sizes {
    /// one page of every binding
    pub block: u64,
    /// the block as a store keeps it, a multiple of 4 KiB
    pub slot: u64,
    /// one page's replicated record: the bindings that are not CP
    pub rep: u64,
    /// replicated records to a slot, 0 when no binding is CP (no run has a tail)
    pub per_slot: u32,
}

/// Each binding's offset in a page block and in a replicated record (a CP binding has no place in the record).
pub fn offsets(bindings: &[Binding]) -> impl Iterator<Item = (u64, u64)> + '_ {
    bindings.iter().scan((0, 0), |at, b| {
        let here = *at;
        at.0 += b.bytes();
        at.1 += if b.cp != 0 { 0 } else { b.bytes() };
        Some(here)
    })
}

pub fn sizes(bindings: &[Binding]) -> Sizes {
    let block: u64 = bindings.iter().map(Binding::bytes).sum();
    let rep: u64 = bindings.iter().filter(|b| b.cp == 0).map(Binding::bytes).sum();
    let slot = block.div_ceil(ALIGN as u64) * ALIGN as u64;
    let per_slot = if rep != 0 && bindings.iter().any(|b| b.cp != 0) { (slot / rep) as u32 } else { 0 };
    Sizes { block, slot, rep, per_slot }
}

/// `*out` = the sizes of `n` bindings; `off` and `rep`, each NULL or `n` words: every binding's offsets (offsets()).
///
/// # Safety
/// `b` must point to `n` bindings (NULL when n is 0), `out` be valid for a write, `off` and `rep` NULL or `n` words.
#[no_mangle]
pub unsafe extern "C" fn g6kv_layout_sizes(b: *const Binding, n: usize, out: *mut Sizes, off: *mut u64, rep: *mut u64) {
    let bindings = if n == 0 { &[][..] } else { std::slice::from_raw_parts(b, n) };
    *out = sizes(bindings);
    for (i, (o, r)) in offsets(bindings).enumerate() {
        if !off.is_null() {
            *off.add(i) = o;
        }
        if !rep.is_null() {
            *rep.add(i) = r;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_block_is_every_bindings_page() {
        // an indexed CP binding, a plain CP binding, a replicated one with keys
        let b = [
            Binding { latent: 4096, index: 512, cp: 1 },
            Binding { latent: 4096, index: 0, cp: 1 },
            Binding { latent: 1024, index: 256, cp: 0 },
        ];
        assert_eq!(sizes(&b), Sizes { block: 9984, slot: 12288, rep: 1280, per_slot: 9 });
        assert_eq!(offsets(&b).collect::<Vec<_>>(), [(0, 0), (4608, 0), (8704, 0)]);
    }

    #[test]
    fn no_tail_without_a_cp_binding() {
        let b = [Binding { latent: 4096, index: 0, cp: 0 }, Binding { latent: 100, index: 28, cp: 0 }];
        assert_eq!(sizes(&b), Sizes { block: 4224, slot: 8192, rep: 4224, per_slot: 0 });
        assert_eq!(offsets(&b).collect::<Vec<_>>(), [(0, 0), (4096, 4096)]);
        assert_eq!(sizes(&[]), Sizes::default());
    }

    #[test]
    fn the_c_call_fills_what_it_is_given() {
        let b = [Binding { latent: 64, index: 16, cp: 1 }, Binding { latent: 32, index: 0, cp: 0 }];
        let (mut s, mut off, mut rep) = (Sizes::default(), [9u64; 2], [9u64; 2]);
        unsafe { g6kv_layout_sizes(b.as_ptr(), 2, &mut s, off.as_mut_ptr(), rep.as_mut_ptr()) };
        assert_eq!((s, off, rep), (sizes(&b), [0, 80], [0, 0]));
        unsafe { g6kv_layout_sizes(std::ptr::null(), 0, &mut s, std::ptr::null_mut(), std::ptr::null_mut()) };
        assert_eq!(s, Sizes::default());
    }
}
