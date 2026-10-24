//! src/fetch.rs against tests/data/v2fetch (tests/make_v2fetch_fixture.py): the ranges rank 1 of TP=4 fetches under a
//! manifest that mixes the widths, as tools/convert/v2.py works them out from the full layer index files, and the
//! halves a layout gives a rank.

use g6load::fetch::{self, Choice};
use g6load::json;
use g6load::layout::{self, Group, Layout, Model};
use std::collections::{BTreeMap, BTreeSet};

const D: &str = concat!(env!("CARGO_MANIFEST_DIR"), "/tests/data/v2fetch");

fn plan(units: &[fetch::Unit]) -> BTreeMap<String, Vec<(u64, u64)>> {
    let mut out: BTreeMap<String, Vec<(u64, u64)>> = BTreeMap::new();
    for u in units {
        out.entry(fetch::width_file(u.layer, u.k)).or_default().push((u.offset, u.nbytes));
    }
    for r in out.values_mut() {
        r.sort();
    }
    out
}

#[test]
fn the_plan_matches_the_index_files() {
    let choice = Choice::manifest(&format!("{D}/manifest.json")).unwrap();
    let units = fetch::units(&format!("{D}/slices"), 4, 1, &choice, None, None).unwrap();
    let want = json::parse(&std::fs::read_to_string(format!("{D}/plan.json")).unwrap()).unwrap();
    let want: BTreeMap<String, Vec<(u64, u64)>> = want.obj().unwrap().iter()
        .map(|(k, v)| (k.clone(), v.arr().unwrap().iter()
            .map(|r| (r.arr().unwrap()[0].int().unwrap() as u64, r.arr().unwrap()[1].int().unwrap() as u64)).collect()))
        .collect();
    assert_eq!(plan(&units), want);
}

#[test]
fn whole_widths_and_layer_choice() {
    let units = fetch::units(&format!("{D}/slices"), 4, 1, &Choice::Widths(vec![3]), Some(&[4]), None).unwrap();
    // rank 1 owns 8 of layer 4's 2 x 16 halves: 8 units x 3 projections
    assert_eq!(units.len(), 24);
    assert!(units.iter().all(|u| u.layer == 4 && u.k == 3 && u.nbytes == 3 * 98304));
}

#[test]
fn refusals_name_the_cause() {
    let choice = Choice::manifest(&format!("{D}/manifest.json")).unwrap();
    let e = fetch::units(&format!("{D}/slices"), 3, 1, &choice, None, None).err().unwrap();
    assert!(e.contains("no index slices for TP 3 rank 1"), "{e}");
    let e = fetch::units(&format!("{D}/slices"), 4, 1, &choice, Some(&[9]), None).err().unwrap();
    assert!(e.contains("layer 9"), "{e}");
}

fn model() -> Model {
    Model::release(&json::parse(&std::fs::read_to_string(format!("{D}/manifest.json")).unwrap()).unwrap()).unwrap()
}

fn names(s: &str) -> Vec<String> {
    s.split(',').map(str::to_string).collect()
}

type Key = (usize, u8, usize, usize, usize, u64, u64, u64);

fn keys(units: &[fetch::Unit]) -> Vec<Key> {
    units.iter().map(|u| (u.layer, u.k, u.expert, u.half, u.proj, u.offset, u.nbytes, u.checksum)).collect()
}

/// A layout at TP=4 the formulas do not give: planned from TP=3's, so the new member takes a quarter from the others.
fn planned() -> Layout {
    let m = model();
    let p = layout::plan(&m, &Layout::formula(&m, names("a,b,c")).unwrap(), names("a,b,c,d")).unwrap();
    p.check(&m).unwrap();
    p
}

#[test]
fn the_formula_layout_is_todays_plan() {
    let m = model();
    let f4 = Layout::formula(&m, names("0,1,2,3")).unwrap();
    for choice in [Choice::manifest(&format!("{D}/manifest.json")).unwrap(), Choice::Widths(vec![2, 3, 4])] {
        for r in 0..4 {
            let a = fetch::units(&format!("{D}/slices"), 4, r, &choice, None, None).unwrap();
            let b = fetch::units(&format!("{D}/slices"), 4, r, &choice, None, Some(&f4)).unwrap();
            assert_eq!(keys(&a), keys(&b), "rank {r}");
        }
    }
}

#[test]
fn a_layout_fetches_its_halves_once_from_the_slices_that_list_them() {
    let p = planned();
    let choice = Choice::manifest(&format!("{D}/manifest.json")).unwrap();
    // every unit of the model, from the four ranks' own plans
    let all: BTreeSet<Key> = (0..4).flat_map(|r| keys(&fetch::units(&format!("{D}/slices"), 4, r, &choice, None, None).unwrap()))
        .collect();
    let mut seen = BTreeSet::new();
    let mut moved = 0;
    for r in 0..4 {
        let ks = keys(&fetch::units(&format!("{D}/slices"), 4, r, &choice, None, Some(&p)).unwrap());
        let mut want = BTreeSet::new();
        for l in [3, 4] {
            for e in 0..2 {
                for h in p.held(Group::Halves(l, e), r).unwrap() {
                    want.insert((l, e, h));
                    moved += usize::from((16 * e + h + l) % 4 != r);
                }
            }
        }
        let got: BTreeSet<(usize, usize, usize)> = ks.iter().map(|k| (k.0, k.2, k.3)).collect();
        assert_eq!(got, want, "rank {r}");
        assert_eq!(ks.len(), want.len() * 3, "rank {r}: three projections a half");
        for k in ks {
            assert!(all.contains(&k), "rank {r}: {k:?} is not the release's unit");
            assert!(seen.insert((k.0, k.2, k.3, k.4)), "{k:?} fetched twice");
        }
    }
    assert_eq!(seen.len(), all.len());
    assert!(moved > 0, "the planned layout is the formula's");
}

#[test]
fn verify_follows_the_layout() {
    let p = planned();
    let choice = Choice::manifest(&format!("{D}/manifest.json")).unwrap();
    let (r, f4) = (3, Layout::formula(&model(), names("0,1,2,3")).unwrap());
    let units = fetch::units(&format!("{D}/slices"), 4, r, &choice, None, Some(&p)).unwrap();
    let theirs = fetch::units(&format!("{D}/slices"), 4, r, &choice, None, Some(&f4)).unwrap();
    // the fixture's trellis bytes are zero: zero-filled width files hold every half
    let dir = std::env::temp_dir().join(format!("v2fetch-verify-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let mut size: BTreeMap<String, u64> = BTreeMap::new();
    for u in units.iter().chain(&theirs) {
        let n = size.entry(fetch::width_file(u.layer, u.k)).or_default();
        *n = (*n).max(u.offset + u.nbytes);
    }
    let file = |name: &str| std::fs::OpenOptions::new().write(true).create(true).open(dir.join(name)).unwrap();
    for (name, n) in &size {
        file(name).set_len(*n).unwrap();
    }
    let d = dir.to_str().unwrap();
    assert_eq!(fetch::verify(d, &units).unwrap(), Vec::<usize>::new());
    // a byte off in a half the layout gives the rank and the formula does not, and in one only the formula gives it
    let mine = units.iter().position(|u| (16 * u.expert + u.half + u.layer) % 4 != r).unwrap();
    let other = theirs.iter().find(|u| !units.iter().any(|v| (v.layer, v.expert, v.half) == (u.layer, u.expert, u.half))).unwrap();
    use std::os::unix::fs::FileExt;
    for u in [&units[mine], other] {
        file(&fetch::width_file(u.layer, u.k)).write_all_at(&[1], u.offset + 7).unwrap();
    }
    assert_eq!(fetch::verify(d, &units).unwrap(), vec![mine]);
    // a width file that ends before the rank's last half in it
    let last = (0..units.len()).max_by_key(|&i| (fetch::width_file(units[i].layer, units[i].k), units[i].offset)).unwrap();
    let u = &units[last];
    file(&fetch::width_file(u.layer, u.k)).set_len(u.offset + 1).unwrap();
    let bad = fetch::verify(d, &units).unwrap();
    assert!(bad.contains(&mine) && bad.contains(&last), "{bad:?}");
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn a_layout_of_another_width_is_refused() {
    let m = model();
    let choice = Choice::manifest(&format!("{D}/manifest.json")).unwrap();
    let e = fetch::units(&format!("{D}/slices"), 4, 1, &choice, None, Some(&Layout::formula(&m, names("a,b,c")).unwrap()))
        .err().unwrap();
    assert!(e.contains("layout of 3 members"), "{e}");
}
