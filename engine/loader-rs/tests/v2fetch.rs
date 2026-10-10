//! src/fetch.rs against tests/data/v2fetch (tests/make_v2fetch_fixture.py): the ranges rank 1 of TP=4 fetches under a
//! manifest that mixes the widths, as tools/convert/v2.py works them out from the full layer index files.

use g6load::fetch::{self, Choice};
use g6load::json;
use std::collections::BTreeMap;

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
    let units = fetch::units(&format!("{D}/slices"), 4, 1, &choice, None).unwrap();
    let want = json::parse(&std::fs::read_to_string(format!("{D}/plan.json")).unwrap()).unwrap();
    let want: BTreeMap<String, Vec<(u64, u64)>> = want.obj().unwrap().iter()
        .map(|(k, v)| (k.clone(), v.arr().unwrap().iter()
            .map(|r| (r.arr().unwrap()[0].int().unwrap() as u64, r.arr().unwrap()[1].int().unwrap() as u64)).collect()))
        .collect();
    assert_eq!(plan(&units), want);
}

#[test]
fn whole_widths_and_layer_choice() {
    let units = fetch::units(&format!("{D}/slices"), 4, 1, &Choice::Widths(vec![3]), Some(&[4])).unwrap();
    // rank 1 owns 8 of layer 4's 2 x 16 halves: 8 units x 3 projections
    assert_eq!(units.len(), 24);
    assert!(units.iter().all(|u| u.layer == 4 && u.k == 3 && u.nbytes == 3 * 98304));
}

#[test]
fn refusals_name_the_cause() {
    let choice = Choice::manifest(&format!("{D}/manifest.json")).unwrap();
    let e = fetch::units(&format!("{D}/slices"), 4, 2, &choice, None).err().unwrap();
    assert!(e.contains("no index slices for TP 4 rank 2"), "{e}");
    let e = fetch::units(&format!("{D}/slices"), 4, 1, &choice, Some(&[9])).err().unwrap();
    assert!(e.contains("layer 9"), "{e}");
}
