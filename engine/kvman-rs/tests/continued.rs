use g6kvman::pool::STOCK_NAMESPACE;
use g6kvman::{Finished, Pool, Rules};

fn finished(prompt: &[i32]) -> Finished {
    Finished {
        history: prompt.to_vec(),
        valid: prompt.len() as i32 - 1,
        written_end: prompt.len() as i32,
        pf_end: prompt.len() as i32,
    }
}

// The next turn of a conversation continues its run; another conversation that shares only the system prompt does not.
#[test]
fn a_next_turn_continues_its_own_run_only() {
    let mut p = Pool::new(16, 16, 16 * 64).unwrap();
    p.keep(Rules { strict: false, noend: true, grid: 0 });
    let system: Vec<i32> = (0..64).collect();
    let turn = |agent: i32, len: i32| -> Vec<i32> {
        system.iter().copied().chain((0..len).map(|i| 1000 * agent + i)).collect()
    };
    let (a, b) = (turn(1, 64), turn(2, 64));
    let ra = p.reserve(128, 160).unwrap();
    p.retain(&ra, finished(&a)).unwrap();
    let rb = p.reserve(128, 160).unwrap();
    p.retain(&rb, finished(&b)).unwrap();
    let next_a: Vec<i32> = a.iter().copied().chain([7, 8, 9]).collect();
    let (age_a, age_b) = (p.continued_by(&next_a, STOCK_NAMESPACE), p.continued_by(&b, STOCK_NAMESPACE));
    assert!(age_a.is_some() && age_b.is_some() && age_a < age_b, "{age_a:?} {age_b:?}");
    assert_eq!(p.continued_by(&turn(3, 80), STOCK_NAMESPACE), None); // the system prompt alone
    let mut other = STOCK_NAMESPACE;
    other[0] = 1;
    assert_eq!(p.continued_by(&next_a, other), None);
}
