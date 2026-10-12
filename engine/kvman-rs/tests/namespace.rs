use g6kvman::pool::{Namespace, STOCK_NAMESPACE};
use g6kvman::{Finished, Pool, Rules, Why};

fn namespace(tag: u8) -> Namespace {
    let mut namespace = STOCK_NAMESPACE;
    namespace[0] = tag;
    namespace
}

#[test]
fn retained_prefixes_only_match_their_exact_namespace() {
    let mut pool = Pool::new(16, 16, 16 * 64).unwrap();
    pool.keep(Rules { strict: false, noend: true, grid: 0 });

    let prompt: Vec<i32> = (0..96).collect();
    let run = pool.reserve(96, 128).unwrap();
    pool.retain_namespaced(
        &run,
        Finished { history: prompt.clone(), valid: 95, written_end: 96, pf_end: 96 },
        namespace(1),
    )
    .unwrap();

    let exact = pool.match_prefix_namespaced(&prompt, 96, 128, 95, namespace(1));
    assert_eq!(exact.why, Why::Hit);
    assert!(exact.skip > 0);

    for wrong in [STOCK_NAMESPACE, namespace(2)] {
        let miss = pool.match_prefix_namespaced(&prompt, 96, 128, 95, wrong);
        assert_eq!(miss.skip, 0);
        assert_eq!(miss.why, Why::Namespace);
    }
}

fn finished(prompt: &[i32]) -> Finished {
    Finished {
        history: prompt.to_vec(),
        valid: prompt.len() as i32 - 1,
        written_end: prompt.len() as i32,
        pf_end: prompt.len() as i32,
    }
}

fn pool() -> Pool {
    let mut p = Pool::new(8, 8, 512).unwrap();
    p.keep(Rules { strict: false, noend: true, grid: 128 });
    p
}

#[test]
fn every_byte_participates_in_matching_and_digest() {
    let prompt: Vec<i32> = (0..192).collect();
    let mut stock = pool();
    let r = stock.reserve(192, 256).unwrap();
    stock.retain(&r, finished(&prompt)).unwrap();
    let baseline = stock.digest();
    for byte in 0..STOCK_NAMESPACE.len() {
        let mut ns = STOCK_NAMESPACE;
        ns[byte] = 1;
        assert_eq!(stock.match_prefix_namespaced(&prompt, 192, 256, 191, ns).skip, 0);
        let mut p = pool();
        let r = p.reserve(192, 256).unwrap();
        p.retain_namespaced(&r, finished(&prompt), ns).unwrap();
        assert_ne!(p.digest(), baseline, "byte {byte}");
        assert_eq!(p.match_prefix_namespaced(&prompt, 192, 256, 191, ns).skip, 128);
        assert_eq!(p.match_prefix(&prompt, 192, 256, 191).skip, 0);
    }
}

#[test]
fn a_matching_namespace_wins_over_a_newer_other_namespace() {
    let prompt: Vec<i32> = (0..192).collect();
    let mut p = pool();
    let first = p.reserve(192, 256).unwrap();
    p.retain_namespaced(&first, finished(&prompt), namespace(1)).unwrap();
    let second = p.reserve(192, 256).unwrap();
    p.retain_namespaced(&second, finished(&prompt), namespace(2)).unwrap();
    let hit = p.match_prefix_namespaced(&prompt, 192, 256, 191, namespace(1));
    let taken = p.take(hit.skip, hit.age, 192, 256).unwrap();
    assert_eq!(taken.first_page, first.first_page);
    p.release(&taken).unwrap();
    assert!(p.match_prefix_namespaced(&prompt, 192, 256, 191, namespace(2)).skip > 0);
}

#[test]
fn explicit_zero_namespace_equals_the_legacy_stock_trace() {
    let prompt: Vec<i32> = (0..192).collect();
    let mut old = pool();
    let mut new = pool();
    for with_store in [false, true] {
        if with_store {
            old.stores(2);
            new.stores(2);
            old.add_tier(16, g6kvman::stash::Keep::All, None).unwrap();
            new.add_tier(16, g6kvman::stash::Keep::All, None).unwrap();
        }
        assert_eq!(old.count(&prompt, Some(128)), new.count_namespaced(&prompt, Some(128), STOCK_NAMESPACE));
        let a = old.reserve(192, 256).unwrap();
        let b = new.reserve(192, 256).unwrap();
        assert_eq!(a, b);
        assert_eq!(old.digest(), new.digest());
        old.retain(&a, finished(&prompt)).unwrap();
        new.retain_namespaced(&b, finished(&prompt), STOCK_NAMESPACE).unwrap();
        assert_eq!(old.digest(), new.digest());
        assert_eq!(old.io(), new.io());
        old.io_done().unwrap();
        new.io_done().unwrap();
        assert_eq!(
            old.match_prefix(&prompt, 192, 256, 191),
            new.match_prefix_namespaced(&prompt, 192, 256, 191, STOCK_NAMESPACE)
        );
    }
}

#[test]
fn non_stock_admissions_do_not_count_or_leave_live_stock_metadata() {
    let prompt: Vec<i32> = (0..192).collect();
    let mut p = pool();
    p.stores(2);
    p.add_tier(16, g6kvman::stash::Keep::All, None).unwrap();
    // Even an abandoned prepared stock admission must not attach its metadata to a non-stock grant.
    assert_eq!(p.count(&prompt, Some(128)), 1);
    assert_eq!(p.count_namespaced(&prompt, Some(128), namespace(1)), 0);
    let run = p.reserve(192, 256).unwrap();
    assert_eq!(p.count(&prompt, Some(128)), 1, "non-stock live prompt cannot suppress stock counting");
    p.retain_namespaced(&run, finished(&prompt), namespace(1)).unwrap();
    assert_eq!(p.io_pending(), 0, "no shared promotion");
    let all = p.reserve(512, 512).unwrap();
    assert_eq!(p.io_pending(), 0, "no eviction spill");
    p.release(&all).unwrap();
    assert!(p.stash().unwrap().entries(0).is_empty());
    assert_eq!(p.count(&prompt, Some(128)), 1, "non-stock eviction cannot remember a stock continuation head");
}

#[test]
fn non_stock_prefixes_never_enter_or_restore_a_persistent_tier() {
    use g6kvman::stash::Keep;
    let dir = std::env::temp_dir().join(format!("g6kv-namespace-index-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let prompt: Vec<i32> = (0..192).collect();
    {
        let mut p = pool();
        p.stores(2);
        p.add_tier(16, Keep::All, Some((&dir, "namespace-test"))).unwrap();
        for tag in [1, 2, 3] {
            assert_eq!(p.count_namespaced(&prompt, Some(128), namespace(tag)), 0);
            let r = p.reserve(192, 256).unwrap();
            p.retain_namespaced(&r, finished(&prompt), namespace(tag)).unwrap();
            assert_eq!(p.io_pending(), 0);
            let all = p.reserve(512, 512).unwrap();
            assert_eq!(p.io_pending(), 0);
            p.release(&all).unwrap();
            p.io_done().unwrap();
            assert!(p.stash().unwrap().entries(0).is_empty());
        }
    }
    let mut reopened = pool();
    reopened.stores(2);
    reopened.add_tier(16, Keep::All, Some((&dir, "namespace-test"))).unwrap();
    for ns in [STOCK_NAMESPACE, namespace(1), namespace(2)] {
        let hit = reopened.match_prefix_namespaced(&prompt, 192, 256, 191, ns);
        assert_eq!(hit.skip, 0);
        assert!(!hit.stored);
    }
    std::fs::remove_dir_all(dir).unwrap();
}
