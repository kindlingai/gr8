//! The tools' GLM 5.3 layout (src/bin/glm53) through layout::sizes gives the engine's page blocks: PageBlocks
//! (engine/runtime/s2/kv_tier.cuh) passes its bindings to the same function.

#[path = "../src/bin/glm53/mod.rs"]
mod glm53;

use g6kvman::layout::sizes;

#[test]
fn glm53_page_blocks() {
    let fp8 = glm53::bindings(glm53::MLA_ROW);
    assert_eq!((fp8.len(), fp8.iter().filter(|b| b.index != 0).count()), (79, 22));
    let s = sizes(&fp8);
    assert_eq!((s.block, s.slot), (3_502_592, 856 * 4096));
    // G6_KV4=1: 79 x 64 x 416 + 22 x 64 x 132
    let s = sizes(&glm53::bindings(glm53::MLA_ROW_KV4));
    assert_eq!((s.block, s.slot), (2_289_152, 559 * 4096));
    assert_eq!(glm53::page_slot(glm53::MLA_ROW_KV4), 559 * 4096);
    assert_eq!(glm53::ROW_BYTES, 2_624);
}
