//! GLM 5.3's page block, the layout the benches and the simulator model. The engine sizes its stores from its own
//! bindings (g6kv_layout_sizes); these are the same numbers written out for tools that run without an engine.
//!
//! Per layer and page the MLA cache holds 64 rows x 656 B (512 FP8 latent + scales + 64 BF16 rope; 416 B in an engine
//! built with G6_KV4=1: 256 B of INT4 latent, 16 FP16 scales, the rope), and the full-index layers also hold the
//! indexer's keys, 64 rows x 132 B (128 FP8 + scale), engine/attn/kvfmt.h. There are 78 layers plus the MTP layer, and
//! the full-index layers are 0, 1, 2, 6, 10, .., 74 (rank.cuh cp_full_index) and the MTP layer: 79 MLA caches and 22
//! index caches, 3,502,592 B a page, 856 x 4 KiB on disk.
#![allow(dead_code)] // each tool takes what it models

use g6kvman::layout::{sizes, Binding};

pub const MLA_ROW: usize = 656;
pub const MLA_ROW_KV4: usize = 416;
const IDX_ROW: usize = 132;
const LAYERS: usize = 78;
/// One token's MLA rows of one indexer group: a full-index layer's top-2,048 list serves the three layers after it, so
/// a sparse-attention step reads, per selected token and group, 4 layers x 656 B.
pub const ROW_BYTES: usize = 4 * MLA_ROW;

/// The main model's layers (sharded past R), then the MTP layer.
pub fn bindings(mla_row: usize) -> Vec<Binding> {
    (0..=LAYERS)
        .map(|l| {
            let index = l < 3 || l == LAYERS || (l >= 6 && l <= 74 && l % 4 == 2);
            Binding {
                latent: (64 * mla_row) as u32,
                index: if index { 64 * IDX_ROW as u32 } else { 0 },
                cp: (l < LAYERS) as u32,
            }
        })
        .collect()
}

/// One page block's slot for an MLA row of `mla_row` bytes.
pub fn page_slot(mla_row: usize) -> usize {
    sizes(&bindings(mla_row)).slot as usize
}
