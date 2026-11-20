//! The engine build's description (module_info.h `G6ModuleInfo`), as the C++ fills it from the build's constants.

use std::ffi::{c_char, CStr};

/// module_info.h G6_MODULE_ABI
pub const MODULE_ABI: u32 = 3;
pub const MODEL_GLM53: u32 = 1;
pub const MODEL_GLM53F: u32 = 2;
pub const MODEL_DS41: u32 = 3;
pub const MODEL_INK2: u32 = 4;
pub const CAP_KV_STORE: u32 = 1 << 9;

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct ModuleInfo {
    pub abi: u32,
    pub model: u32,
    pub name: [c_char; 16],
    pub nr: i32,
    pub mx: i32,
    pub s2n: i32,
    pub max_sessions: i32,
    pub hidden: i32,
    pub layers: i32,
    pub mtp_layers: i32,
    pub vocab: i32,
    pub experts: i32,
    pub topk: i32,
    pub kv_row_bytes: i32,
    pub kv_page: i32,
    pub kv_format: [c_char; 16],
    pub caps: u32,
    pub dfl_block: i32,
    pub pack_members: i32,
    pub dfl_drafts: i32,
    pub dfl_rep: i32,
    pub tp_sort: i32,
    pub kv4: i32,
    pub xgrammar: i32,
    pub ds_markov_maxs: i32,
    pub ds_block: i32,
}

fn name(b: &[c_char; 16]) -> String {
    let bytes: Vec<u8> = b.iter().take_while(|&&c| c != 0).map(|&c| c as u8).collect();
    String::from_utf8_lossy(&bytes).into_owned()
}

impl ModuleInfo {
    pub fn name(&self) -> String {
        name(&self.name)
    }
    pub fn kv_format(&self) -> String {
        name(&self.kv_format)
    }
}

/// A NUL-terminated array field as a C string, for tests that fill one.
pub fn set_name(dst: &mut [c_char; 16], s: &CStr) {
    *dst = [0; 16];
    for (d, &b) in dst.iter_mut().zip(s.to_bytes().iter().take(15)) {
        *d = b as c_char;
    }
}
