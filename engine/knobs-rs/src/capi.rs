//! The C API behind gr8_knobs.h. Ids are table indexes. A getter returns 1 (or a length) when the knob has a value, set
//! or default, and 0 (or -1) when it has none. Values live as long as the process.

use std::ffi::{c_char, c_int, CString};
use std::sync::OnceLock;

use crate::{env, problems, table_hash, value, Value, TABLE};

fn get(id: u32, n: u32) -> Option<&'static Value> {
    let Some(s) = TABLE.get(id as usize) else {
        eprintln!("gr8: knob id {id} is not in the table: gr8_knobs.h and the library are from different trees");
        std::process::exit(2)
    };
    value(id as usize, if s.family { n } else { 0 })
}

/// Prints every problem of the environment to stderr and returns the number of errors. `table_hash` is the hash
/// gr8_knobs.h was written with: a header from another table is an error.
#[no_mangle]
pub extern "C" fn gr8_knobs_init(hash: u64) -> c_int {
    let mut errors = 0;
    if hash != table_hash() {
        eprintln!("gr8: gr8_knobs.h is from another knob table than the library: run `knobs gen` and rebuild");
        errors += 1;
    }
    for p in problems() {
        eprintln!("gr8: {}: {}", if p.error { "error" } else { "warning" }, p.text);
        errors += p.error as c_int;
    }
    errors
}

#[no_mangle]
pub extern "C" fn gr8_knobs_signature() -> *const c_char {
    static S: OnceLock<CString> = OnceLock::new();
    S.get_or_init(|| CString::new(env().signature()).unwrap()).as_ptr()
}

/// # Safety
/// `out` must be valid for one write.
#[no_mangle]
pub unsafe extern "C" fn gr8_knob_flag(id: u32, n: u32, out: *mut c_int) -> c_int {
    let Some(Value::Flag(b)) = get(id, n) else { return 0 };
    *out = *b as c_int;
    1
}

/// # Safety
/// `out` must be valid for one write.
#[no_mangle]
pub unsafe extern "C" fn gr8_knob_int(id: u32, n: u32, out: *mut i64) -> c_int {
    let Some(Value::Int(i)) = get(id, n) else { return 0 };
    *out = *i;
    1
}

/// # Safety
/// `out` must be valid for one write.
#[no_mangle]
pub unsafe extern "C" fn gr8_knob_float(id: u32, n: u32, out: *mut f64) -> c_int {
    let Some(Value::Float(f)) = get(id, n) else { return 0 };
    *out = *f;
    1
}

#[no_mangle]
pub extern "C" fn gr8_knob_text(id: u32, n: u32) -> *const c_char {
    match get(id, n) {
        Some(Value::Text(s)) => s.as_ptr(),
        _ => std::ptr::null(),
    }
}

/// # Safety
/// `out` must be valid for one write.
#[no_mangle]
pub unsafe extern "C" fn gr8_knob_ints(id: u32, n: u32, out: *mut *const i64) -> isize {
    let Some(Value::Ints(v)) = get(id, n) else { return -1 };
    *out = v.as_ptr();
    v.len() as isize
}

/// # Safety
/// `out` must be valid for one write.
#[no_mangle]
pub unsafe extern "C" fn gr8_knob_floats(id: u32, n: u32, out: *mut *const f64) -> isize {
    let Some(Value::Floats(v)) = get(id, n) else { return -1 };
    *out = v.as_ptr();
    v.len() as isize
}
