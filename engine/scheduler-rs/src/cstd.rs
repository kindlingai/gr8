//! The C library's number parsing and formatting, for the values and strings that must match the C++ they replaced
//! byte for byte (atoi's and strtoul's edge cases, printf's %f of any double).

use std::ffi::{c_char, c_double, c_int, c_long, c_ulong, c_ulonglong, c_void, CStr, CString};

extern "C" {
    fn snprintf(s: *mut c_char, n: usize, format: *const c_char, ...) -> c_int;
    fn atoi(s: *const c_char) -> c_int;
    fn atof(s: *const c_char) -> c_double;
    fn strtol(s: *const c_char, end: *mut *mut c_char, base: c_int) -> c_long;
    fn strtoul(s: *const c_char, end: *mut *mut c_char, base: c_int) -> c_ulong;
    fn strtoull(s: *const c_char, end: *mut *mut c_char, base: c_int) -> c_ulonglong;
    fn fopen(path: *const c_char, mode: *const c_char) -> *mut c_void;
    fn fscanf(f: *mut c_void, format: *const c_char, ...) -> c_int;
    fn fclose(f: *mut c_void) -> c_int;
}

/// A C string of the bytes up to the first NUL, where C would stop anyway (argv and knob values hold none).
pub fn cs(s: &[u8]) -> CString {
    let end = s.iter().position(|&b| b == 0).unwrap_or(s.len());
    CString::new(&s[..end]).unwrap_or_default()
}

fn printf_double(format: &CStr, x: f64) -> String {
    // %f of the largest double is 317 characters
    let mut b = [0 as c_char; 512];
    // SAFETY: b holds b.len() bytes and snprintf writes at most that many, NUL included
    let n = unsafe { snprintf(b.as_mut_ptr(), b.len(), format.as_ptr(), x as c_double) };
    let n = (n.max(0) as usize).min(b.len() - 1);
    // SAFETY: snprintf wrote n bytes
    let bytes = unsafe { std::slice::from_raw_parts(b.as_ptr() as *const u8, n) };
    String::from_utf8_lossy(bytes).into_owned()
}

/// std::to_string(double), and of a float (promoted)
pub fn to_string_f(x: f64) -> String {
    printf_double(c"%f", x)
}

pub fn c_atoi(s: &[u8]) -> i32 {
    let c = cs(s);
    // SAFETY: c is NUL-terminated
    unsafe { atoi(c.as_ptr()) }
}

pub fn c_atof(s: &[u8]) -> f64 {
    let c = cs(s);
    // SAFETY: c is NUL-terminated
    unsafe { atof(c.as_ptr()) }
}

pub fn c_strtoul10(s: &[u8]) -> u64 {
    let c = cs(s);
    // SAFETY: c is NUL-terminated; a null end pointer is allowed
    unsafe { strtoul(c.as_ptr(), std::ptr::null_mut(), 10) as u64 }
}

pub fn c_strtoull10(s: &[u8]) -> u64 {
    let c = cs(s);
    // SAFETY: as above
    unsafe { strtoull(c.as_ptr(), std::ptr::null_mut(), 10) as u64 }
}

/// strtol base 10: the value and whether the whole string was read
pub fn c_strtol10(s: &[u8]) -> (i64, bool) {
    let c = cs(s);
    let mut end: *mut c_char = std::ptr::null_mut();
    // SAFETY: c is NUL-terminated and end is written with a pointer into it
    let v = unsafe { strtol(c.as_ptr(), &mut end, 10) };
    // SAFETY: end points into c, at its NUL at the latest
    (v as i64, unsafe { *end } == 0)
}

/// The integers of a file, comma or white space separated, read as rank_host.h parse_ints reads them (fscanf
/// "%d%*[, \n\t]"). None when the file cannot be opened.
pub fn read_ints(path: &[u8]) -> Option<Vec<i32>> {
    let p = cs(path);
    // SAFETY: both strings are NUL-terminated
    let f = unsafe { fopen(p.as_ptr(), c"r".as_ptr()) };
    if f.is_null() {
        return None;
    }
    let mut v = Vec::new();
    let mut x: c_int = 0;
    // SAFETY: f is open and the format writes one int to x
    while unsafe { fscanf(f, c"%d%*[, \n\t]".as_ptr(), &mut x as *mut c_int) } == 1 {
        v.push(x);
    }
    // SAFETY: f is open
    unsafe { fclose(f) };
    Some(v)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn printf_forms() {
        assert_eq!(to_string_f(0.5), "0.500000");
        assert_eq!(to_string_f(-0.0), "-0.000000");
        assert_eq!(to_string_f(0.0000005), "0.000000");
        assert_eq!(to_string_f(1e300).len(), 301 + 7);
        assert_eq!(c_atoi(b" 12x"), 12);
        assert_eq!(c_strtol10(b"12x"), (12, false));
        assert_eq!(cs(b"a\0b").as_bytes(), b"a");
    }
}
