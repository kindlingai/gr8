// The typed statics and TABLE, from the YAML files in knobs/ at the repository root (knobs/README.md): build.rs reads
// them and writes the `table!` call below into $OUT_DIR/table.rs. Change the YAML, not the generated call.
//
// After a change, `cargo run --manifest-path engine/knobs-rs/Cargo.toml --bin knobs -- gen` writes gr8_knobs.h and
// docs/ENV.md.

use crate::{Family, Kind, Knob, OptKnob, Scope, Spec};

macro_rules! rt {
    (flag) => { bool };
    (int) => { i64 };
    (float) => { f64 };
    (text) => { &'static str };
    (path) => { &'static str };
    (ints) => { &'static [i64] };
    (floats) => { &'static [f64] };
}

macro_rules! knob_type {
    ($k:ident = $d:literal) => { Knob<rt!($k)> };
    ($k:ident ? $u:literal) => { OptKnob<rt!($k)> };
    ($k:ident ? $u:literal [$n:ident]) => { Family<rt!($k)> };
}

macro_rules! opt {
    () => {
        None
    };
    ($v:literal) => {
        Some($v)
    };
}

macro_rules! family {
    () => {
        false
    };
    ($n:ident) => {
        true
    };
}

macro_rules! table {
    ($($(#[doc = $doc:literal])* $name:ident $([$n:ident])? : $scope:ident $kind:ident $(= $d:literal)? $(? $u:literal)?,
       $values:literal $(, env $env:literal)?, legacy [$($legacy:literal),*];)*) => {
        #[allow(non_camel_case_types, clippy::upper_case_acronyms)]
        #[repr(usize)]
        enum Ix { $($name,)* }

        /// The table, sorted by name. A knob's index here is its id in the C API.
        pub static TABLE: &[Spec] = &[$(Spec {
            name: stringify!($name),
            family: family!($($n)?),
            scope: match stringify!($scope).as_bytes() {
                b"agree" => Scope::Agree,
                b"local" => Scope::Local,
                b"test" => Scope::Test,
                b"outside" => Scope::Outside,
                _ => panic!("scope"),
            },
            kind: match stringify!($kind).as_bytes() {
                b"flag" => Kind::Flag,
                b"int" => Kind::Int,
                b"float" => Kind::Float,
                b"text" => Kind::Text,
                b"path" => Kind::Path,
                b"ints" => Kind::Ints,
                b"floats" => Kind::Floats,
                _ => panic!("kind"),
            },
            default: opt!($($d)?),
            unset: concat!("" $(, $u)?),
            values: $values,
            env: opt!($($env)?),
            legacy: &[$($legacy),*],
            doc: concat!($($doc, "\n",)*),
        },)*];

        $($(#[doc = $doc])* pub static $name: knob_type!($kind $(= $d)? $(? $u)? $([$n])?) =
            <knob_type!($kind $(= $d)? $(? $u)? $([$n])?)>::new(Ix::$name as usize);)*
    };
}

include!(concat!(env!("OUT_DIR"), "/table.rs"));
