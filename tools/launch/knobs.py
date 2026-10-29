"""The knob table through the Rust library (engine/knobs-rs), with ctypes: the engine's knobs and the launcher's options,
both from the YAML files in knobs/. Python standard library only.

The library is $GR8_KNOBS_LIB, else the crate's own build (rebuilt with cargo when a knobs/ file is newer), else
bin/lib/libgr8knobs.so beside the engine, as scripts/build.sh and the image leave it."""
import ctypes
import json
import os
import shutil
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
CRATE = os.path.join(REPO, 'engine', 'knobs-rs')
LIB = 'libgr8knobs.dylib' if sys.platform == 'darwin' else 'libgr8knobs.so'


class KnobsError(Exception):
    pass


def _newest_yaml():
    d = os.path.join(REPO, 'knobs')
    return max((os.path.getmtime(os.path.join(d, f)) for f in os.listdir(d) if f.endswith('.yaml')), default=0)


def library():
    """The path of the library to load, building the crate's when it is missing or older than knobs/."""
    if os.environ.get('GR8_KNOBS_LIB'):
        return os.environ['GR8_KNOBS_LIB']
    built = os.path.join(CRATE, 'target', 'release', LIB)
    shipped = os.path.join(REPO, 'bin', 'lib', LIB)
    fresh = os.path.isfile(built) and os.path.getmtime(built) >= _newest_yaml()
    if not fresh and shutil.which('cargo'):
        r = subprocess.run(['cargo', 'build', '--release', '--quiet', '--features', 'capi', '--manifest-path',
                            os.path.join(CRATE, 'Cargo.toml')], capture_output=True, text=True)
        if r.returncode:
            raise KnobsError(f'cargo build of engine/knobs-rs failed:\n{r.stderr.strip()}')
        return built
    for p in (built, shipped):
        if os.path.isfile(p):
            return p
    raise KnobsError(f'no {LIB}: build it with `cargo build --release --features capi --manifest-path '
                     'engine/knobs-rs/Cargo.toml`, or `./gr8 <model> build`')


def _load():
    lib = ctypes.CDLL(library())
    lib.gr8_knobs_table_json.restype = ctypes.c_char_p
    return json.loads(lib.gr8_knobs_table_json().decode())


_TABLE = None


def table():
    """{'knobs': [{name, var, family, scope, kind, default, unset, values, legacy, doc, file}, ...],
    'options': [{for, key, scope, default, flag, doc}, ...]}, loaded once."""
    global _TABLE
    if _TABLE is None:
        _TABLE = _load()
    return _TABLE


def options(command):
    """The options of `command` (launch: ./gr8 <model>, kv: ./gr8 kv) in file order, {key: option}."""
    return {o['key']: o for o in table()['options'] if o['for'] == command}


def engine_names():
    """{variable name: knob} for every name a knob is read under, its old names included. A family's names end in
    its number, so this holds its stem."""
    out = {}
    for k in table()['knobs']:
        for n in [k['var']] + k['legacy']:
            out[n] = k
    return out


def knob_of(var):
    """The knob a variable name sets, or None."""
    names = engine_names()
    if var in names:
        return names[var]
    stem = var.rstrip('0123456789')
    k = names.get(stem)
    return k if k and k['family'] and stem != var else None
