# Prefix-cache namespaces

A namespace is an opaque, fixed-size **70-byte** identity. The pool compares every byte before reusing a retained
prefix. Callers define the bytes; the allocator does not parse model, profile, scale or request fields.
`STOCK_NAMESPACE` (Rust), `STOCK_CACHE_NAMESPACE` (C++) and an all-zero C buffer select existing stock behavior.

This foundation exposes no new request fields or launch options. Existing serving callers use zero by default.
Mapping a request's identity into a namespace belongs at the runtime boundary in a later change.

## Rust and C++

Rust's `retain_namespaced` and `match_prefix_namespaced` take a `Namespace` array. The original `retain` and
`match_prefix` methods are stock wrappers. For admission with stores enabled, call `count_namespaced` before the
following reserve/take; a nonzero namespace clears pending stock admission metadata and counts toward no stock
shared-prefix promotion. It returns zero. Stock `count` behavior is unchanged.

The C++ `g6rt::CacheNamespace` is the same-size `std::array`. `PrefixReuse::begin` and `resume` accept an optional
namespace, remember it, and treat a change as a cache miss. `clear` resets it to zero. An omitted namespace means
stock. `g6s2::RunPool::match` and `count` likewise accept a final optional namespace. `retain(run, finished)` uses the
namespace carried by the finished `PrefixReuse`; a third explicit namespace can be supplied when the caller tracks
identity outside that object. The caller must label the actual cached state consistently and use a hit only for the
lookup that produced it.

## C ABI

`g6kv_retain`, `g6kv_match` and `g6kv_count` retain their original signatures and stock semantics. New symbols are:

- `g6kv_retain_namespaced`: retain a finished run under the supplied identity.
- `g6kv_match_namespaced`: search only the supplied identity.
- `g6kv_count_namespaced`: prepare admission metadata; return a status code and write the count through `out`.

Each takes a namespace pointer and byte count. The pointer must reference exactly `G6KV_NAMESPACE_BYTES` readable
bytes; the bytes are copied during the call. Null buffers, incorrect lengths and null required output pointers return
`BadArgument` without changing the pool or output. As with the existing ABI, nonnull pool/run/data/output pointers
must refer to valid objects. The C++ adapter links the new symbols, so build it with the updated Rust static library.

## Stored tiers

An evicted non-stock run is stored whole as an evicted entry that carries its namespace (`stash::Entry::namespace`),
and only a lookup with the same 70 bytes can hit it (memory, file and remote block stores alike). A non-stock run
never promotes a shared prefix, never gets or becomes a prefix base (no dedupe), and never leaves a stock
continuation head. Prefix kinds, and so the persistent index, stay stock-only: its format is unchanged, and a
restart keeps no non-stock entry. The direct prototype `Tier::spill` still refuses non-stock runs. A stock lookup
never sees a non-stock entry, and a non-stock entry folds all its namespace bytes into the replicated digest.

Non-stock requests must use namespace-aware admission counting so their live metadata is excluded from stock
promotion. The pool never infers a namespace from token IDs. Equal tokens alone do not establish equal cache state.

Stock retained entries keep the existing replicated-state digest exactly. Nonzero retained namespaces contribute
all their bytes to that digest. This digest is an agreement check, not a cryptographic authentication mechanism.
It does not add a namespace field to the scheduler's replicated control protocol; request/grant transport is separate.

## CPU tests

```sh
cargo test --release --manifest-path engine/kvman-rs/Cargo.toml
scripts/host_tests.sh
```

Rust tests cover exact-byte matching, digests, LRU selection, stock wrappers, promotion/counting, persistent reopen,
isolation of stock prefix bases, and non-stock store/restore by namespace. The host suite runs `pool_namespace_test` against memory, an SSD-format file
and a loopback KV node when that binary is available. It tests both original and new C symbols, malformed buffer
lengths/null pointers, C++ prefix state, resident misses, and stock and non-stock spill/restore bytes.
