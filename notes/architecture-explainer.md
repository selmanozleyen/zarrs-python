# zarrs, structurally — why the types are shaped the way they are

For someone who knows what read amplification is, can read Rust, and wants to know *why*
`zarrs` is carved up the way it is: which types exist, which function names matter, and where
the design is genuinely odd. Opinionated on purpose — the "this is weird" notes are marked as
such and are my reading, not the authors'.

---

## 1. Eleven crates, and the reason is the plugin registry

```
zarrs                   the Array API + every built-in codec implementation
├── zarrs_codec         the codec TRAITS, and ArrayBytes
├── zarrs_chunk_grid    ArraySubset, chunk grids, Indexer
├── zarrs_storage       store traits, ByteRange
├── zarrs_metadata      zarr.json parsing
├── zarrs_metadata_ext  extension metadata (sharding, blosc config, ...)
├── zarrs_data_type     data types and their byte representation
├── zarrs_plugin        the extension registry
├── zarrs_chunk_key_encoding   "c/0/1" naming
├── zarrs_filesystem    a directory-backed store
└── zarrs_conformance   spec conformance test binary
```

The split is not organisational tidiness. Zarr version 3 is explicitly extensible: third
parties define new codecs, data types, chunk grids. If the traits lived in `zarrs`, every
extension crate would depend on all of `zarrs` — including every codec implementation — and
you would get a dependency cycle the moment `zarrs` wanted to use one. Putting the *interfaces*
in leaf crates (`zarrs_codec`, `zarrs_data_type`, `zarrs_chunk_grid`) means an extension
depends only on the interface it implements.

**Practical consequence when reading the code:** a name you cannot find in `zarrs` is usually a
re-export. `zarrs/src/array.rs` has long `pub use` blocks pulling `ArraySubset`,
`ChunkShapeTraits`, `ravel_indices` and friends up from the leaf crates, so `zarrs::array::X`
and `zarrs_chunk_grid::X` are the same type. Grep both.

---

## 2. The bound/unbound codec split — the most distinctive design here

There are two codec chain types:

```rust
pub struct CodecChain {                              // codec_chain.rs:85
    array_to_array: Vec<Arc<dyn UnboundArrayToArrayCodecTraits>>,
    array_to_bytes: Arc<dyn UnboundArrayToBytesCodecTraits>,
    bytes_to_bytes: Vec<Arc<dyn BytesToBytesCodecTraits>>,
}

pub struct CodecChainBound { ... }                   // codec_chain.rs:93
```

`CodecChain` is what the JSON said. `CodecChainBound` is that chain **specialised to a concrete
data type and fill value**. The conversion is one call:

```rust
CodecChain::with_context(data_type, fill_value) -> Arc<CodecChainBound>   // codec_chain.rs:105
```

The reason is in its body, and it is a better reason than it first looks:

```rust
for codec in &self.array_to_array {
    let bound = codec.with_context(data_type.clone(), fill_value.clone())?;
    data_type  = bound.encoded_data_type().clone();     // <-- the type CHANGED
    fill_value = bound.encoded_fill_value().clone();
    array_to_array.push(bound);
}
```

Codecs can change the element type. A "fixed scale offset" codec takes `float32` in and stores
`int16`. A transpose changes the layout. So the chain is only meaningful once you thread a
concrete type through it, front to back, each codec binding against the *previous codec's
output* type. Only after binding does a codec know its element size — and therefore how to
compute a byte offset, which is the thing every partial read needs.

**Why you care:** anything that wants to ask "is this chain just raw bytes?" must ask the
*bound* chain, and the bound types are mostly private. `BytesCodecBound` is declared
`struct BytesCodecBound` with no `pub` (`bytes_codec.rs:38`), and the bound trait carries no
name method, so you cannot downcast to it or ask it what it is from outside the crate. That is
a real hole: the unbound side has `name()` via `ExtensionName`, the bound side has nothing.

**Weird, in my reading:** the asymmetry. Unbound codecs are introspectable (`name`,
`configuration`), bound ones are opaque. Since binding is mandatory before any read, the useful
half is the one you cannot query.

---

## 3. `ArrayBytes` — one enum for three memory layouts

```rust
pub enum ArrayBytes<'a> {                            // zarrs_codec/src/array_bytes.rs:32
    Fixed(ArrayBytesRaw<'a>),                        // C-contiguous, last axis fastest
    Variable(ArrayBytesVariableLength<'a>),          // bytes + per-element offsets
    Optional(...),                                   // data + validity mask
}
```

Every codec's `decode` returns this, so the type system carries "what shape is this data,
really" rather than an implicit contract. `Fixed` is the fast case and the one this project
lives in. `Variable` is strings and ragged data. `Optional` is null-able data with a mask,
which is Arrow-shaped thinking.

The `'a` lifetime is there so a decode can *borrow* rather than copy: internally these carry
`Cow<'a, [u8]>` (copy-on-write — either a borrow or an owned buffer, decided at runtime). A
codec that does not change the bytes can hand back a borrow of its input.

**Nice:** the `Cow` means a no-op codec costs nothing.
**Weird:** it makes signatures noisy, and every consumer has to match on three variants even
when two are impossible for their data type. Most call sites immediately do the equivalent of
"give me `Fixed` or error".

---

## 4. `ArraySubset` — a rectangle, and the type most worth knowing

```rust
pub struct ArraySubset { start: ArrayIndices, shape: ArrayShape }   // both Vec<u64>
```

"A rectangular region of an array", stored as a corner plus extents. Nearly every read verb
takes one. The methods worth knowing:

| method | what it gives you |
|---|---|
| `new_with_ranges(&[0..4, 2..7])` | build from ranges |
| `new_with_start_shape(start, shape)` | build from corner + size |
| `contiguous_indices(array_shape)` | the runs of consecutive elements this region covers |
| `contiguous_linearised_indices(array_shape)` | the same, as flat `(offset, length)` pairs |
| `overlap(other)`, `bound(end)` | intersection, clamping |
| `inbounds_shape(array_shape)` | does it fit |

`contiguous_linearised_indices` is the important one and worth understanding once, because it
encodes the central fact of row-major layout:

```
   subset = rows 1..3, cols 2..5   of a 4 x 8 array

   row 0  . . . . . . . .
   row 1  . . X X X . . .     the subset is NOT contiguous in memory
   row 2  . . X X X . . .     it is 2 runs of 3 elements,
   row 3  . . . . . . . .     at flat offsets 10 and 18

   contiguous_linearised_indices -> [(10, 3), (18, 3)]
```

And the merge rule that makes whole-trailing-axes cheap: walking axes in reverse, a run keeps
growing while an axis is taken *whole* (`start == 0 && size == extent`). So a subset spanning
every column collapses to a single run per row — or one run total if it spans every row too.
That single rule is why "take all columns" is fast and "take some columns" is not.

---

## 5. Partial decoders — the mechanism that makes sharding worth anything

A normal decode is "give me these bytes, get the whole chunk". A **partial decoder** is a
stateful object that can serve *pieces* of a chunk without materialising all of it:

```rust
pub trait BytesPartialDecoderTraits   // zarrs_codec/src/codec_traits/bytes_partial_sync.rs:17
pub trait ArrayPartialDecoderTraits   // .../array_partial_sync.rs:66
```

Two levels because the codec chain has two halves. Below the array-to-bytes hinge you are
slicing *bytes*; above it you are slicing *elements*.

For a sharding codec, `ShardingPartialDecoder` is the thing that reads the shard index once,
keeps it, and then serves inner chunks by byte range. Constructing one per shard and reusing it
is precisely the "shard index cache" this project's read path depends on — without it, every
inner-chunk read re-reads and re-parses the index.

**Weird, and it bit this project:** the partial decoder for a coordinate list
(`partial_decode_fixed_indexer`) is a *serial per-element loop* — two allocations and a decoder
lookup per element, measured elsewhere in these notes at 640 nanoseconds per element. So the
API that looks like it was made for "read these scattered rows" is the slowest way to do it,
and the fast path is instead: decode whole inner chunks, gather rows yourself. That is
non-obvious and worth knowing before you reach for it.

---

## 6. `ArrayBytesFixedDisjointView` — the write-side counterpart, and why it is `unsafe`

```rust
pub struct ArrayBytesFixedDisjointView<'a> {
    bytes: UnsafeCellSlice<'a, u8>,     // aliasable handle to someone else's buffer
    shape: &'a [u64],                   // the WHOLE array's shape
    subset: ArraySubset,                // the region THIS view owns
    contiguous_indices: ContiguousIndices,           // precomputed run layout
    contiguous_linearised_indices: ContiguousLinearisedIndices,
}
```

It exists because decoding fills a *sub-box of a larger output*. Handing a codec a bare
`&mut [u8]` would lose the information that rows land at the array's stride, not end to end.

`new` and `subdivide` are `unsafe`, with the obligation stated on the type: *"the `subset`
represented by this view must not overlap with the `subset` of any other created views that
reference the same array bytes."* Several chunks decode in parallel into disjoint regions of
one buffer; Rust's borrow checker cannot prove the regions are disjoint, so the invariant is
carried as a documented promise instead.

**This is the right call.** The alternative is every codec doing raw pointer arithmetic with
its own ad-hoc argument. Here the unsoundness is concentrated in one type with one invariant.

**But:** there is no safe constructor for the single-view case. If you own a buffer and want to
decode one subset into all of it, you still go through an `unsafe` whose invariant you satisfy
vacuously — there is no other view. A `TryFrom<&mut [u8]>` for that case would let a whole
class of caller contain no `unsafe` at all.

---

## 7. The read verbs, and how to pick one

`Array` has a large, regular vocabulary. The pattern is
`retrieve_<what>[_if_exists|_into|_at_level]`:

```
retrieve_array_subset          a rectangle of the array          -> ArrayBytes
retrieve_array_subset_into     ... into a buffer you supply      -> ()
retrieve_chunk                 one whole chunk
retrieve_chunk_subset          part of one chunk
retrieve_chunks                several whole chunks
retrieve_subchunk              one INNER chunk of a shard
retrieve_subchunk_at_level     ... at a given nesting depth
retrieve_encoded_chunk         the bytes, still compressed
```

Reading these names: **"chunk" is the chunk grid's unit; "subchunk" is inside a shard**. The
`_encoded_` verbs stop before decoding, which is what lets a caller move the decode to another
thread. The `_into` verbs are the ones that write into your memory rather than allocating —
they take an `ArrayBytesDecodeIntoTarget`, which wraps the disjoint view from section 6.

**Weird:** `_into` exists at array level in some versions and not others — it is present in the
0.23 line and absent from the 0.24 development tree I compared against. If you are targeting a
specific revision, check rather than assume.

---

## 8. Where the design is genuinely awkward

Collected, so you can decide for yourself:

1. **Bound codecs are opaque.** You must bind before reading, and after binding you cannot ask
   what a codec is. Predicates like "is this chain plain bytes?" have to be answered from the
   metadata JSON instead — which means string-matching codec names and duplicating the
   registry's alias handling.
2. **The indexer path is a trap.** The API shaped like "read scattered elements" is the slow
   one. Nothing in the type system or docs warns you.
3. **`ArrayBytes` forces a three-way match** on every consumer, including ones whose data type
   can only ever be `Fixed`.
4. **Sharding is a codec, which is elegant and confusing.** A shard's inner chunk grid is
   reachable only through the sharding codec, so anything wanting "the real decode unit" has to
   downcast to `ShardingCodecBound` and walk it — and nested sharding means walking it
   repeatedly. There is no `array.decode_unit_shape()`.
5. **Sync and async are duplicated wholesale.** `array_sync_readable.rs` and
   `array_async_readable.rs`, `BytesPartialDecoderTraits` and `AsyncBytesPartialDecoderTraits`,
   and so on. Idiomatic for Rust today — there is no good way to be generic over async-ness —
   but it doubles the surface you have to read, and the two halves can drift.

None of these are mistakes exactly. They are the cost of an extensible format with a
strongly-typed implementation, and mostly they buy something. But knowing them ahead of time
saves an afternoon each.

---

## 9. Reading order

1. `zarrs_chunk_grid/src/array_subset.rs` — small, self-contained, and the vocabulary
   everything else uses.
2. `zarrs_codec/src/array_bytes.rs` — what a decode actually returns.
3. `zarrs/src/array/codec/array_to_bytes/codec_chain.rs` — the bind step, section 2.
4. `zarrs/src/array/codec/array_to_bytes/sharding/sharding_partial_decoder_sync.rs` — how a
   shard serves one inner chunk.
5. `zarrs_codec/src/array_bytes_fixed_disjoint_view.rs` — the write side and its `unsafe`.

Then `zarrs-python`'s `src/read_decode.rs`, which uses all five.
