# zarrs-python

[![PyPI](https://img.shields.io/pypi/v/zarrs.svg)](https://pypi.org/project/zarrs)
[![Downloads](https://static.pepy.tech/badge/zarrs/month)](https://pepy.tech/project/zarrs)
[![Downloads](https://static.pepy.tech/badge/zarrs)](https://pepy.tech/project/zarrs)
[![Stars](https://img.shields.io/github/stars/zarrs/zarrs-python?style=flat&logo=github&color=yellow)](https://github.com/zarrs/zarrs-python/stargazers)
![CI](https://github.com/zarrs/zarrs-python/actions/workflows/ci.yml/badge.svg)
![CD](https://github.com/zarrs/zarrs-python/actions/workflows/cd.yml/badge.svg)

This project serves as a bridge between [`zarrs`](https://docs.rs/zarrs/latest/zarrs/) (Rust) and [`zarr`](https://zarr.readthedocs.io/en/latest/index.html) (`zarr-python`) via [`PyO3`](https://pyo3.rs/v0.22.3/).  The main goal of the project is to speed up i/o (see [`zarr_benchmarks`](https://github.com/LDeakin/zarr_benchmarks)).

To use the project, simply install our package `zarrs` from PyPI (which depends on `zarr-python>=3.0.0`), and run:

```python
import zarr
zarr.config.set({"codec_pipeline.path": "zarrs.ZarrsCodecPipeline"})
```

You can then use your `zarr` as normal (with some caveats)!

## API

We export a `ZarrsCodecPipeline` class so that `zarr-python` can use the class but it is not meant to be instantiated and we do not guarantee the stability of its API beyond what is required so that `zarr-python` can use it.  Therefore, it is not documented here.

At the moment, we only support a subset of the `zarr-python` stores:

- [`LocalStore`](https://zarr.readthedocs.io/en/latest/api/zarr/storage/#zarr.storage.LocalStore) (local filesystem)
- [`ObjectStore`](https://zarr.readthedocs.io/en/latest/user-guide/storage/#object-store) (cloud storage)
- [`HTTPFileSystem`](https://filesystem-spec.readthedocs.io/en/latest/api.html#fsspec.implementations.http.HTTPFileSystem) via [`FsspecStore`](https://zarr.readthedocs.io/en/latest/api/zarr/storage/#zarr.storage.FsspecStore)

A `NotImplementedError` will be raised if a store is not supported.

### Configuration

`ZarrsCodecPipeline` options are exposed through `zarr.config`.

Standard `zarr.config` options control some functionality (see the defaults in the [config.py](https://github.com/zarr-developers/zarr-python/blob/main/src/zarr/core/config.py) of `zarr-python`):
- `threading.max_workers`: the maximum number of threads used internally by the `ZarrsCodecPipeline` on the Rust side.
  - Defaults to the number of threads in the global `rayon` thread pool if set to `None`, which is [typically the number of logical CPUs](https://docs.rs/rayon/latest/rayon/struct.ThreadPoolBuilder.html#method.num_threads).
- `array.write_empty_chunks`: whether or not to store empty chunks.
  - Defaults to false if `None`. Note that checking for emptiness has some overhead, see [here](https://docs.rs/zarrs/latest/zarrs/config/struct.Config.html#store-empty-chunks) for more info.

The `ZarrsCodecPipeline` specific options are:
- `codec_pipeline.chunk_concurrent_maximum`: the maximum number of chunks stored/retrieved concurrently.
  - Defaults to the number of logical CPUs if `None`. It is constrained by `threading.max_workers` as well.
- `codec_pipeline.chunk_concurrent_minimum`: the minimum number of chunks retrieved/stored concurrently when balancing chunk/codec concurrency.
  - Defaults to 4 if `None`. See [here](https://docs.rs/zarrs/latest/zarrs/config/struct.Config.html#chunk-concurrent-minimum) for more info.
- `codec_pipeline.validate_checksums`: enable checksum validation (e.g. with the CRC32C codec).
  - Defaults to `True`. See [here](https://docs.rs/zarrs/latest/zarrs/config/struct.Config.html#validate-checksums) for more info.
- `codec_pipeline.file_handle_cache_size`: the capacity of the filesystem store's file handle cache. If nonzero, files are kept open in a least-recently-used cache and reused across partial reads instead of being reopened per read, which cuts `open`/`stat`/`close` operations when many byte ranges are read from the same files, such as partial reads of sharded arrays. This is particularly beneficial on network filesystems (e.g. Lustre, NFS), where each metadata operation is a server round trip.
  - Defaults to `0` (disabled). Only applies to filesystem stores, and has no effect when `direct_io` is enabled.
  - Cached handles are invalidated on writes through this pipeline, but not on modification from anywhere else — and `zarr-python` itself is such a writer, since `resize`, `delete_dir` and metadata writes go through its own store. A cached handle can then still read a chunk file that has been deleted. Only enable this while nothing is modifying the array.
  - The cache is per `Array` object, not per process, so compare `file_handle_cache_size` times the number of open arrays against `ulimit -n`. See [here](https://docs.rs/zarrs_filesystem/latest/zarrs_filesystem/struct.FilesystemStoreOptions.html#method.file_handle_cache_size) for more info.
- `codec_pipeline.shard_index_cache_size`: the number of shard indexes kept in a least-recently-used cache. If nonzero, a partial read of a shard reuses an index decoded by an earlier read instead of fetching and decoding it again, removing one read per shard per call. Matters most on network filesystems and for repeated small reads of the same shards, such as random-access sampling.
  - Defaults to `0` (disabled). Only sharded arrays have an index to cache. Each entry costs 16 bytes per inner chunk, so size it against the number of shards read repeatedly, not the array size.
  - Read-only by design: any write through this pipeline empties the cache, and modification from elsewhere is not seen — the same caveat as `file_handle_cache_size`. Only enable it while nothing else is modifying the array.
- `codec_pipeline.integer_array_indexing`: serve reads with a sorted integer-array index (`z[[3, 7, 8, 9], :]`, `z[:, cols]`, 1-D `z.vindex[idx]`) through `zarrs` instead of falling back to the `zarr-python` pipeline. A chunk read is described to `zarrs` as a rectangular subset, and such a selection is a stack of them, so the pipeline emits one per run of consecutive indices.
  - Defaults to `False`. Whether it helps depends on how dense the selection is: each run decodes separately, so several runs inside one inner chunk decode it once each where `zarr-python` decodes it once in total. That duplication is what `plan_reads` removes, so the two are normally enabled together. Measure before enabling it alone.
  - Reads only, and only when the indices are sorted and strictly increasing on a single axis. Unsorted or repeated indices, and two or more integer-array axes, still fall back, because `zarr-python` reorders the output for those and a run's position in the selection is then not its position in the output.
- `codec_pipeline.plan_reads`: read the shard index directly and issue the byte ranges of exactly the inner chunks the selection needs, instead of asking the sharding partial decoder for one region at a time. Each needed inner chunk is fetched and decoded once however many requested boxes fall inside it.
  - Defaults to `False`. Needs `integer_array_indexing` to be useful, and uses `shard_index_cache_size` for the parsed indexes.
  - Applies only to a chunk whose codec chain is exactly the sharding codec. Any codec wrapping it means the shard bytes are not laid out as the index describes, so the array falls through to the normal path. Unsharded arrays fall through too.
  - Units from every shard are issued into one pool, so reads overlap within a shard as well as across shards, and fetching and decoding run on separate pools. Sub-chunk reads are preserved when there is no compressor; a compressor makes the inner chunk atomic.
- `codec_pipeline.plan_reads_fetch_threads`: threads issuing reads for `plan_reads`. These block on IO, so they are their own pool rather than sharing the decode pool — a fetch waiting on the byte budget must not occupy the thread that would release it.
  - Defaults to `0`, meaning the size of the `rayon` pool. Raise it where per-request latency dominates. The pool is process-wide and sized on first use.
- `codec_pipeline.plan_reads_fetch_byte_budget`: ceiling on bytes fetched but not yet decoded. Without one, a read plan covering an array would fetch all of it before the first decode finished.
  - Defaults to `0`, meaning 256 MiB. A read larger than the whole budget is admitted on its own, so a small budget throttles rather than deadlocks.
- `codec_pipeline.plan_reads_merge_gap_bytes`: merge two inner-chunk reads into one request when the gap between them is at most this many bytes.
  - Defaults to `0`, which merges only reads that touch, so not a byte outside the request is read. That still collapses dense selections into single large reads, because a shard written in one pass lays its inner chunks down contiguously in index order.
  - Raising it reads unrequested bytes to save requests — a bandwidth-for-latency trade that depends on the storage, which is why it is a knob rather than a default. A merged read never spans more than one shard.
- `codec_pipeline.direct_io`: enable `O_DIRECT` read/write, needs support from the operating system (currently only Linux) and file system.
  - Defaults to `False`.
- `codec_pipeline.strict`: raise exceptions for unsupported operations instead of falling back to the default codec pipeline of `zarr-python`.
  - Defaults to `False`.

For example:
```python
zarr.config.set({
    "threading.max_workers": None,
    "array.write_empty_chunks": False,
    "codec_pipeline": {
        "path": "zarrs.ZarrsCodecPipeline",
        "validate_checksums": True,
        "chunk_concurrent_maximum": None,
        "chunk_concurrent_minimum": 4,
        "file_handle_cache_size": 0,
        "shard_index_cache_size": 0,
        "integer_array_indexing": False,
        "plan_reads": False,
        "plan_reads_merge_gap_bytes": 0,
        "plan_reads_fetch_threads": 0,
        "plan_reads_fetch_byte_budget": 0,
        "direct_io": False,
        "strict": False
    }
})
```

If the `ZarrsCodecPipeline` is pickled, and then un-pickled, and during that time one of `chunk_concurrent_minimum`, `chunk_concurrent_maximum`, or `num_threads` has changed, the newly un-pickled version will pick up the new value.  However, once a `ZarrsCodecPipeline` object has been instantiated, these values are then fixed.  This may change in the future as guidance from the `zarr` community becomes clear.

## Concurrency

Concurrency can be classified into two types:
- chunk (outer) concurrency: the number of chunks retrieved/stored concurrently.
  - This is chosen automatically based on various factors, such as the chunk size and codecs.
  - It is constrained between `codec_pipeline.chunk_concurrent_minimum` and `codec_pipeline.chunk_concurrent_maximum` for operations involving multiple chunks.
- codec (inner) concurrency: the number of threads encoding/decoding a chunk.
  - This is chosen automatically in combination with the chunk concurrency.

The product of the chunk and codec concurrency will approximately match `threading.max_workers`.

Chunk concurrency is typically favored because:
- parallel encoding/decoding can have a high overhead with some codecs, especially with small chunks, and
- it is advantageous to retrieve/store multiple chunks concurrently, especially with high latency stores.

`zarrs-python` will often favor codec concurrency with sharded arrays, as they are well suited to codec concurrency.

## Supported Indexing Methods

The following methods will trigger use with the old zarr-python pipeline:

1. Any `oindex` or `vindex` integer `np.ndarray` indexing with dimensionality >=3 i.e.,

   ```python
   arr[np.array([...]), :, np.array([...])]
   arr[np.array([...]), np.array([...]), np.array([...])]
   arr[np.array([...]), np.array([...]), np.array([...])] = ...
   arr.oindex[np.array([...]), np.array([...]), np.array([...])] = ...
   ```

2. Any `vindex` or `oindex` discontinuous integer `np.ndarray` indexing for writes in 2D

   ```python
   arr[np.array([0, 5]), :] = ...
   arr.oindex[np.array([0, 5]), :] = ...
   ```

3. `vindex` writes in 2D where both indexers are integer `np.ndarray` indices i.e.,

   ```python
   arr[np.array([...]), np.array([...])] = ...
   ```

4. Ellipsis indexing.  We have tested some, but others fail even with `zarr-python`'s default codec pipeline.  Thus for now we advise proceeding with caution here.

   ```python
   arr[0:10, ..., 0:5]
   ```


Furthermore, using anything except contiguous (i.e., slices or consecutive integer) `np.ndarray` for numeric data will fall back to the default `zarr-python` implementation.

Please file an issue if you believe we have more holes in our coverage than we are aware of or you wish to contribute!  For example, we have an [issue in zarrs for integer-array indexing](https://github.com/LDeakin/zarrs/issues/52) that would unblock a lot the use of the rust pipeline for that use-case (very useful for mini-batch training perhaps!).

Further, any codecs not supported by `zarrs` will also automatically fall back to the python implementation.
