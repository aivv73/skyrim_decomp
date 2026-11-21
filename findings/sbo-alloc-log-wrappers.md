# BNet-runtime SBO, custom-allocator, buffer-clear, and guarded-log wrappers

On the pinned Steam build, fifteen parameterized C++ templates covering four cohesive runtime patterns in the same custom-allocator and guarded-logging subsystem (small-buffer-optimization functor binders/cloners, guarded console-and-sink log/JSON extractors, custom-allocator factory wrappers including `std::shared_ptr`-style atomic refcount copy-construction, and custom-allocator buffer clear/destroy/reset wrappers) produced 490 candidates totaling 79,270 bytes outside the matched inventory. Each compiled function passed the local comparator, including unwind information, static storage references, read-only table/format addresses, indirect function-pointer calls, and bound direct call targets.

| Template | Function observed | Local passes | Bytes | Shards | Distinct alloc sizes / strides | Distinct callees / cleanups | Distinct read-only tables / strings | Already in the published target-hash list |
| --- | --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |
| `SboPayloadBinders` | Small-buffer vs logged-heap allocation of a 16-byte polymorphic payload wrapper returning `(instance, cleanup)` | 104 | 25,064 | 6 | 1 size | 208 | 209 | 0 |
| `SboEmptyBinders` | Small-buffer vs logged-heap allocation of a payload-free polymorphic wrapper returning `(instance, cleanup)` | 50 | 9,500 | 3 | 1 size | 100 | 51 | 0 |
| `SboPayloadCloners` | Unconditional heap allocation and initialization of a 16-byte polymorphic payload wrapper | 34 | 1,802 | 2 | 1 size | 0 | 68 | 0 |
| `GuardedLogEmitters` | Condition-guarded console and sink log emission (`if (!ok)` emit filtered log with message argument) | 76 | 9,348 | 4 | — | 0 | 1 | 0 |
| `NullCheckedLogForwarders` | Null-checked string forwarder emitting a guarded console/sink log on null or forwarding `(ctx, str, strlen(str))` | 15 | 2,595 | 1 | — | 17 | 2 | 0 |
| `JsonFieldExtractors` | JSON field lookup and decode with missing, type-mismatch, and fallback error formatting, error codes, and guarded console/sink logs | 30 | 14,070 | 2 | — | 32 | 5 | 0 |
| `JsonCallbackExtractors` | JSON field lookup invoking a callback and checking error state, with error formatting and guarded console/sink logging on failure | 5 | 1,285 | 1 | — | 2 | 2 | 0 |
| `AllocSharedConstruct4Wrappers` | Custom-allocator allocation, atomic refcount copy-by-value of a two-pointer shared handle, and 4-argument constructor call (direct or dereferenced `a2`) | 71 | 7,810 | 4 | 4 sizes | 71 | 0 | 0 |
| `AllocSharedConstruct5Wrappers` | Custom-allocator allocation, atomic refcount copy-by-value of a two-pointer shared handle, and 5-argument constructor call (dereferenced `a2`, direct or dereferenced `a3`) | 10 | 1,300 | 1 | 3 sizes | 10 | 0 | 0 |
| `AllocTailConstruct2Wrappers` | Custom-allocator allocation followed by a 2-argument tail-call constructor (direct or dereferenced `a1`) | 37 | 1,887 | 2 | 6 sizes | 35 | 0 | 0 |
| `AllocDerefU32Tail3Wrappers` | Custom-allocator allocation followed by a 3-argument tail-call constructor loading a 32-bit unsigned value from `a1` | 10 | 660 | 1 | 1 size | 5 | 0 | 0 |
| `AllocDeref2Construct5Wrappers` | Custom-allocator allocation followed by a 5-argument constructor call dereferencing `a2` and `a3` | 6 | 516 | 1 | 2 sizes | 2 | 0 | 0 |
| `BufferClearFreeWrappers` | Non-null guarded indirect deallocation of a three-pointer buffer followed by clearing `begin`, `end`, and `cap` | 14 | 840 | 1 | — | 0 | 0 | 0 |
| `BufferDestroyClearFreeWrappers` | Non-null range destruction, guarded indirect deallocation, and clearing `begin`, `end`, and `cap` | 11 | 825 | 1 | — | 11 | 0 | 0 |
| `BufferResetFreeWrappers` | Non-null guarded indirect deallocation followed by rebinding `begin`, `end`, and `cap` with typed element stride (`1`, `2`, `4`, or `8` bytes) | 17 | 1,768 | 1 | strides 1, 2, 4, 8 | 0 | 0 | 0 |

## Reproduce

With Python, `iced-x86==1.21.0`, and the configured contributor toolchain, run the following with a fresh output directory. Omit `--out` and `--compare` for candidate counts without compiling.

```sh
python3 tools/sbo_alloc_log_generate.py --exe "$SKYRIM_EXE"
python3 tools/sbo_alloc_log_generate.py --exe "$SKYRIM_EXE" --out .local/sbo-alloc-log --compare
```

The snapshot contains the sharded C++ units (at most 20 functions per unit, 31 units total) under `reconstruction/leaves/` and, under `tasks/<family>/`, the task, symbol map and per-RVA comparator reports. The command returns a non-zero exit status if any comparison fails.

## Evidence and limits

- Target `sse-steam-1-7-104`, image SHA-256 `846efccf0c1374d71f892907f46549560f2fcb0a75cb87a3eed438baa0f1402f`. Input inventory, target-hash manifest, recipe and community boundary shards came from repository revision `f2837c2218c7f481c211b4d4da3c0f6812d61965`. Boundary source `CRASHLOGGER_PDB(59818@2026.10.24)`.
- The 490 candidates are disjoint from the matched inventory and from all previously merged leaf-family generators (`static_data_generate.py`, `fixed_argument_generate.py`, `return_argument_generate.py`, `conditional_dealloc_generate.py`, `vftable_reset_generate.py`, `range_copy_generate.py`, `guarded_callback_generate.py`, `lazy_static_generate.py`, and `custom_dealloc_generate.py`).
- Validation partitioned the 490 discovered candidates into a development split (263 functions, 42,437 bytes, containing all inspected exemplars) and an untouched held-out validation split (227 functions, 36,833 bytes). Both splits passed 100% locally (`263/263` dev, `227/227` held-out) without per-function edits, followed by `490/490` on the full end-to-end snapshot run.
- Scope covers complete, unambiguous indexed non-EH extents (rejecting any function whose `.pdata` unwind entries declare an exception or unwind handler) matching the default MSVC `/O2` code generation. Every candidate validates indexed executable direct callees outside its own extent, 16-byte-aligned or indexed executable cleanup function pointers, non-executable read-only `.rdata` table and format targets, and writable non-executable `.data` allocator, deallocator, and log-channel globals.
- Sharding at 20 functions per unit keeps every emitted `.cpp` and `.h` file well under `contrib_screen`'s 64 KiB file-size cap and 32-level delimiter-depth bound (all 62 emitted files pass `contrib_screen`, `lint_reconstruction`, and `check_code_rules` MATCH rules with zero findings).
- Shared-handle factory templates (`AllocSharedConstruct4Wrappers`, `AllocSharedConstruct5Wrappers`) use a self-contained `<intrin.h>` control-block refcount helper rather than `<memory>`, keeping all includes within `contrib_screen.STANDARD_HEADERS`.
- Verification used the pinned MSVC 19.16.27054 x64 recipe under Wine and the ordinary project comparator. Confidence is high in the compiled code, unwind information, static storage references, allocation sizes, element strides, and bound call targets on this host. Original class ownership, subsystem names, member layouts, and symbol identities remain unknown; all emitted names are `GUESS(low) STANDIN`.
- Exposure was mixed, from local binary analysis and the committed `CRASHLOGGER_PDB(59818@2026.10.24)` boundary metadata. Original templates and analysis by Antigravity (Gemini 4 Neon), 2026-11-21. These runs supply one-host evidence only; second-host comparison and admitted review are pending.
