# 0005 — Package the Rerun viewer from the upstream wheel

**Status:** accepted · 2026-09-15

## Context

`rr.spawn()` appeared to work and did not raise, but the viewer never opened.
The failure surfaced several seconds later as a gRPC timeout:

```
WARN  re_sdk::spawn: Spawned Rerun Viewer did not bind to port 9876 in time.
ERROR re_sdk::recording_stream: Failed to flush previous sink: gRPC has been
      unable to connect for 6s, uri: rerun+http://127.0.0.1:9876/proxy
```

Two separate problems:

1. `python3Packages.rerun-sdk` installs a `rerun` shim at `bin/rerun` that does
   `from rerun_cli.__main__ import main`, but the nixpkgs build never installs
   `rerun_cli`. The shim is therefore always broken.
2. nixpkgs also has a standalone `rerun` viewer, but at the pinned revision it
   is **0.27.2** against an SDK of **0.37.2** — ten minor versions apart.
   Rerun's recording format changes across releases, so pairing them is not
   safe even if the shim were fixed.

The consequence was that `.rrd` files were write-only: no 3D view, no
scrubbable timeline, no way to watch the arm move. Visualisation is most of the
reason Rerun is in the stack at all.

## Decision

Package the viewer by extracting it from the official `rerun-sdk` wheel, which
ships it at `rerun_sdk/rerun_cli/rerun` (a 260 MB binary), and patch it with
`autoPatchelfHook` — the same approach already used for the OpenCASCADE
bindings. It is placed ahead of the Python environment on `PATH` so it wins
over the broken shim.

`wgpu` and `winit` `dlopen` their graphics libraries, so those never appear in
the ELF headers and `autoPatchelf` cannot find them. They are injected through
a `makeWrapper` `LD_LIBRARY_PATH` prefix instead.

## Consequences

**Gained.** A viewer whose version matches the SDK exactly, with no Rust
toolchain and no multi-hour compile. `rerun --version` reports 0.37.2 and
`rr.spawn()` now binds port 9876 as intended. `just view` opens a recording.

**Cost.** A 164 MB wheel download for a binary we extract one file from, and a
version string that must be bumped in two places when the SDK moves. If
nixpkgs ever ships a matched pair, this derivation should be deleted.

**Not solved by this.** The dataframe read-back API is still unavailable — but
that is unrelated to packaging. `rerun.dataframe` is absent from the *official
upstream wheel* too, so the API has moved since the documentation describing
it. The parquet sink in `arm.telemetry` stands on its own merits.
