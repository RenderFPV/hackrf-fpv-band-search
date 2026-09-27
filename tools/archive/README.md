# Development scaffolding

These are not part of the check suite and nothing in the project runs them
automatically. They are the scripts written *while* working out how the DSP
behaves — comparing the rasteriser against a slow reference, timing candidate
implementations, profiling where decode time goes, and scoring output against
the separate `hackrf-fpv-still-decoder` project.

They are archived rather than deleted because they still answer useful
questions, and because several of them are the evidence behind claims made
elsewhere in the README. `check_cluster.py` and `check_centroids.py` in
particular are what justify the vectorised cluster and centroid code in
`fpv_rf/dsp.py` being as fast as it is.

Run them from the repository root:

```
python tools\archive\check_cluster.py
python tools\archive\prof_sim.py
```

Two caveats:

- **Several need the optional real capture.** Set `$FPV_CAPTURE` or they skip.
  See the section on that in the top-level README.
- **Four scripts that used to be here were removed.** `compare_ref.py`,
  `score_all.py`, `test_downscale.py` and `test_resize.py` all imported from a
  *separate* project, `hackrf-fpv-still-decoder`, which is not vendored here — so
  none of them could run for anyone who did not already have that checkout. They
  answered their questions during development and were never part of the suite.
  Nothing in this repository references them any more.

## What lives here

| Script | What it was for |
|---|---|
| `check_centroids.py` | vectorised centroid finder vs a slow reference |
| `check_cluster.py` | vectorised clustering vs a slow reference |
| `diag_anchor.py` | where sync pulses anchor against the expected grid |
| `diag_frames.py` | frame-by-frame inspection of a decode |
| `diag_jitter.py` | line-to-line timing jitter in the rasteriser |
| `line_profile.py` | signal profile along one scanline |
| `prof.py` / `prof_sim.py` | where decode time actually goes |
| `sweep_geom.py` | sweep of rasteriser geometry parameters |

`check_frame.py` is **not** in here despite the similar name: it is a shared
correlation helper, and the live `check_sources.py` depends on it.
