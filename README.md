# Exact phase-averaged PPV with shared threshold queries

Code and numerical results for batched exact integration (EI) on periodic
piecewise-linear responses. The implementation shares response endpoints across
thresholds and combines direct evaluation, cached endpoints and an event sweep.

## Installation

The validated environment is 64-bit Windows with Python 3.12. Create a virtual
environment at the repository root. In PowerShell:

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -m pip install --require-hashes -r requirements-windows-hashed.txt
.\.venv\Scripts\python.exe -m pip check
```

Use that environment's Python for the commands below. Python 3.12 and
`requirements-lock.txt` define the reference package versions. The Windows hashed
lock additionally verifies the specific wheels used for validation. PyTorch is
unnecessary for the ridge-classifier experiments provided here.

**Note — data download:** If an automated request returns HTTP 401, open the
[Fish provider page](https://www.timeseriesclassification.com/description.php?Dataset=Fish),
download [Fish.zip](https://www.timeseriesclassification.com/aeon-toolkit/Fish.zip),
and import it before running the experiment:

```powershell
.\.venv\Scripts\python.exe -I -B raw_data.py fetch --dataset Fish --archive "C:\Downloads\Fish.zip"
.\.venv\Scripts\python.exe -I -B raw_data.py rebuild --dataset Fish
```

The same import procedure supports the other datasets; see
[DATA_SOURCES.md](DATA_SOURCES.md).

## Start here

```powershell
.\.venv\Scripts\python.exe -I -B reproduce.py verify
.\.venv\Scripts\python.exe -I -B reproduce.py adapter
.\.venv\Scripts\python.exe -I -B reproduce.py numerical
.\.venv\Scripts\python.exe -I -B plot_results.py
.\.venv\Scripts\python.exe -I -B raw_data.py rebuild --dataset Fish
.\.venv\Scripts\python.exe -I -B reproduce.py core --case fish_32 --seed 23 --budget 9996
.\.venv\Scripts\python.exe -I -B reproduce.py modern --case fish_32 --recipe multi_controlled
.\.venv\Scripts\python.exe -I -B reproduce.py modern --case fish_32 --recipe eippv_controlled
```

`verify` checks the release manifest and computational source fingerprints.
`adapter` checks the MultiRocket compatibility changes and compares upstream and
adapted fitted parameters, transformed features and ridge predictions.
`numerical` checks the EI implementations against direct and high-precision
references, including flat edges, threshold ties and short-edge conditioning.
`core` refits the thresholds, scaler and classifier in the original training-only
selection procedure and checks every prediction and the selected parameters against
the stored reference. The two `modern` commands rerun the 9,408-feature comparison.
Initial Numba compilation can take several minutes.

Generated files go to `reproduced/`. Provider archives and reconstructed inputs are
local caches excluded from version control. Input verification hashes each array's
shape, dtype and values, so ZIP compression metadata does not affect equivalence.
`outputs/` contains the frozen reference results and is not used as a run cache.

File and archive SHA256 checks verify exact bytes. Input-array fingerprints also
verify shape, dtype and exact numerical values against the reference environment.
Other operating systems or CPU/library builds can produce floating-point
differences. For such a reproduction, record the environment, run `numerical`, and
compare numerical errors and predictions separately while retaining the original
integrity checks. A fingerprint mismatch requires that comparison before results
can be treated as a reproduction of the reference run.

## Baseline compatibility and matched comparison

The MultiRocket adapter removes an unused import of the torch-based logistic
classifier and the obsolete scikit-learn `normalize=False` keyword. The upstream
source is retained. `reproduce.py adapter` checks the exact edits and tests
feature/parameter equality separately from classifier-API compatibility; its
method and results are recorded in `verification/adapter_equivalence.json`.

The controlled comparison uses 9,408 actual features for both MultiRocket and EI,
the same input split, training folds and ridge-penalty grid, with each transform
and scaler refitted inside each fold. This compares the two feature frameworks
at a matched dimensional budget. Adapter equivalence is established by the
separate check above. The author-recipe MultiRocket result at 49,728 features is
retained as an additional comparison.

## Experiments and figures

| Command after `reproduce.py` | Experiment |
| --- | --- |
| `core` | All 40 EI/K32 pipeline configurations; `--jobs 2` controls concurrent workers |
| `chain --case fish_32 --method circular` | Hard PPV with the common training-only selection procedure |
| `modern --case fish_32 --recipe multi_standard` | Author-recipe MultiRocket, requested 50,000 features |
| `modern --case fish_32 --recipe self_standard` | Author-recipe SelF-Rocket |
| `modern --case fish_32 --recipe circular --split 101` | Four pooling methods on a grouped training/test resplit |
| `costs --case fish_32 --budget 9996` | Feature extraction and full inference timing |
| `statistics` | Recompute the primary paired statistics from the stored predictions |
| `synthetic` | Response-length and threshold-density timing study |
| `eth80` | Object-disjoint real multiview experiment |

Cases are `mpeg7_32`, `mpeg7_64`, `hand_32`, `hand_64`, `fish_32`, `fish_64`,
`osuleaf_32` and `osuleaf_64`. Model seeds are 23, 37 and 71; feature budgets are
756, 9,996 and 49,728. Resplit seeds are 101, 211, 307, 401 and 503.
The default `modern` seed is 23 and the default split is the original split (0).

`plot_results.py` renders the five manuscript figures as PNG and vector PDF in
`reproduced/paper_figures/`. They use the recorded CSV values; the source file hashes
are saved beside the figures. To choose a destination, use
`--output path/to/figures`.

`redraw_all_figures.py` additionally redraws all 18 archived PNG/PDF figure pairs
from their numerical tables. Pixel comparisons, PDF drawing-stream checks and
manuscript embedded-image checks are summarized in
[FIGURE_PROVENANCE.md](FIGURE_PROVENANCE.md), with per-file records in
`verification/figure_provenance.json`.
Use `redraw_all_figures.py --check` to compare every redrawn PNG with its archived
pixel array in the reference environment.

Run timings serially on an otherwise idle machine. The entrypoint uses two Numba
threads and one BLAS thread. Recomputed timings reflect the current machine;
stored prediction comparisons use exact array equality.

## Layout

| Location | Contents |
| --- | --- |
| `work/batched_pooling.py` | Shared endpoint queries, compensated sweep, hybrid dispatch |
| `work/batched_study.py` | Fold-local selection and matched extraction benchmark |
| `work/modern_extension.py` | Controlled and author-recipe baselines; grouped resplits |
| `work/eippv_extension/project/` | Periodic responses, interpolation and source preprocessing |
| `work/modern_baselines/` | Pinned third-party source and ridge compatibility adapter |
| `outputs/evidence_chain_strict/` | Main paired comparisons and feature-budget checks |
| `outputs/modern_baselines_stability/` | Modern baselines, resplits and high-K diagnostics |
| `outputs/shared_threshold_study/` | Numerical checks, classification replays and timing measurements |
| `outputs/eth80_multiview/` | ETH-80 predictions, splits and aggregate results |

See [DATA_SOURCES.md](DATA_SOURCES.md) for provider links and reconstruction, and
[THIRD_PARTY.md](THIRD_PARTY.md) for upstream attribution and licenses.

## License

This combined source release is licensed under GNU GPL version 3; see
[LICENSE](LICENSE). Copyright and attribution for original contributions remain
with their respective authors. Upstream notices and license files are preserved;
reuse and redistribution follow the GPLv3 terms. Dataset rights remain with their
providers as described in [DATA_SOURCES.md](DATA_SOURCES.md).

## Use of AI tools

AI tools (ChatGPT/Codex) were used as development assistants for exploring
implementation ideas, generating and debugging code, preparing experiment and
analysis scripts, and drafting documentation. Numerical results and figures are
computed by the supplied code from the documented data sources and explicitly
identified synthetic workloads. The authors reviewed the key implementation and
checked the experimental results; numerical audits and reproduction checks are
documented in the verification report (`VERIFICATION.md`). The authors retain full
responsibility for scientific interpretation and the submitted work.
