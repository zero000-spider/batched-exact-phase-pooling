# Third-party source and licensing

The following upstream source is included with its original authorship comments
and GNU GPL version 3 license text. Exact file checksums are recorded in
`third_party_sources.json`. Full licenses remain in each source directory; a
convenience copy is in `LICENSES/GPL-3.0.txt`.

| Component | Authors | Upstream and pinned revision | Included location |
| --- | --- | --- | --- |
| MiniRocket | Angus Dempster, Daniel F. Schmidt, Geoffrey I. Webb | [angus924/minirocket](https://github.com/angus924/minirocket), `0b1c245d9c9dbc50886f28bc7b32d5d45b5663d6` | `work/eippv_extension/project/minirocket_reference/` |
| MultiRocket | Chang Wei Tan, Angus Dempster, Christoph Bergmeir, Geoffrey I. Webb | [ChangWeiTan/MultiRocket](https://github.com/ChangWeiTan/MultiRocket), `3ccaa4f8ed11769b904af885e17018c39c89a2ac` | `work/modern_baselines/MultiRocket/` |
| SelF-Rocket | Mouhamadou Mansour Lo, Gildas Morvan, Mathieu Rossi, Fabrice Morganti, David Mercier | [ANR-MYEL/SelF-Rocket](https://github.com/ANR-MYEL/SelF-Rocket), `0c7d18c791181a9b8817d420711b148eaf395cf7` | `work/modern_baselines/SelF-Rocket/` |

## Compatibility adapter

`work/modern_baselines/multirocket_ridge_compat.py` is derived from the pinned
MultiRocket `multirocket/multirocket.py`. It removes the unused
`from multirocket.logistic_regression import LogisticRegression` import and the
obsolete `normalize=False` argument. The import belongs to the torch-based
logistic-regression option, which is not used by this study's ridge classifiers.
The keyword was removed from the current scikit-learn API; the study retains an
explicit `StandardScaler` before the ridge classifier. The adaptation addresses
dependency/API compatibility and leaves the feature-generating functions intact.
The adapter header records the changes and modification date. The original file
is also included for comparison. SelF-Rocket and MiniRocket source files retain
the pinned upstream contents.

`reproduce.py adapter` verifies the permitted source edits, compares fitted
parameters and transformed features with the pinned upstream implementation, and
compares ridge predictions using an explicitly documented compatibility shim for
the removed keyword. The report is `verification/adapter_equivalence.json`.
The separate 9,408-feature EI/MultiRocket comparison matches feature budgets,
training folds and the ridge grid; it evaluates the two feature frameworks.

These upstream GPL terms must be preserved when distributing the combined code.
The combined candidate includes a root GPL version 3 license. This does
not grant rights over the provider datasets. Python dependencies are installed
separately and retain their respective package licenses.

Copyright and attribution for each author's original contributions are retained.
No additional restriction on GPL-permitted uses is imposed by this notice.
Upstream sample-data descriptions and benchmark README documents are omitted from
this focused source release; upstream project links, authorship notices, licenses
and citations are retained here.

## Upstream publications

- Dempster, Schmidt and Webb. *MiniRocket: A Very Fast (Almost) Deterministic
  Transform for Time Series Classification*. [Paper](https://arxiv.org/abs/2012.08791).
- Tan, Dempster, Bergmeir and Webb. *MultiRocket: Multiple pooling operators and
  transformations for fast and effective time series classification*.
  [Paper](https://arxiv.org/abs/2102.00457).
- Lo, Morvan, Rossi, Morganti and Mercier. *Time series classification with random
  convolution kernels: pooling operators and input representations matter*.
  [Paper](https://arxiv.org/abs/2409.01115).
