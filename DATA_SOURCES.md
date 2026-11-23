# Data sources and reconstruction

Download data from the providers below and follow their applicable terms. This
repository distributes experiment code, checksums and prediction results, rather
than provider archives, images or reconstructed training/test inputs.

| Source | Provider page | Archive used by the entrypoint |
| --- | --- | --- |
| Fish | [Time Series Classification](https://www.timeseriesclassification.com/description.php?Dataset=Fish) | [Fish.zip](https://www.timeseriesclassification.com/aeon-toolkit/Fish.zip) |
| OSULeaf | [Time Series Classification](https://www.timeseriesclassification.com/description.php?Dataset=OSULeaf) | [OSULeaf.zip](https://www.timeseriesclassification.com/aeon-toolkit/OSULeaf.zip) |
| HandOutlines | [Time Series Classification](https://www.timeseriesclassification.com/description.php?Dataset=HandOutlines) | [HandOutlines.zip](https://www.timeseriesclassification.com/aeon-toolkit/HandOutlines.zip) |
| MPEG-7 CE-Shape-1 | [Temple shape dataset](https://dabi.temple.edu/external/shape/MPEG7/dataset.html) | [MPEG7dataset.zip](https://dabi.temple.edu/external/shape/MPEG7/MPEG7dataset.zip) |
| ETH-80 | [ETH-80 provider](https://www.d2.mpi-inf.mpg.de/datasets/eth80) | [eth80-contours.tgz](https://datasets.d2.mpi-inf.mpg.de/eth80/eth80-contours.tgz) |

The first three provider descriptions identify the original data sources. Fish
contains fish outlines donated by D. Lee; OSULeaf contains leaf outlines donated
by A. Grandhi. HandOutlines labels concern the correctness of hand outlines.
Use the provider's dataset references when citing these sources. MPEG-7 contains
1,400 shapes in 70 classes. ETH-80 supplies eight categories, ten objects per
category and 41 views per object.

## Download and preprocess

After installing the README environment, run:

```powershell
.\.venv\Scripts\python.exe -I -B raw_data.py rebuild --dataset Fish
.\.venv\Scripts\python.exe -I -B raw_data.py rebuild
.\.venv\Scripts\python.exe -I -B raw_data.py resplits
.\.venv\Scripts\python.exe -I -B raw_data.py fetch --dataset ETH80
```

The first command is sufficient for the representative Fish experiments. The
second generates the eight original-split inputs. The third generates forty
grouped-resplit inputs. ETH-80 preprocessing is performed by `reproduce.py eth80`.

If a provider rejects an automated request (for example, HTTP 401), open its
dataset page and use the download link, then import the downloaded archive:

```powershell
.\.venv\Scripts\python.exe -I -B raw_data.py fetch --dataset Fish --archive C:\Downloads\Fish.zip
```

Use `OSULeaf`, `HandOutlines`, `mpeg7_shape_b` or `ETH80` as the other dataset
identifiers. `raw_sources.json` records the provider URLs and accepted SHA256
checksums. An unrecognized archive is rejected. If providers replace an archive,
inspect its contents before updating the accepted checksums.

## Deterministic preprocessing

For MPEG-7, select the 1,400 named shape GIFs, extract the closed outer contour,
sample its radial distance at 512 equally spaced perimeter positions, and use
the fixed stratified split in `rocket_mpeg7_confirm.py`. The original study ZIP
and the provider ZIP contain the same selected image bytes; both container hashes
are accepted. Image member hashes and the container-equivalence record are in
`work/eippv_extension/project/contour_submission/repro/`.

For the three time-series archives, use the provider training/test split. Remove
raw test sequences that exactly duplicate training sequences. Resample periodically
to 32 and 64 points, standardize each contour, and use zero phase plus eight phases
from NumPy RNG seed 20261108. Subsequent experiment loaders apply the shared
processed-input duplicate rule. MPEG-7 training duplicates remain together in
inner validation. Grouped resplits use the union of duplicates in the raw,
32-point and 64-point representations before allocating objects.

`input_arrays.json` records the shape, dtype and SHA256 of every array in the
eight original and forty resplit inputs. Reconstruction checks those values
before running experiments. Original compressed-file hashes are retained for
interpreting historical experiment metadata; byte differences in NPZ compression
are accepted only when every array matches.

No blanket data-redistribution license is asserted by this repository. Dataset
rights remain with the providers; code-license files do not change those terms.
