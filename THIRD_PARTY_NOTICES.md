# Third-party notices

The M1 source code and documentation in this repository are offered under the
[`MIT license`](LICENSE). Third-party dependencies retain their own copyrights
and licenses; the M1 license does not replace them.

M1 imports separately installed dependencies. Their upstream source licenses
are:

| Dependency | Upstream source license | License text |
| --- | --- | --- |
| PyTorch 2.13.0 | `BSD-3-Clause` | [Official LICENSE](https://github.com/pytorch/pytorch/blob/v2.13.0/LICENSE) |
| NumPy 2.5.2 | `BSD-3-Clause` | [Official LICENSE.txt](https://github.com/numpy/numpy/blob/v2.5.2/LICENSE.txt) |
| safetensors 0.8.0 | `Apache-2.0` | [Official LICENSE](https://github.com/safetensors/safetensors/blob/v0.8.0/LICENSE) |
| pytest 9.1.1, for development | `MIT` | [Official LICENSE](https://github.com/pytest-dev/pytest/blob/9.1.1/LICENSE) |
| setuptools, for building | `MIT` | [Official LICENSE](https://github.com/pypa/setuptools/blob/main/LICENSE) |

These are dependency acknowledgements, not a license inventory for every
component in an installed wheel. PyTorch and NumPy distributions can contain
additional third-party code and associated notices. CUDA distributions can
include NVIDIA software governed by separate terms. Preserve the notices
shipped with those distributions when redistributing them.

This repository does not distribute third-party package source, wheels,
pretrained model weights, tokenizer assets, or datasets. Any separately
obtained model, dataset, or provider implementation remains subject to its own
terms.

The direct dependency inventory and official source references are maintained
in [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md).
