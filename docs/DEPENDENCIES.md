# Dependencies

The runtime uses PyTorch, NumPy, and safetensors. Tests use pytest. Installation
requirements are declared in
[`pyproject.toml`](../pyproject.toml).

This inventory describes direct dependencies installed separately. It is not a
lockfile or a complete inventory of the libraries bundled into a binary wheel.

| Package | Version requirement | Purpose | Upstream source license | Official license |
| --- | --- | --- | --- | --- |
| `torch` | `==2.13.0` | Tensor operations and neural modules | `BSD-3-Clause` | [PyTorch 2.13.0 LICENSE](https://github.com/pytorch/pytorch/blob/v2.13.0/LICENSE) |
| `numpy` | `==2.5.2` | Numerical arrays | `BSD-3-Clause` | [NumPy 2.5.2 LICENSE.txt](https://github.com/numpy/numpy/blob/v2.5.2/LICENSE.txt) |
| `safetensors` | `==0.8.0` | Tensor checkpoint serialization | `Apache-2.0` | [safetensors 0.8.0 LICENSE](https://github.com/safetensors/safetensors/blob/v0.8.0/LICENSE) |
| `pytest` | `==9.1.1` | Development tests | `MIT` | [pytest 9.1.1 LICENSE](https://github.com/pytest-dev/pytest/blob/9.1.1/LICENSE) |
| `setuptools` | `>=77` | Build backend | `MIT` | [setuptools LICENSE](https://github.com/pypa/setuptools/blob/main/LICENSE) |

License references were checked against these official upstream sources on
2026-10-18. They describe the named upstream projects. Installed packages can
contain additional components under other licenses; retain the license and
notice files supplied with the exact distributions you use.

## External features and models

The feature-provider interface permits separately supplied model features.
Model weights, tokenizer assets, model implementation code, and datasets are
outside this source distribution. Obtain them separately and follow their own
licenses. The M1 MIT license does not grant rights to those assets.

CPU execution is sufficient for the included examples and tests. If you choose
a CUDA installation, PyTorch may install NVIDIA components under separate
terms. See the [NVIDIA CUDA license agreement](https://docs.nvidia.com/cuda/eula/index.html)
and the notices included with the installed components. No CUDA binaries are
distributed in this repository.

See [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) for the source-license
boundary.
