# Dependencies

The core runtime uses PyTorch, NumPy, and safetensors. Tests use pytest. Installation
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

The optional `provider` extra adds the exact tested feature-provider libraries.
It is unnecessary for the invented feature examples and core CPU tests.

| Package | Version requirement | Purpose | Upstream source license | Official license |
| --- | --- | --- | --- | --- |
| `transformers` | `==5.15.0` | Official Qwen implementation and loading | `Apache-2.0` | [Transformers LICENSE](https://github.com/huggingface/transformers/blob/v5.15.0/LICENSE) |
| `tokenizers` | `==0.22.2` | Matching fast tokenizer | `Apache-2.0` | [tokenizers LICENSE](https://github.com/huggingface/tokenizers/blob/v0.22.2/LICENSE) |
| `huggingface-hub` | `==1.27.0` | Separately obtaining upstream snapshot files | `Apache-2.0` | [huggingface_hub LICENSE](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/LICENSE) |
| `accelerate` | `==1.14.0` | Provider loading support | `Apache-2.0` | [Accelerate LICENSE](https://github.com/huggingface/accelerate/blob/v1.14.0/LICENSE) |

License references were checked against these official upstream sources on
2026-10-18 for the core dependencies and 2026-10-20 for the optional provider.
They describe the named upstream projects. Installed packages can
contain additional components under other licenses; retain the license and
notice files supplied with the exact distributions you use.

## External features and models

The feature-provider interface permits separately supplied model features. The
Experimental 1 adapter uses the separately obtained official
`Qwen/Qwen3.5-2B-Base` snapshot at revision
`b1485b2fa6dfa1287294f269f5fb618e03d52d7c`, with an
[Apache-2.0 license](https://huggingface.co/Qwen/Qwen3.5-2B-Base/blob/b1485b2fa6dfa1287294f269f5fb618e03d52d7c/LICENSE).
Donor weights, tokenizer assets, third-party model implementation distributions
and datasets are outside this source distribution. Obtain them separately and
follow their own licenses. The M1 MIT license does not grant rights to those
assets. See the [provider audit](FEATURE_PROVIDER.md) and
[Experimental 1 rights record](EXPERIMENTAL_1_RIGHTS.md).

CPU execution is sufficient for the included examples and tests. If you choose
a CUDA installation, PyTorch may install NVIDIA components under separate
terms. See the [NVIDIA CUDA license agreement](https://docs.nvidia.com/cuda/eula/index.html)
and the notices included with the installed components. No CUDA binaries are
distributed in this repository.

See [`THIRD_PARTY_NOTICES.md`](../THIRD_PARTY_NOTICES.md) for the source-license
boundary.
