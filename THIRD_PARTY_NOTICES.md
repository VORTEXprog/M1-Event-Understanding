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
| Transformers 5.15.0, optional provider | `Apache-2.0` | [Official LICENSE](https://github.com/huggingface/transformers/blob/v5.15.0/LICENSE) |
| tokenizers 0.22.2, optional provider | `Apache-2.0` | [Official LICENSE](https://github.com/huggingface/tokenizers/blob/v0.22.2/LICENSE) |
| huggingface_hub 1.27.0, optional provider | `Apache-2.0` | [Official LICENSE](https://github.com/huggingface/huggingface_hub/blob/v1.27.0/LICENSE) |
| Accelerate 1.14.0, optional provider | `Apache-2.0` | [Official LICENSE](https://github.com/huggingface/accelerate/blob/v1.14.0/LICENSE) |

These are dependency acknowledgements, not a license inventory for every
component in an installed wheel. PyTorch and NumPy distributions can contain
additional third-party code and associated notices. CUDA distributions can
include NVIDIA software governed by separate terms. Preserve the notices
shipped with those distributions when redistributing them.

This repository does not distribute third-party package source, wheels,
third-party pretrained model weights, tokenizer assets, or datasets. Any
separately obtained model or dataset remains subject to its own terms. The
Experimental 1 project weight package is a separate candidate attachment;
its project-owned parameter tensors use MIT as declared in the
[rights and distribution record](docs/EXPERIMENTAL_1_RIGHTS.md).

The optional compatible feature donor is `Qwen/Qwen3.5-2B-Base`, revision
`b1485b2fa6dfa1287294f269f5fb618e03d52d7c`. Its official
[Apache-2.0 license](https://huggingface.co/Qwen/Qwen3.5-2B-Base/blob/b1485b2fa6dfa1287294f269f5fb618e03d52d7c/LICENSE)
applies to those separately obtained assets. The M1 adapter does not redistribute
the donor or tokenizer and does not claim rights in the donor's upstream
pretraining data. Preserve upstream licenses and notices if redistributing any
third-party asset independently.

The direct dependency inventory and official source references are maintained
in [`docs/DEPENDENCIES.md`](docs/DEPENDENCIES.md).
