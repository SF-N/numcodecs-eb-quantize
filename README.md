[![image](https://img.shields.io/github/actions/workflow/status/SF-N/numcodecs-eb-quantize/ci.yml?branch=main)](https://github.com/SF-N/numcodecs-eb-quantize/actions/workflows/ci.yml?query=branch%3Amain)
[![image](https://img.shields.io/pypi/v/numcodecs-eb-quantize.svg)](https://pypi.python.org/pypi/numcodecs-eb-quantize)
[![image](https://img.shields.io/pypi/l/numcodecs-eb-quantize.svg)](https://github.com/SF-N/numcodecs-eb-quantize/blob/main/LICENSE)
[![image](https://img.shields.io/python/required-version-toml?tomlFilePath=https%3A%2F%2Fraw.githubusercontent.com%2FSF-N%2Fnumcodecs-eb-quantize%2Frefs%2Fheads%2Fmain%2Fpyproject.toml)](https://pypi.python.org/pypi/numcodecs-eb-quantize)
[![image](https://readthedocs.org/projects/numcodecs-eb-quantize/badge/?version=latest)](https://numcodecs-eb-quantize.readthedocs.io/en/latest/?badge=latest)

# numcodecs-eb-quantize

`ErrorBoundedQuantizeCodec` for the [`numcodecs`] buffer compression API.

The `ErrorBoundedQuantizeCodec` is a meta-codec that quantises floating-point data linearly (uniformly) with an absolute error bound `eb`: the data is mapped to the integer indices `k = round((x - offset) / (2 * eb))`, which are encoded with an inner codec, and reconstructed as `offset + k * 2 * eb`, so that `|x_dec - x| <= eb` for every finite value. The `offset` defaults to the finite minimum of the data and the indices use the smallest fitting integer dtype. The reconstruction is verified during encoding, and the quantisation step is shrunk slightly if floating-point rounding would exceed the bound.

```python
from numcodecs_eb_quantize import ErrorBoundedQuantizeCodec

codec = ErrorBoundedQuantizeCodec(codec=dict(id="zstd", level=19), eb=0.5)
```

Non-finite values decode to the `offset`; combine with a masking meta-codec such as [`numcodecs-mask`](https://numcodecs-mask.readthedocs.io) to preserve them. For a pointwise *relative* error bound, wrap this codec in [`numcodecs-pw-ratio`](https://numcodecs-pw-ratio.readthedocs.io), which translates the ratio bound into an absolute bound on the logarithms.

[`numcodecs`]: https://numcodecs.readthedocs.io/en/stable/

## License

Licensed under the Mozilla Public License, Version 2.0 ([LICENSE](LICENSE) or https://www.mozilla.org/en-US/MPL/2.0/).


## Funding

The `numcodecs-eb-quantize` package has been developed as part of [ESiWACE3](https://www.esiwace.eu), the third phase of the Centre of Excellence in Simulation of Weather and Climate in Europe.

Funded by the European Union. This work has received funding from the European High Performance Computing Joint Undertaking (JU) under grant agreement No 101093054.
