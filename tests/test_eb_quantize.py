import numcodecs
import numcodecs.registry
import numpy as np
import pytest

INNER = dict(id="zlib", level=1)


def test_from_config():
    codec = numcodecs.registry.get_codec(dict(id="eb_quantize", codec=INNER, eb=0.5))
    assert codec.__class__.__name__ == "ErrorBoundedQuantizeCodec"
    assert codec.__class__.__module__ == "numcodecs_eb_quantize"
    assert codec.get_config() == dict(
        id="eb_quantize", codec=INNER, eb=0.5, offset=None
    )


def test_invalid():
    from numcodecs_eb_quantize import ErrorBoundedQuantizeCodec

    with pytest.raises(ValueError):
        ErrorBoundedQuantizeCodec(codec=INNER, eb=0.0)
    with pytest.raises(ValueError):
        ErrorBoundedQuantizeCodec(codec=INNER, eb=np.inf)
    with pytest.raises(ValueError):
        ErrorBoundedQuantizeCodec(codec=INNER, eb=1.0, offset=np.nan)
    with pytest.raises(TypeError):
        ErrorBoundedQuantizeCodec(codec=INNER, eb=1.0).encode(np.arange(10))
    # an error bound below the resolution of the data type cannot be satisfied
    with pytest.raises(ValueError):
        ErrorBoundedQuantizeCodec(codec=INNER, eb=1e-6).encode(
            np.linspace(-30.0, 30.0, 1001, dtype=np.float32)
        )


def test_map():
    from numcodecs_eb_quantize import ErrorBoundedQuantizeCodec

    codec = ErrorBoundedQuantizeCodec(codec=INNER, eb=0.25, offset=1.0)
    mapped = codec.map(lambda c: numcodecs.registry.get_codec(dict(id="zlib", level=9)))
    assert mapped.get_config() == dict(
        id="eb_quantize", codec=dict(id="zlib", level=9), eb=0.25, offset=1.0
    )


def check_roundtrip(data: np.ndarray, eb: float, **kwargs):
    codec = numcodecs.registry.get_codec(
        dict(id="eb_quantize", codec=INNER, eb=eb, **kwargs)
    )

    encoded = codec.encode(data)
    decoded = np.asarray(codec.decode(encoded))

    assert decoded.dtype == data.dtype
    assert decoded.shape == data.shape

    finite = np.isfinite(data)
    assert np.all(np.abs(decoded[finite] - data[finite]) <= eb)

    out = np.empty_like(data)
    codec.decode(encoded, out=out)
    np.testing.assert_array_equal(out, decoded)

    return encoded


def test_roundtrip():
    rng = np.random.default_rng(42)

    data = rng.normal(size=(50, 60)) * 10.0
    for eb in (1.0, 0.1, 1e-3, 1e-6):
        check_roundtrip(data, eb)
    for eb in (1.0, 0.1, 1e-3):
        check_roundtrip(data.astype(np.float32), eb)
    check_roundtrip(data, 0.5, offset=0.0)
    check_roundtrip(data, 0.5, offset=-100.0)

    # the error bound must also hold for values far from zero
    check_roundtrip(1e6 + rng.random(1000), 1e-3)
    check_roundtrip((1e3 + rng.random(1000)).astype(np.float32), 1e-2)

    # non-finite values decode to the offset
    codec = numcodecs.registry.get_codec(dict(id="eb_quantize", codec=INNER, eb=0.5))
    decoded = np.asarray(
        codec.decode(codec.encode(np.array([1.0, np.nan, 3.0, np.inf])))
    )
    assert decoded[1] == 1.0 and decoded[3] == 1.0
    assert abs(decoded[0] - 1.0) <= 0.5 and abs(decoded[2] - 3.0) <= 0.5

    # degenerate cases
    check_roundtrip(np.full((3, 3), 7.5), 0.1)
    check_roundtrip(np.zeros((0,), dtype=np.float64), 0.1)
    check_roundtrip(np.array(1.0), 0.1)
    check_roundtrip(np.array([np.nan, np.nan]), 0.1)


def test_index_dtype():
    from numcodecs_eb_quantize import ErrorBoundedQuantizeCodec

    class Capture(numcodecs.abc.Codec):
        codec_id = "capture-test"

        def encode(self, buf):
            self.seen = np.asarray(buf)
            return self.seen.tobytes()

        def decode(self, buf, out=None):
            return numcodecs.compat.ndarray_copy(
                np.frombuffer(buf, dtype=self.seen.dtype).reshape(self.seen.shape), out
            )

    inner = Capture()
    data = np.linspace(0.0, 100.0, 1001)
    ErrorBoundedQuantizeCodec(codec=inner, eb=0.5).encode(data)
    assert inner.seen.dtype == np.uint8
    ErrorBoundedQuantizeCodec(codec=inner, eb=0.001).encode(data)
    assert inner.seen.dtype == np.uint16
