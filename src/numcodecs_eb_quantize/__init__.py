"""
[`ErrorBoundedQuantizeCodec`][numcodecs_eb_quantize.ErrorBoundedQuantizeCodec] for the [`numcodecs`][numcodecs] buffer compression API.
"""

__all__ = ["ErrorBoundedQuantizeCodec"]

import math
from collections.abc import Callable
from functools import reduce
from io import BytesIO

import leb128
import numcodecs.compat
import numcodecs.registry
import numpy as np
from numcodecs.abc import Codec
from numcodecs_combinators.abc import CodecCombinatorMixin
from numcodecs_mask.abc import MaskAwareCodecMixin
from typing_extensions import Buffer  # MSPV 3.12


class ErrorBoundedQuantizeCodec(Codec, CodecCombinatorMixin, MaskAwareCodecMixin):
    """
    Meta-codec that quantises floating-point data linearly (uniformly) with an
    absolute error bound and encodes the quantisation indices with the `codec`.

    The data is mapped to the integer indices `k = round((x - offset) / step)`
    with `step = 2 * eb`, and reconstructed as `offset + k * step`, so that the
    pointwise absolute error `|x_dec - x|` is at most `eb` for every finite
    value. The `offset` defaults to the finite minimum of the data and the
    indices are stored in the smallest fitting integer dtype.

    The reconstruction is verified during encoding: if floating-point rounding
    would make any error exceed `eb`, the step is shrunk slightly and the
    quantisation is repeated.

    Non-finite values (NaN, +-inf) are mapped to the index `0` and thus decode
    to the `offset`. To preserve them, combine this codec with a masking
    meta-codec such as
    [`numcodecs_mask.MaskMetaCodec`](https://numcodecs-mask.readthedocs.io).
    This codec implements the
    [`MaskAwareCodecMixin`][numcodecs_mask.abc.MaskAwareCodecMixin]: masked
    values are ignored when the quantisation grid is determined and the mask
    is forwarded to the inner `codec` if it is mask-aware as well.

    Parameters
    ----------
    codec : dict | Codec
        The configuration or instantiated codec that encodes the integer
        indices.
    eb : float
        The positive absolute error bound.
    offset : None | float, optional
        The quantisation grid offset, or [`None`][None] to use the finite
        minimum of the data.
    """

    __slots__: tuple[str, ...] = ("_codec", "_eb", "_offset")
    _codec: Codec
    _eb: float
    _offset: None | float

    codec_id: str = "eb_quantize"  # type: ignore

    def __init__(
        self,
        *,
        codec: dict | Codec,
        eb: float,
        offset: None | float = None,
    ) -> None:
        if not (math.isfinite(eb) and eb > 0):
            raise ValueError("eb must be finite and positive")
        if offset is not None and not math.isfinite(offset):
            raise ValueError("offset must be finite")

        self._codec = (
            codec if isinstance(codec, Codec) else numcodecs.registry.get_codec(codec)
        )
        self._eb = float(eb)
        self._offset = None if offset is None else float(offset)

    def encode(self, buf: Buffer) -> bytes:
        """
        Encode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Floating-point data to be encoded. May be any object supporting
            the new-style buffer protocol.

        Returns
        -------
        enc : bytes
            Encoded data as a bytestring.
        """

        return self._encode(buf, None)

    def encode_masked(
        self, buf: Buffer, mask: np.ndarray[tuple[int, ...], np.dtype[np.bool]]
    ) -> bytes:
        """
        Encode the data in `buf`, ignoring the values where `mask` is
        [`True`][True].

        Parameters
        ----------
        buf : Buffer
            Floating-point data to be encoded. May be any object supporting
            the new-style buffer protocol. The values at masked positions are
            unspecified.
        mask : np.ndarray[tuple[int, ...], np.dtype[np.bool]]
            The [boolean][numpy.bool] mask, of the same shape as the data, of
            the values that do not need to be preserved.

        Returns
        -------
        enc : bytes
            Encoded data as a bytestring.
        """

        return self._encode(buf, mask)

    def _encode(
        self,
        buf: Buffer,
        mask: None | np.ndarray[tuple[int, ...], np.dtype[np.bool]],
    ) -> bytes:
        a = numcodecs.compat.ensure_ndarray(buf)
        dtype, shape = a.dtype, a.shape

        if not np.issubdtype(dtype, np.floating):
            raise TypeError("can only encode floating point values")

        is_finite = np.isfinite(a)
        if mask is not None:
            is_finite &= ~np.asarray(mask, dtype=np.bool).reshape(shape)
        finite = a[is_finite].astype(np.float64)

        offset = self._offset
        if offset is None:
            offset = float(finite.min()) if finite.size > 0 else 0.0

        # quantise; if floating-point rounding of the reconstruction (e.g. the
        # cast back to a narrower dtype) exceeds the bound, shrink the step
        step = 2.0 * self._eb
        for _ in range(16):
            indices = np.where(is_finite, np.rint((a - offset) / step), 0.0)
            reconstructed = (offset + indices * step).astype(dtype)
            error = np.abs(reconstructed[is_finite].astype(np.float64) - finite)
            error_max = float(error.max()) if error.size > 0 else 0.0
            if error_max <= self._eb:
                break
            step *= (self._eb / error_max) * (1.0 - 1e-9)
        else:
            raise ValueError(
                f"cannot satisfy the error bound {self._eb} with dtype {dtype}, "
                "which does not resolve the data finely enough"
            )

        index_min = int(indices.min()) if indices.size > 0 else 0
        index_max = int(indices.max()) if indices.size > 0 else 0
        index_dtype: np.dtype
        for candidate in (np.uint8, np.uint16, np.uint32, np.int64):
            info = np.iinfo(candidate)
            if index_min >= info.min and index_max <= info.max:
                index_dtype = np.dtype(candidate)
                break
        else:  # pragma: no cover
            raise ValueError("quantisation indices exceed the int64 range")

        if mask is not None and isinstance(self._codec, MaskAwareCodecMixin):
            encoded_buf = self._codec.encode_masked(indices.astype(index_dtype), mask)  # type: ignore
        else:
            encoded_buf = self._codec.encode(indices.astype(index_dtype))
        encoded = numcodecs.compat.ensure_ndarray(encoded_buf)

        # message: dtype shape offset step index-dtype
        #          encoded-dtype encoded-shape [padding] encoded
        message: list[bytes | bytearray] = []

        message.append(leb128.u.encode(len(dtype.str)))
        message.append(dtype.str.encode("ascii"))

        message.append(leb128.u.encode(len(shape)))
        for s in shape:
            message.append(leb128.u.encode(s))

        message.append(np.array([offset, step], dtype="<f8").tobytes())

        message.append(leb128.u.encode(len(index_dtype.str)))
        message.append(index_dtype.str.encode("ascii"))

        message.append(leb128.u.encode(len(encoded.dtype.str)))
        message.append(encoded.dtype.str.encode("ascii"))

        message.append(leb128.u.encode(encoded.ndim))
        for s in encoded.shape:
            message.append(leb128.u.encode(s))

        # insert padding to align with encoded itemsize
        message.append(
            b"\0"
            * (
                encoded.dtype.itemsize
                - (sum(len(m) for m in message) % encoded.itemsize)
            )
        )

        # ensure that the encoded values are encoded in little endian binary
        message.append(encoded.astype(encoded.dtype.newbyteorder("<")).tobytes())

        return b"".join(message)

    def decode(self, buf: Buffer, out: None | Buffer = None) -> Buffer:
        """
        Decode the data in `buf`.

        Parameters
        ----------
        buf : Buffer
            Encoded data. Must be an object representing a bytestring, e.g.
            [`bytes`][bytes] or a 1D array of [`np.uint8`][numpy.uint8]s etc.
        out : Buffer, optional
            Writeable buffer to store decoded data. N.B. if provided, this
            buffer must be exactly the right size to store the decoded data.

        Returns
        -------
        dec : Buffer
            Decoded data. May be any object supporting the new-style buffer
            protocol.
        """

        return self._decode(buf, None, out)

    def decode_masked(
        self,
        buf: Buffer,
        mask: np.ndarray[tuple[int, ...], np.dtype[np.bool]],
        out: None | Buffer = None,
    ) -> Buffer:
        """
        Decode the data in `buf`, which was encoded with the same `mask`.

        Parameters
        ----------
        buf : Buffer
            Encoded data. Must be an object representing a bytestring, e.g.
            [`bytes`][bytes] or a 1D array of [`np.uint8`][numpy.uint8]s etc.
        mask : np.ndarray[tuple[int, ...], np.dtype[np.bool]]
            The [boolean][numpy.bool] mask, of the same shape as the decoded
            data, that was passed to `encode_masked`.
        out : Buffer, optional
            Writeable buffer to store decoded data. N.B. if provided, this
            buffer must be exactly the right size to store the decoded data.

        Returns
        -------
        dec : Buffer
            Decoded data. May be any object supporting the new-style buffer
            protocol. The values at masked positions are unspecified.
        """

        return self._decode(buf, mask, out)

    def _decode(
        self,
        buf: Buffer,
        mask: None | np.ndarray[tuple[int, ...], np.dtype[np.bool]],
        out: None | Buffer,
    ) -> Buffer:
        b = numcodecs.compat.ensure_bytes(buf)

        b_io = BytesIO(b)

        # message: dtype shape offset step index-dtype
        #          encoded-dtype encoded-shape [padding] encoded
        dtype = np.dtype(b_io.read(leb128.u.decode_reader(b_io)[0]).decode("ascii"))
        shape = tuple(
            leb128.u.decode_reader(b_io)[0]
            for _ in range(leb128.u.decode_reader(b_io)[0])
        )

        offset, step = np.frombuffer(b_io.read(16), dtype="<f8", count=2)

        index_dtype = np.dtype(
            b_io.read(leb128.u.decode_reader(b_io)[0]).decode("ascii")
        )

        encoded_dtype = np.dtype(
            b_io.read(leb128.u.decode_reader(b_io)[0]).decode("ascii")
        )
        encoded_shape = tuple(
            leb128.u.decode_reader(b_io)[0]
            for _ in range(leb128.u.decode_reader(b_io)[0])
        )
        encoded_size = reduce(lambda a, b: a * b, encoded_shape, 1)

        # remove padding to align with encoded itemsize
        b_io.read(encoded_dtype.itemsize - (b_io.tell() % encoded_dtype.itemsize))

        encoded = (
            np.frombuffer(
                b_io.read(encoded_size * encoded_dtype.itemsize),
                dtype=encoded_dtype.newbyteorder("<"),
                count=encoded_size,
            )
            .astype(encoded_dtype)
            .reshape(encoded_shape)
        )

        indices = np.empty(shape, dtype=index_dtype)
        if mask is not None and isinstance(self._codec, MaskAwareCodecMixin):
            indices_decoded = self._codec.decode_masked(encoded, mask, out=indices)  # type: ignore
        else:
            indices_decoded = self._codec.decode(encoded, out=indices)
        indices = numcodecs.compat.ensure_ndarray(indices_decoded).reshape(shape)

        decoded = (float(offset) + indices.astype(np.float64) * float(step)).astype(
            dtype
        )

        return numcodecs.compat.ndarray_copy(decoded, out)  # type: ignore

    def get_config(self) -> dict:
        """
        Returns the configuration of this quantisation meta-codec.

        [`numcodecs.registry.get_codec(config)`][numcodecs.registry.get_codec]
        can be used to reconstruct this codec from the returned config.

        Returns
        -------
        config : dict
            Configuration of this quantisation meta-codec.
        """

        return dict(
            id=type(self).codec_id,
            codec=self._codec.get_config(),
            eb=self._eb,
            offset=self._offset,
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(codec={self._codec!r}, eb={self._eb!r}, offset={self._offset!r})"

    def map(self, mapper: Callable[[Codec], Codec]) -> "ErrorBoundedQuantizeCodec":
        """
        Apply the `mapper` to the inner `codec` of this quantisation
        meta-codec.

        Parameters
        ----------
        mapper : Callable[[Codec], Codec]
            The callable that is applied to the inner codec.

        Returns
        -------
        mapped : ErrorBoundedQuantizeCodec
            The mapped quantisation meta-codec.
        """

        return ErrorBoundedQuantizeCodec(
            codec=mapper(self._codec), eb=self._eb, offset=self._offset
        )


numcodecs.registry.register_codec(ErrorBoundedQuantizeCodec)
