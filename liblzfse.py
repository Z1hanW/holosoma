import lzfse


def decompress(data: bytes) -> bytes:
    return lzfse.decompress(data)


def compress(data: bytes) -> bytes:
    return lzfse.compress(data)
