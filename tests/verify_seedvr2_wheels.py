"""Windows CI: check real wheel hashes and DLL/API imports without a GPU.

Run only in a disposable Python 3.12 / Torch 2.7.1+cu126 environment.
"""
import hashlib
import platform
import sys
import tempfile
from pathlib import Path
from urllib.parse import unquote, urlsplit

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from atelier import seedvr2_acceleration as acceleration
from scripts import setup_media


def main():
    import torch
    if (platform.system() != "Windows" or sys.version_info[:2] != (3, 12)
            or torch.__version__.split("+")[0] != "2.7.1" or torch.version.cuda != "12.6"):
        raise RuntimeError("Use a disposable Windows Python 3.12 / Torch 2.7.1+cu126 environment.")
    before = torch.__version__
    with tempfile.TemporaryDirectory() as temporary:
        for wheel in (acceleration.TRITON, acceleration.SAGE, acceleration.FLASH):
            archive = Path(temporary) / unquote(Path(urlsplit(wheel.url).path).name)
            setup_media.download(wheel.url, archive)
            with archive.open("rb") as stream:
                if hashlib.file_digest(stream, "sha256").hexdigest() != wheel.sha256:
                    raise RuntimeError(wheel.name + " checksum mismatch.")
            setup_media.pip(sys.executable, "--no-deps", "--only-binary", ":all:", archive)
    import triton
    import flash_attn_2_cuda  # noqa: F401
    from flash_attn import flash_attn_varlen_func
    from sageattention import sageattn_varlen
    assert callable(flash_attn_varlen_func) and callable(sageattn_varlen)
    assert torch.__version__ == before
    assert triton.__version__.split(".")[:2] == ["3", "3"]
    print("Windows wheel checks passed: hashes, attention APIs, DLL imports, unchanged Torch.")


if __name__ == "__main__":
    main()
