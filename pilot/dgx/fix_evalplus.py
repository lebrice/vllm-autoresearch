"""Fix EvalPlus 0.3.1 find_zero bookkeeping without changing its test oracle.

The special oracle returns early before recording progress, so correct solutions
are incorrectly marked failed by untrusted_check's completeness check.
"""
import hashlib
import importlib.metadata
from pathlib import Path

import evalplus.eval


def main():
    path = Path(evalplus.eval.__file__)
    original = path.read_text()
    before = 'assert abs(_poly(*inp, out)) <= atol\n                            continue'
    after = ('assert abs(_poly(*inp, out)) <= atol\n'
             '                            details[i] = True\n'
             '                            progress.value += 1\n'
             '                            continue')
    if before in original:
        assert original.count(before) == 1
        path.write_text(original.replace(before, after))
    elif after not in original:
        raise RuntimeError("Unexpected EvalPlus oracle implementation; inspect before patching")
    print({"evalplus_version": importlib.metadata.version("evalplus"),
           "patched_file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
           "patch": "find_zero-progress-bookkeeping-v1"})


if __name__ == "__main__":
    main()
