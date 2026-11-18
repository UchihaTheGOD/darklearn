"""Build darklearn's optional C core — stdlib only.

    python3 scripts/build_c.py

Compiles darklearn/_darkcore.c with cc/gcc/clang (whichever is found)
into darklearn/_darkcore<EXT_SUFFIX> next to the Python files, with
-O3 -fPIC -shared and the CPython include dir from sysconfig.

The C core is strictly optional:
- No compiler found  -> prints a message and exits 0; darklearn keeps
  working on the pure-Python kernels.
- Compile fails      -> prints the compiler output and exits 1.
- Success            -> prints the .so path; `from darklearn import
  darkcore; darkcore.C_AVAILABLE` becomes True.

Run from anywhere; the script finds the repo root from its own path.
"""

import shutil
import subprocess
import sys
import sysconfig
from pathlib import Path


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    src = root / "darklearn" / "_darkcore.c"
    if not src.exists():
        print(f"missing source: {src}")
        return 1

    cc = shutil.which("cc") or shutil.which("gcc") or shutil.which("clang")
    if cc is None:
        print("No C compiler found (looked for cc, gcc, clang).")
        print("Skipping the C core — darklearn will use the pure-Python "
              "DarkCore kernels. Install a compiler and re-run this script "
              "to enable the C path.")
        return 0

    include = sysconfig.get_paths()["include"]
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    out = root / "darklearn" / ("_darkcore" + suffix)
    cmd = [cc, "-O3", "-fPIC", "-shared", f"-I{include}",
           str(src), "-o", str(out)]
    print(" ".join(cmd))
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print("C core build FAILED:")
        print(res.stdout)
        print(res.stderr)
        return 1
    if res.stdout.strip():
        print(res.stdout)
    if res.stderr.strip():
        print(res.stderr)
    print(f"C core built: {out}")
    print("Verify with: python3 -c "
          "\"from darklearn import darkcore; print(darkcore.C_AVAILABLE)\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
