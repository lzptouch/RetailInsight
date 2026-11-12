"""修复 macOS 上 LightGBM 的 libomp.dylib 链接问题。

背景
----
LightGBM 的 ``lib_lightgbm.dylib`` 通过 ``@rpath/libomp.dylib`` 引用 OpenMP 运行库，
但它默认只在 Homebrew 的几个固定路径里找；这些路径不存在时
``import lightgbm`` 会直接抛 ``OSError: Library not loaded``。

scikit-learn 自带了一份兼容的 ``libomp.dylib``（sklearn 内部用 OpenMP），
本脚本把它复制到 venv 的 ``lib/`` 并写入 ``lib_lightgbm.dylib`` 的 rpath，
使 LightGBM 可以直接使用，无需 brew install libomp。

用法
----
    python scripts/fix_lightgbm_dylib.py

幂等：重复执行不会破坏已有修复。修复后 ``import lightgbm`` 可正常工作。
"""
import shutil
import subprocess
import sys
from pathlib import Path


def find_venv_libomp(venv: Path) -> Path | None:
    """在 venv 内寻找 sklearn 附带的 libomp.dylib。"""
    for candidate in venv.glob("lib/python*/site-packages/sklearn/.dylibs/libomp.dylib"):
        return candidate
    return None


def main() -> int:
    venv = Path(sys.executable).resolve().parent.parent
    target_lib = venv / "lib"

    libs = list(venv.glob("lib/python*/site-packages/lightgbm/lib/lib_lightgbm.dylib"))
    if not libs:
        print("[fix-lgbm] 未找到 lib_lightgbm.dylib，跳过（可能未安装 lightgbm）")
        return 0
    lib = libs[0]

    src = find_venv_libomp(venv)
    if src is None:
        print("[fix-lgbm] 未找到 sklearn 自带的 libomp.dylib，无法自动修复。")
        print("           请手动执行： brew install libomp")
        return 1

    target_lib.mkdir(parents=True, exist_ok=True)
    dst = target_lib / "libomp.dylib"
    shutil.copy2(src, dst)
    print(f"[fix-lgbm] 已复制 {src.name} -> {dst}")

    subprocess.run(["install_name_tool", "-add_rpath", str(target_lib), str(lib)], check=True)
    subprocess.run(["install_name_tool", "-change", "@rpath/libomp.dylib", str(dst), str(lib)],
                   check=True)
    print(f"[fix-lgbm] 已写入 rpath：{target_lib}")
    print("[fix-lgbm] 完成，现在可以正常 import lightgbm")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
