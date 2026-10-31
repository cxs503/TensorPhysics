"""pytest 根配置：确保测试导入的是**本工作树**的 ``src/tensorphysics``。

平台上其他 worktree 的 editable 安装可能抢占 ``tensorphysics`` 包名，因此在
``sys.path`` 首位插入本仓库的 ``src`` 目录，保证测试对象与工作树一致。
"""
import pathlib
import sys

_SRC = pathlib.Path(__file__).resolve().parent / "src"
if _SRC.is_dir() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))