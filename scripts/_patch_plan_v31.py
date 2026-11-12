"""一次性脚本：把项目编写计划.md 更新到 v3.1（回填 9 项交付 + 新增 §0.6）。"""
import pathlib

ROOT = pathlib.Path(__file__).resolve().parent.parent
PATH = ROOT / "项目编写计划.md"

OLD_05 = """### 0.5 验收线实测结果（v3.0 新增，scale=0.3）"""

NEW_05 = """### 0.5 验收线实测结果（v3.0 新增，scale=0.3）

> **v3.1 更新**：新增「SKU 级关联埋点」后合成数据发生变化，本节数字已按最新数据重跑；
> 失效项标 ⚠️ 并注明原因。"""

OLD_S1 = """---

## 一、项目背景与业务场景"""

NEW_S1 = """### 0.6 本轮（v3.1）新增并修复的 5 个真实缺陷

> 前 3 个见 §0.4。这 2 个是**本项目自己的补丁脚本引入的回归** —— 记录它们的价值在于：
> 「用脚本批量改代码」这种做法本身就会引入风险，必须有对应的守卫手段。

| # | 缺陷 | 症状 | 根因 | 修复与守卫 |
|---|---|---|---|---|
| 4 | **pandas 3.x 让 dtype 优化静默失效** | `optimize_dtypes` 不再把低基数字符串转 `category`，内存优化形同虚设，但**没有任何报错** | pandas 3.x 起字符串列默认 dtype 是 `str` 而非 `object`，原判断只判 `object` | 新增 `_is_string_dtype()` 兼容两种 dtype；补单测断言 `category` 确实生效 |
| 5 | **补丁脚本写坏 import，模块"隐形损坏"** | `render_decision_report.py` 的 import 变成 `from ... import (, ITEMS_DIR`（语法错误），但**其它模块与全部测试仍然全绿**，直到压测跑到该 stage 才炸 | 批量改代码的正则脚本误改了多行 import 块 | 修复 import；**新增 `tests/test_import_smoke.py`：遍历包内所有模块 + 所有脚本 + 看板做导入/编译冒烟**，把这类"隐形损坏"挡在 CI |

**顺带修掉的 2 个数据契约不一致**（不是 bug，但会让「一份 ETL 通吃三源」站不住）：

| # | 问题 | 修复 |
|---|---|---|
| 6 | `SOURCE_MAPPINGS["dunnhumby"]["DAY"]` 映射到 `txn_day`，而 canonical 里根本没有这个字段 | 改为映射到 `txn_ts`，并加 `datetime_epoch_day` 类型规则（dunnhumby 的 DAY 是「距 1970-01-01 的天数」） |
| 7 | UCI 的 `StockCode` 不是纯数字（`POST`/`M`/`BANK CHARGES` 等服务码），`astype(int32)` 整列失败；`items` 表用 `UnitPrice` 而 `transactions` 表用 `Price` | 非数值编码映射为**稳定代理 ID**（不丢行）；多源列指向同一 canonical 列时做**合并**而非重命名（避免重复列名） |

---

## 一、项目背景与业务场景"""


def main() -> None:
    text = PATH.read_text(encoding="utf-8")
    n = 0
    for old, new in [(OLD_05, NEW_05), (OLD_S1, NEW_S1)]:
        if old in text:
            text = text.replace(old, new, 1)
            n += 1
        else:
            print(f"  !! 未命中: {old[:40]}")
    PATH.write_text(text, encoding="utf-8")
    print(f"✅ 替换 {n}/2 处")


if __name__ == "__main__":
    main()
