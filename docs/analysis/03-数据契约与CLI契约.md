# 数据契约与 CLI 契约

> 对应域：D1·D2·D9。本篇对应 Kquery 的「03-接口与契约」——本项目没有 HTTP 接口，契约的载体是**数据 schema、stage 产物、降级协议**。

## 〇、摘要

三类契约：(1) **canonical 数据契约**——5 张表的列名与类型由生成器直接产出，主链路不经过任何外部源映射层；(2) **stage 产物契约**——每个 stage 的输入/输出产物固定，stage 间只通过产物通信；(3) **降级契约**——可选依赖缺失时的行为与留痕方式，调用方可以**从产物本身**判断本次运行走了哪条路径。

## 一、canonical schema（`retailinsight/io/schema.py`）

| 表 | 列 |
|---|---|
| members | `member_id, reg_date, gender, age_band, city_tier, store_id` |
| stores | `store_id, city, city_tier, area_m2, open_date` |
| products | `sku, category, sub_category, brand, base_price` |
| transactions | `txn_id, member_id, store_id, txn_ts, amount, discount` |
| items | `txn_id, sku, qty, unit_price` |

领域常量同模块定义：60 个品类（`CATEGORIES`）、品类别名表（`CATEGORY_ALIASES`）、年龄段权重、4 组关联埋点（`ANCHOR_COMBOS`）、缺失哨兵 `MISSING_MEMBER_ID`。

> **v3.5 架构决策**：原先的第三方数据源映射层（`SOURCE_MAPPINGS`/`apply_mapping()` 等）已**整体删除**——它唯一的消费方是开源数据集下载器，主链路从不经过（三源 sha256 全为 None，校验形同虚设）；且项目的数据故事（中国商超 + 8 类可验证先验人群）只能由合成数据承载。生成器**直接产出 canonical 列名**，映射层没有存在的必要。

## 二、清洗后派生列契约（`etl/clean.py` 产出）

| 派生列 | 含义 |
|---|---|
| `is_return` | `amount < 0`（退货不删除，单列分析） |
| `net_amount` | 净额 = 实付（退货自然为负） |
| `member_type` | 会员 / 非会员（缺失 member_id 的填充后归类） |
| `is_logic_error` | 逻辑异常（含生成器故意注入的 1,511 行时间乱序），分析时剔除 |
| `txn_month` | 分区键（`io/loader.py: derive_txn_month`） |

## 三、stage 产物契约

| stage | 输入 | 输出（路径常量见 `settings.py`） |
|---|---|---|
| gen | —（scale/seed/members/chunk_size 参数） | `data/raw/*.parquet`（明细可分区） |
| etl | raw | `data/interim/*` + 质量报告三件套 |
| rfm | interim/transactions | `MART_RFM` |
| cluster | MART_RFM + interim | `MART_CLUSTER` + `reports/metrics_cluster.json` |
| rules / rules_sku / rules_store | interim 明细 | `MART_RULES` + SKU/门店规则表 |
| churn | interim + ITEMS_DIR | `MART_CHURN` + `reports/metrics_churn.json` |
| profile | members/txn/items/products + MART_RFM + MART_CHURN | `MART_PROFILE` + `reports/metrics_clv.json` |
| warehouse | interim | `data/*.duckdb` 星型模型 + 一致性断言 |
| reports | MART_RFM/CHURN/RULES | Excel + 13 图 |
| report | 各 mart + metrics json | `reports/决策建议报告.md`（模板回填） |
| cards | MART_PROFILE + MART_RFM | `reports/客群画像.md` |
| segment_compare | MART_PROFILE/RFM/CLUSTER | `mart_segment_conflict` + `reports/客群方法差异论证.md` |
| dashboard | marts（只读） | 交互页面（不落盘） |

**约定**：stage 失败直接非零退出（`subprocess.run(check=True)` / 断言抛出）；产物写盘采用覆盖式幂等，任意 stage 可单独重跑。

## 四、CLI 契约（`retailinsight/cli.py:161-174`）

```
python -m retailinsight.cli run --stage <15选1|all>
    [--scale 0.3] [--seed 42] [--port 8501]
    [--members N]      # 覆盖 scale 计算的会员数
    [--chunk-size N]   # gen 分块大小（会员数/块），大规模必需
```

- `--stage all` = `ALL_STAGES`（14 个，dashboard 被 `INTERACTIVE_STAGES` 排除）。
- 每个 stage 结束打印 `[cli] stage=X elapsed=Ns`。
- 看板数据目录可用环境变量 `RETAILINSIGHT_MARTS` 重定向（`dashboard/app.py`）——AppTest 测试隔离与多份产物对比都靠它。

## 五、降级契约（3 条，全部显式留痕）

| 缺失依赖 | 降级行为 | 留痕位置 |
|---|---|---|
| `lightgbm` | GBDT 主模型退为 sklearn HistGB（指标略低） | `metrics_churn.json` 的 `backend` 字段 |
| `lifetimes` | CLV 主口径 BG/NBD 退为简化法 | `mart_member_profile.clv_method = "simple_fallback"`，且 CLI 打印警告（`cli.py:110`） |
| `shap` | 单会员归因退为 permutation importance | 运行告警 |

反契约（明确禁止）：**静默降级产出空列**。v3.4 的教训是 duckdb/plotly/shap 漏声明时「不报错、只是列全空」，因此这 4 项被提升为主依赖（`pyproject.toml` 注释留有决策记录）。

## 六、指标文件契约（`reports/metrics_*.json`）

| 文件 | 生产者 | 关键字段 |
|---|---|---|
| `metrics_churn.json` | churn | AUC/KS/lift、backend（lightgbm/histgb）、阈值与成本系数 |
| `metrics_cluster.json` | cluster | 轮廓系数、ARI、各特征空间评估、神谕上界 |
| `metrics_clv.json` | profile | 双口径 MAE/Spearman/偏差、cutoff、horizon_months、gross_margin |

这些 json 是**报告回填与面试答辩的事实来源**，也是降级留痕的载体。

## 七、边界与已知缺口

1. 契约靠约定 + 测试守卫，无 pandera/pydantic 式的运行时 schema 校验框架（`tests/etl/test_schema.py` 承担了主要校验职责）。
2. 产物无版本号/ schema 演进机制（作品集规模，重跑即迁移）。
3. metrics json 无统一 schema 文档，字段以生产者代码为准。

## 维护触发

改 canonical 列 / 新增产物 / 新增降级路径 → 同步改本篇与 `tests/guards/` 相关守卫。
