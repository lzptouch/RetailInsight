# RetailInsight — 商超会员用户画像与决策支持系统

模拟数据服务公司为连锁商超提供会员画像分析：从数据生成/清洗到 RFM 分群、K-Means 聚类、流失预测、关联规则，最终产出可执行的会员运营与采购建议。详见 `项目编写计划.md`（含需求评审记录）与 `docs/plan/功能实现差距清单.md`（计划-代码差距与实施方案）。

## 核心结果（scale=0.3：3.6 万会员 / 31 万订单 / 219 万明细，窗口 2020-01 ~ 2022-07 共 31 个月）

- **趋势分析**：月度/季度环比与同比；季节性双峰（春节档 + 双 11 促销季），旺季约为淡季基准 **12.9 倍**
- **RFM 8 类客群**：重要价值客户 6,600 人（18.5%）贡献 42.7% GMV；流失风险客户 13,526 人（38.0%）
- **K-Means 聚类**：4 簇（高价值活跃 / 低频低额 / 沉睡流失 / 一般维持），轮廓系数 **0.416**（验收线 0.35 达标），与生成器埋的 8 类先验人群 ARI 0.135；**神谕上界 -0.056**，说明消费体量特征无法分离行为型人群（详见下方「关键取舍」）
- **流失预测**：LightGBM AUC **0.824**，较 LR 基线（0.797）**+0.027**；标准 KS **0.311**；Top20% 名单捕获 46.9% 真实流失者（lift 2.34，基线流失率 36.8% 下理论上限 2.72 的 86%）；RandomForest 对照 AUC 0.829
- **关联规则**：FP-Growth 挖出 361 条规则（lift>1.5），4 组预埋组合全部回收（洗发水→护发素 lift 6.06、纸尿裤→啤酒 lift 2.23）
- **交付物**：Excel 报表（6 sheet）、12 张分析图、决策建议报告（8 条建议全部含量化测算）、Streamlit 4 页看板、数据质量报告、5 个分析 Notebook

## 关键取舍（面试时最值得讲的三个判断）

1. **聚类先验上界检验**：直接用生成器埋的真实 8 类人群当标签算轮廓系数只有 **-0.056**，证明「消费体量」这一类特征在数学上就无法分离行为型人群。再调 K、再调参都无济于事 —— 因此改为在多个候选特征空间里择优，并把「哪个维度没有区分度」写进结论，而不是硬凑验收线。
2. **加入独立验证集后，AUC 增益从 +0.031 降到 +0.027**：早期口径没有验证集，训练窗口紧贴测试期；补上验证集后训练窗口被迫前移一个季度。**这是时间外推的真实难度，也是原方案「看起来达标」的原因。**
3. **KS 采用标准定义 `max|TPR-FPR|`**，而不是自定义的 lift 型统计量 —— 后者上限被基线率锁死，与「KS ≥ 0.30」的业界验收线不可比。

## 快速开始

```bash
# 1) 安装（推荐虚拟环境）
python -m venv .venv && source .venv/bin/activate
make install                 # 等价于 pip install -e ".[dev]"

# 2) macOS 上若 lightgbm 报 libomp 缺失（会降级为 sklearn HistGB，指标略低）
python scripts/fix_lightgbm_dylib.py

# 3) 一键跑通全链路（gen → etl → rfm → cluster → rules → churn → profile → reports → report，约 100 秒）
make all
# 或：python -m retailinsight.cli run --stage all --scale 0.3

# 4) 测试（带覆盖率门禁 ≥60%）
make test                    # 59 个用例
make cov                     # 覆盖率明细

# 5) 看板
streamlit run dashboard/app.py

# 6) 分析 Notebook（5 个，均已执行通过）
python scripts/build_notebooks.py && jupyter lab notebooks/
```

## 目录结构

```text
retailinsight/     # 核心包：io(schema/loader) / etl(clean/pipeline/quality) /
                   # features(rfm/tags/behavior/churn_features/cluster_features/common) /
                   # models(churn/cluster/rules) / viz(charts) / reports(core/excel/render_decision_report) / cli
scripts/           # generate_synthetic.py 合成数据生成器（8 类人群 + 关联埋点 + 脏数据埋点）
                   # fix_lightgbm_dylib.py macOS libomp 修复 / build_notebooks.py Notebook 生成
dashboard/         # Streamlit 4 页看板
notebooks/         # 01_EDA → 02_RFM → 03_cluster → 04_rules → 05_churn（5 个）
docs/              # 数据字典 / 指标体系 / 数据质量报告 / 决策建议报告 / 项目深度讲解与面试指南
                   # plan/功能实现差距清单.md（计划-代码差距与实施方案）
reports/           # Excel 报表 + 图表 + metrics_*.json（产物，不入 git）
tests/             # pytest 59 例（含防泄漏专项、聚类护栏、阈值调优、特征防泄漏）
```

## 数据说明（合规）

- 全部数据由 `scripts/generate_synthetic.py` 算法生成，**不含任何真实个人信息**
- 开源数据集（Online Retail II / Dunnhumby / Instacart）为 P1 可插拔数据源，schema 映射见 `retailinsight/io/schema.py`；`SOURCE_MAPPINGS` 映射层已定义，下载器与统一入口仍在计划中
- 合成数据下各门店品类结构高度同质（差异度 <0.001），因此门店级策略（S8）无法被量化验证 —— 该局限已写入决策建议报告

## 已知限制

- 关联规则目前为品类级（60 类）；SKU 级（5000 维）需 Top300 裁剪 + 分门店分批，见计划 §8.3
- 流失模型 AUC 增益 +0.027 未达计划的 +0.03 验收线（原因见「关键取舍」第 2 条）
- 聚类的 ARI 上限约 0.14：8 类先验人群中的「促销敏感 / 周末囤货 / 季节性」在消费体量上本质重合，需要行为时序特征（计划 §7.3 的 P1 扩展标签）才能进一步区分
- DuckDB 星型模型、Polars 备选路径、`--scale 5` 压测尚未实现（P1/P2）
