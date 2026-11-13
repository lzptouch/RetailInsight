# RetailInsight — 商超会员用户画像与决策支持系统

模拟数据服务公司为连锁商超提供会员画像分析：从数据生成/清洗到 RFM 分群、K-Means 聚类、流失预测、关联规则，最终产出可执行的会员运营与采购建议。详见 `项目编写计划.md`（含需求评审记录与 v3.3 全量核查结论）与 `docs/plan/功能实现差距清单.md`（计划-代码差距与实施方案）。

## 核心结果（scale=0.3：3.6 万注册会员 / 31.4 万交易 / 217 万明细，有效分析窗口 2020-01 ~ 2022-07 共 31 个月）

- **趋势分析**：月度/季度环比与同比；季节性双峰（春节档 + 双 11 促销季），旺季约为淡季基准 **12.8 倍**
- **RFM 8 类客群**：重要价值客户 6,531 人（18.3%）贡献 42.5% GMV；流失风险客户 13,485 人（37.8%）
- **K-Means 聚类**：4 簇（高价值活跃 / 一般维持 / 低频低额 / 沉睡流失），轮廓系数 **0.4165**（验收线 0.35 达标），与生成器埋的 8 类先验人群 ARI 0.134；**神谕上界 −0.056**，说明消费体量特征无法分离行为型人群（详见下方「关键取舍」）
- **模型选型对比**：K-Means 0.4165 / GMM 0.2536（碎群 0.7%）/ DBSCAN 0.0422（噪声 5.2%）→ **选 K-Means**，结论由规则化准则给出
- **关联规则**：品类级 **177 条**（lift>1.5，4 组预埋组合全部回收，最强 lift 8.30）、**SKU 级 48 条**（全「单品 A → 单品 B」，lift 中位 46.4）、**分门店 1,092 条 / 15 店全覆盖**
- **CLV 分层**：4 档，主口径 **BG/NBD + Gamma-Gamma**（未来 12 个月期望毛利）。91 天时间外回测：金额 MAE **229.3 vs 简化法 270.8（降 15.3%）**、偏差从 **+74.4 元**压到 **−1.4 元**；只买过一次（14,897 人）与注册未购（462 人）分别退简化法 / 恒 0，逐行标在 `clv_method`
- **P1 扩展标签 18 个**：促销响应率 / 券使用率 / 跨店消费 / 生鲜占比 / 进口偏好 / 复购周期中位数 / 潜力指数 / 生命周期阶段 等，画像宽表 **54 列**（新增 `clv_bgnbd` / `clv_simple` / `clv_method`）
- **月度复购率**：20.3%~48.0%（月度口径，与漏斗的累计口径区分）
- **门店对比**：末期 3 月 vs 前 3 月增长率 + VIP 占比结构差异（+7.3~+9.7 个百分点）
- **流失预测**：**RandomForest AUC 0.830**（主模型，由验收线自动选型），较 LR 基线（0.7995）**+0.0305 达标**；标准 KS **0.4936**；Top20% 名单捕获 46.77% 真实流失者（lift 2.34，基线流失率 36.7% 下理论上限 2.72 的 86%）；LightGBM 对照 AUC 0.8179（+0.0184 未达标）
- **交付物**：Excel 报表（6 sheet）、**14 张分析图**、决策建议报告（8 条建议全部含量化测算）、**客群画像卡片**（8 群各一页）、**性能压测报告**（scale 5）、Streamlit 4 页看板（**plotly 点选下钻**）、数据质量报告（md + json + **自包含 HTML**）、5 个**已执行**的分析 Notebook
- **工程**：DuckDB 星型模型（SQL 版 RFM 与 pandas 版 **35,650 行零差异**）、明细按月分区（读 3 个月仅占全量 20.2%）、**大规模分块流式生成**（20 万会员 / 1,200 万明细 5.2s、峰值 1.0GB）、**200 个测试 / 覆盖率 82.78%**（含 Streamlit `AppTest` 看板端到端）、关联规则网络图、看板首屏计时（实测 0.51s）

## 关键取舍（面试时最值得讲的三个判断）

1. **聚类先验上界检验**：直接用生成器埋的真实 8 类人群当标签算轮廓系数只有 **−0.056**，证明「消费体量」这一类特征在数学上就无法分离行为型人群。再调 K、再调参都无济于事 —— 因此改为在多个候选特征空间里择优（`rfm3` ARI 0.134 最高），并把「哪个维度没有区分度」写进结论，而不是硬凑验收线。
2. **主模型不是我选的，是验收线选的**：LightGBM AUC 0.8179（增益 +0.019）打不到 +0.03，RandomForest AUC 0.830（增益 **+0.0305**）刚好过线 —— 于是我把选型规则**写死在代码里**（`select_main_model`：达标组 → KS 最高者 → 全不达标才取 AUC 最高者）。GBDT 打不到线的原因我跑了 6 组敏感性实验才敢下结论：增益与「训练期数」正相关、与「特征维度数」负相关（12→31 维时 LR 基线同步变强，相对增益被摊薄），**GBDT 口径下 +0.03 是当前数据窗口下的天花板**。
3. **KS 采用标准定义 `max|TPR-FPR|`**，而不是自定义的 lift 型统计量 —— 后者上限被基线率锁死，与「KS ≥ 0.30」的业界验收线不可比。

## 快速开始

```bash
# 1) 安装（推荐虚拟环境）
python -m venv .venv && source .venv/bin/activate
make install                 # 等价于 pip install -e ".[dev]"
# 全链路必需的 duckdb / plotly / shap 已在**主依赖**里（v3.4 修复漏声明）；
# lightgbm 仍是可选：不装会自动降级为 sklearn HistGB，并在 metrics_churn.json 的 backend 字段留痕

# 2) macOS 上若 lightgbm 报 libomp 缺失（会降级为 sklearn HistGB，指标略低）
python scripts/fix_lightgbm_dylib.py

# 3) 一键跑通全链路（14 个 stage，约 2.5 分钟；第 15 个 dashboard 是交互式，不在 all 里）
make all
# 或：python -m retailinsight.cli run --stage all --scale 0.3
# stage 清单：gen etl rfm cluster rules rules_sku rules_store churn
#             profile warehouse reports report cards segment_compare
#             （dashboard = 交互式看板，单独跑）

# 4) 测试（带覆盖率门禁 ≥60%）
make test                    # 169 个用例
make cov                     # 覆盖率明细

# 大规模生成（分块流式，峰值内存 ≈ 单块）
python -m retailinsight.cli run --stage gen --members 200000 --chunk-size 20000

# 5) 看板（两种等价方式）
make dashboard               # = python -m retailinsight.cli run --stage dashboard
streamlit run dashboard/app.py

# 6) 分析 Notebook（5 个，均已执行并保存输出）
python scripts/build_notebooks.py && jupyter lab notebooks/

# 7) 性能压测（逐 stage 计时 + 峰值内存采样）
python scripts/exp_scale5.py --scale 5
```

## 目录结构

```text
retailinsight/     # 核心包：io(schema/loader) / etl(clean/pipeline/quality/quality_html/warehouse) /
                   # features(rfm/tags/p1_tags/behavior/churn_features/cluster_features/common) /
                   # models(churn/cluster/cluster_compare/rules/clv/segment_compare) /
                   # viz(charts) / reports(core/excel/render_decision_report/profile_cards) / cli
scripts/           # generate_synthetic.py 合成数据生成器（8 类人群 + 门店档位/人群品类乘子
                   #   + 关联埋点 + 脏数据埋点 + --members/--chunk-size 分块流式生成）
                   # fix_lightgbm_dylib.py macOS libomp 修复 / build_notebooks.py Notebook 生成
                   # exp_churn_sensitivity.py 流失模型增益敏感性实验 / exp_scale5.py scale 压测
dashboard/         # Streamlit 4 页看板（plotly 点选下钻 + 首屏计时）
notebooks/         # 01_EDA → 02_RFM → 03_cluster → 04_rules → 05_churn（5 个，已执行）
docs/              # 数据字典 / 指标体系 / 数据质量报告(.md+.json+.html) / 决策建议报告 /
                   # 客群画像 / 客群方法差异论证 / 性能报告 / 项目深度讲解与面试指南
                   # plan/功能实现差距清单.md（计划-代码差距与实施方案）
reports/           # Excel 报表 + 14 张图 + metrics_*.json（产物，不入 git）
tests/             # pytest 191 例（含防泄漏专项、聚类护栏、阈值调优、索引错位回归、
                   #   依赖声明 AST 校验、分块生成一致性、看板 AppTest 端到端）
```

## 数据说明（合规）

- 全部数据由 `scripts/generate_synthetic.py` 算法生成，**不含任何真实个人信息**；canonical 列名由生成器**直接产出**，不再经过任何外部源列名映射层（该映射层与下载器已于 v3.5 整体删除，理由见 `项目编写计划.md` §0.13）
- 生成器支持**分块流式生成**：`--members 200000 --chunk-size 20000` 实测 1,200 万条明细 5.2s 完成，峰值内存约 1.0GB、落盘 108MB（一次性生成同规模约需 4GB）。`chunk_size=0` 走原路径，产物与历史基线**字节级一致**
- **原始数据范围 2010-01-01 ~ 2022-07-31**：其中 1,511 行是生成器故意注入的「时间乱序」脏数据（全部标记 `is_logic_error=True` 并被剔除）；**有效分析窗口 2020-01-01 ~ 2022-07-31（308,956 笔有效交易）**
- 合成数据**已具备**门店级与人群级品类结构差异（v3.5 引入「城市档位 × 先验人群」品类乘子矩阵）：门店级差异度实测 **0.0502（最高 0.1244）**，落在真实商超 0.05~0.30 的下沿；规则法客群级差异度 0.0002~0.0095（受 ARI≈0.13 稀释）。因此 S8 门店差异化**方向可验证、量级偏保守**，客群级品类偏好仍只能算「弱差异」—— 两处口径与局限均已写入决策建议报告与客群画像卡片

## 已知限制与后续计划

**v3.0 收敛的 9 项范围（R-1~R-9）已在 v3.1 全部交付，v3.3 / v3.4 两轮核查确认代码与产物一致；v3.5 又删掉 1 项「幻象能力」（下载器 + 列名映射层）并补齐 3 项真实缺口（大规模生成、看板自动化测试、门店/人群品类差异化）。**

尚存的已知限制（均已如实写入文档）：

- 流失模型在 **GBDT 口径**下增益 +0.019 打不到 +0.03 验收线（主模型已改选 RandomForest +0.0305 达标；GBDT 口径经 6 组敏感性实验确认是数据窗口的天花板）
- 聚类的 ARI 上限约 0.13：8 类先验人群中的「促销敏感 / 周末囤货 / 季节性」在消费体量上本质重合，需行为时序特征才能进一步区分
- 客群级品类偏好仍只有**弱差异**（差异度 0.0002~0.0095）：品类金额分布是 Zipf 长尾，头部品类会出现在所有客群 TOP3 里；而规则法 RFM 客群与 8 类先验人群的 ARI 仅 0.13，人群级差异在聚合后被稀释。门店级差异度 0.0502 已可支撑 S8 的**方向验证**，但绝对值仍需甲方真实门店-品类数据校准
- 聚类与流失模型在**合成数据**上评估，真实业务数据上的口径需重新校准
- `members.gender` 的 3.94% 缺失**不做填补**（保留「未知」为独立类别，猜性别比留空更危险）—— 已在质量报告第三节显式列出
- `ruff format --check` 未纳入门禁（仓库尚未统一格式化，`make lint` = `ruff check .`，当前全绿）
- `.gitignore` 的路径规则已全部锚定仓库根（`/reports/`、`/data/...`）—— 早期未锚定时把**源码包** `retailinsight/reports/` 一起忽略了，导致 clone 出来的仓库不可运行，v3.3 已修复并纳入版本控制
- **依赖声明有守卫**：`tests/test_v34_audit_fixes.py` 会 AST 扫描全部 import，断言每个第三方包都在 `pyproject.toml` 的主依赖里（唯一例外 `lightgbm`，且必须「try/except + metrics 记 backend」两条同时成立）—— 这条测试的由来：v3.4 发现 `plotly`/`shap`/`psutil` 从未声明、`duckdb` 被错放进可选组，导致新机器上 `make all` 与看板直接崩
- **测试不得污染产物**：`write_quality_report()` 的 HTML 路径跟随 md 入参，`tests/test_v34_audit_fixes.py` 会逐字节比对 `docs/数据质量报告.*` 在测试前后是否被改动
- **「幻象能力」已清理**：v3.5 删除了 `scripts/download_data.py` 与 `io/schema.py::apply_mapping()` 一整套（含 `SOURCE_MAPPINGS` / `SOURCE_DTYPES` / `REQUIRED_COLUMNS` / `data/external/` 与对应测试）。它长期被列为 ✅ 交付物，但三个数据源的 sha256 **全是 `None`**、两个源没有 url、`apply_mapping()` 的唯一消费方就是下载器本身 —— **没有任何真实调用链**。留着它等于给面试官一个「代码就绪」的假象；删掉后 canonical 口径由生成器直接产出，链路更短也更诚实。教训：**能力必须有真实调用链，否则是负债**
