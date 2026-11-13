# 悦鲜生活超市 · 会员运营与商品采购决策建议报告

> 数据窗口：{{data_window}}（{{n_months}} 个月）｜ 分析会员 {{n_members}} 人（全量会员 {{n_profile}} 人）｜ 有效订单 {{n_orders}} 笔 ｜ 总 GMV {{gmv_wan}} 万元
> 产出物：`reports/{{excel_name}}`（6 个 sheet）、`reports/figures/`、`data/marts/*`
> **本报告由 `python -m retailinsight.reports.render_decision_report` 自动生成**，所有数字与测算均从 `data/marts/` 产物回填，模板见 `retailinsight/reports/templates/`。数据快照：{{generated_at}}
> 说明：本报告基于合成数据（见 README 合规声明），标注「假设」的参数为行业经验值，上线前需 A/B 验证。
{{scale_line}}

## 〇、执行摘要

1. **价值高度集中**：{{vip_share}} 的会员（{{n_vip}} 人）贡献 {{vip_gmv_share}} 的 GMV，人均净消费 {{vip_arpu}} 元；叠加次大客群后 {{top2_share}} 的会员贡献 {{top2_gmv_share}} 的 GMV。
2. **流失风险积聚**：{{churn_risk_share}} 的会员（{{n_churn_risk}} 人）落入「流失风险」客群，平均 {{churn_risk_avg_recency}} 天未消费，仅贡献 {{churn_risk_gmv_share}} 的 GMV；模型标记的高危会员 {{n_high_risk}} 人。
3. **强季节性**：峰值月 {{peak_month}}（{{peak_gmv_wan}} 万元）是淡季基准（{{s7_trough_gmv_wan}} 万元，最低 3 个月均值）的 **{{s7_ratio}} 倍**，采购与排班必须按季节结构走。
4. **转化瓶颈在复购→高频**：注册→首购 {{conv_first}}，首购→复购 {{conv_repeat}}，复购→高频 {{conv_high_freq}}；处于「有复购但未达高频」的会员有 {{n_repeat_not_high}} 人，是最大的一批可运营人群。
5. **关联结构可直接转化**：挖掘出 {{n_rules}} 条强关联规则（lift>1.5），其中 {{rule_洗发水_护发素}}、{{rule_火锅底料_牛羊肉}} 可直接落到陈列与捆绑。
6. **模型口径已按时间切分校正**：{{model_verdict_paragraph}}

---

## 一、关键发现

| 编号 | 发现 | 证据 |
|---|---|---|
| F1 | 季度同比：最后一个完整季度 {{q_last}} 为 {{q_last_yoy}}，全期最高同比出现在 {{q_yoy_max_q}}（{{q_yoy_max}}）；**{{q_last_incomplete}} 仅含 1 个月，为口径假象，不参与同比解读** | 图 01c、sheet 1b |
| F2 | 最新完整月同比 {{m_last_yoy}}，整体仍处放量但增速回落通道 | 图 01b、sheet 1 |
| F3 | 季节性双峰：峰值月 {{peak_month}} 为谷底 {{trough_month}} 的 {{season_ratio}} 倍，11 月单月可达 {{nov_gmv_wan}} 万元 | 图 01/01b |
| F4 | 会员订单占比 97.0%，非会员交易极少，会员体系是唯一抓手 | sheet 1 |
| F5 | 重要价值客户 {{n_vip}} 人（{{vip_share}}）贡献 {{vip_gmv_share}} GMV，人均净消费 {{vip_arpu}} 元 | 图 05/06 |
| F6 | 流失风险客户 {{n_churn_risk}} 人（{{churn_risk_share}}），平均 {{churn_risk_avg_recency}} 天未消费，仅贡献 {{churn_risk_gmv_share}} GMV | 图 05/06 |
| F7 | 高危流失会员 {{n_high_risk}} 人，历史人均净消费 {{high_risk_arpu}} 元 | 图 07/08、sheet 5 |
| F8 | 会员漏斗：注册→首购 {{conv_first}}，首购→复购 {{conv_repeat}}，复购→高频 {{conv_high_freq}} | 报表 funnel |
| F9 | Top5 品类（{{top5_categories}}）合计 {{top5_share}} GMV；A 类品类 {{n_abc_a}}/{{n_categories}} 个 | 图 03 |
| F10 | 销售高峰为 {{peak_hours}} 点，周末订单占比 {{weekend_ratio}} | 图 04 |
| F11 | K-Means 聚类（特征空间 {{cluster_space}}，K={{cluster_k}}）轮廓系数 {{cluster_silhouette}}、与先验人群 ARI {{cluster_ari}}；**神谕上界（直接用真实先验人群当标签）为 {{cluster_oracle}}，说明该特征空间无法完全分离行为型人群** | `reports/metrics_cluster.json`、图 09/10 |

---

## 二、改进建议（S1~S8，每条含测算公式、假设与验证方式）

### 会员服务改进

**S1 高危会员挽留（对应 F7）**

- **结论**：对 {{n_high_risk}} 名高危会员分层触达，预计挽回 GMV **{{s1_gmv_lo}} ~ {{s1_gmv_hi}} 万元**。
- **测算**：`ΔGMV = N_高危 × 触达率 {{s1_touch}} × 挽回率 {{s1_save_lo}}~{{s1_save_hi}} × 月均消费 {{s1_arpu_m}} 元 × {{s1_months}} 个月`
  （`N_高危={{n_high_risk}}`，取自 `mart_churn_score` 中 `risk_level=="高"`）
- **假设**：触达率 {{s1_touch}}（**假设值**）；挽回率 {{s1_save_lo}}~{{s1_save_hi}}（**假设值**，需 A/B 验证）；月均消费恢复到 {{s1_arpu_m}} 元（**假设值**，高危客群历史人均月消费仅 {{high_risk_arpu_month}} 元）
- **验证方式**：触达组 vs 对照组的 90 天复购率差异，周期 2 个月

**S2 重要价值客户防流失（对应 F5）**

- **结论**：把 {{n_vip}} 名重要价值客户纳入专属权益体系，年化可挽回 GMV **{{s2_gmv_lo}} ~ {{s2_gmv_hi}} 万元**，权益成本约 {{s2_cost_wan}} 万元，ROI 约 **{{s2_roi_lo}} ~ {{s2_roi_hi}} 倍**。
- **测算**：`ΔGMV年 = N_VIP × 年自然流失率 {{s2_attr_lo}}~{{s2_attr_hi}} × 干预挽回率 {{s2_save_lo}}~{{s2_save_hi}} × 月均消费 {{vip_arpu_month}} 元 × 12`
  （`N_VIP={{n_vip}}`，月均消费由人均 {{vip_arpu}} 元 ÷ 24 个月推得）
  成本口径：`成本 = N_VIP × 自然流失率 × 30 元/月 × 6 个月`，**只发生在会流失的那部分人身上**（与收益口径对齐）
- **假设**：自然流失率 {{s2_attr_lo}}~{{s2_attr_hi}}（**假设值**，可由 mart_rfm 的 R 值漂移实测校准）；挽回率 {{s2_save_lo}}~{{s2_save_hi}}（**假设值**）；权益成本人均 30 元/月（**假设值**）
- **验证方式**：每月监控 VIP 群 R 值分位，滑出前 40% 即触发干预；季度复盘权益组与对照组的留存差
- **口径说明**：该群人均贡献为全量均值的 {{vip_arpu}} ÷ (GMV/会员数) 倍量级，**流失 1 人的损失 ≈ 挽留 8 名流失风险客户的收益**（模型阈值调优已用同一 8:1 成本比，见 `metrics_churn.json`）

**S3 复购→高频转化专项（对应 F8）**

- **结论**：对 {{n_repeat_not_high}} 名「有复购未达高频」会员上线 30 天消费任务体系，年化 GMV **约 {{s3_gmv_wan}} 万元**（按 {{s3_conv}} 转化率）。
- **测算**：`ΔGMV年 = N × 转化率 {{s3_conv}} × 人均月频提升 {{s3_freq_gain}} 次/月 × 客单价 {{s3_aov}} 元 × 12`
  （`N={{n_repeat_not_high}}`；人均月频提升按**实测**人均月频 {{monthly_freq_member}} 次/月的 100%（即转化者月频翻倍）推得；客单价 {{s3_aov}} 元取自 `mart_rfm` 实测）
- **假设**：转化率 {{s3_conv}}（**假设值**）；转化者月频翻倍（**假设值**）
- **验证方式**：任务组 30 天内下单次数对比

**S4 促销精准化（对应 F5/F6）**

- **结论**：停止全员发券，改按折扣敏感标签精准投放，券成本下降 **{{s4_save_rate}}**（节省约 {{s4_save_wan}} 万元）。
- **测算**：`券成本节省 = 全域发券 {{s4_n_all}} 人 × {{s4_coupon_cost}} 元 − 精准投放 {{s4_n_target}} 人 × {{s4_coupon_cost}} 元`
  （全域发券成本 {{s4_cost_all_wan}} 万元 → 精准投放 {{s4_cost_target_wan}} 万元）
- **口径说明**：投放人群取 `mart_member_profile.discount_sensitivity` **实测标签**的 **Top {{s4_top_pct}}**（分位阈值 {{s4_threshold}}）。**用分位数而不是固定阈值** —— 该标签分布极度右偏（实测中位 0.026、最大 120.8），固定阈值 0.3 只会圈出 45 个人；Top 20% 的口径也与流失模型「运营只发得动 20% 的人」一致。
- **假设**：人均券成本 {{s4_coupon_cost}} 元（**假设值**）；投放范围取 Top {{s4_top_pct}}（**假设值**）
- **验证方式**：两组券的核销率与连带购买额对比

### 商品与采购策略

**S5 关联陈列与捆绑套餐（对应关联规则）**

- **结论**：按 Top {{s5_rules_used}} 条关联规则调整陈列，年化带动 GMV **{{s5_gmv_lo}} ~ {{s5_gmv_hi}} 万元**。
- **测算**：`ΔGMV年 = Σ_规则 (后件品类月GMV) × (confidence × 转化系数 {{s5_lift_lo}}~{{s5_lift_hi}}) × support × 12`
  其中转化系数表示「相邻陈列后，后件购买率能兑现关联强度的三到六成」
- **假设**：转化系数 {{s5_lift_lo}}~{{s5_lift_hi}}（**假设值**，需 A/B 验证）；support 作为可覆盖比例的代理
- **验证方式**：调整门店 vs 对照门店的后件品类销量差异，周期 4 周

**S6 品类结构优化（对应 F9）**

- **结论**：A 类 {{n_abc_a}} 个品类保货架占比 ≥ 50%；对 C 类品类按「90 天零动销 + 毛利贡献」双维度淘汰 SKU。
- **测算**：`释放资金 = C 类 SKU 库存占用 × 淘汰比例`，当前库存成本需甲方数据，本报告不臆造金额。
- **假设**：淘汰比例（**假设值**）
- **验证方式**：淘汰后 90 天品类 GMV 与毛利率对比

**S7 季节性采购与排班（对应 F3/F10）**

- **结论**：备货倍数按 **{{s7_ratio}} 倍** 执行（淡季基准 {{s7_trough_gmv_wan}} 万元 → 旺季 {{s7_peak_gmv_wan}} 万元），可把淡季滞销资金占用控制在 **{{s7_slow_lo}} ~ {{s7_slow_hi}} 万元**，同时把旺季缺货损失控制在 **{{s7_loss_lo}} ~ {{s7_loss_hi}} 万元**。
- **测算**：
  - `滞销资金 = 淡季基准GMV {{s7_trough_gmv_wan}} 万 × 备货倍数 {{s7_ratio}} × 滞销率 {{s7_slow_rate_lo}}~{{s7_slow_rate_hi}}`
  - `缺货损失 = 旺季GMV {{s7_peak_gmv_wan}} 万 × 缺货率 {{s7_short_rate_lo}}~{{s7_short_rate_hi}}`
- **假设**：滞销率与缺货率均为**假设值**，需用甲方历史库存数据校准
- **验证方式**：旺季实际缺货率与淡季实际滞销率回填校准
- **口径说明**：淡季基准取**最低 3 个月的均值**而非绝对最小月 —— 数据首月往往不完整，用绝对最小值会把备货倍数放大到不合理量级

**S8 门店差异化备货**

- **结论**：{{store_count}} 家门店的品类结构差异度平均 {{store_diff_mean}}、最高 {{store_diff_max}}；按差异化调整品类深度，年化 GMV **{{s8_gmv_lo}} ~ {{s8_gmv_hi}} 万元**。
- **测算**：`ΔGMV = Σ_门店 (门店GMV {{store_gmv_total_wan}} 万元 × 品类结构差异度 × 优化系数 {{s8_coef_lo}}~{{s8_coef_hi}})`
  差异度定义：`1 − cos(门店品类占比向量, 全司品类占比向量)`
- **假设**：优化系数 {{s8_coef_lo}}~{{s8_coef_hi}}（**假设值**）
- **验证方式**：按门店分组对比调整前后的品类动销率
- **数据口径与局限自曝**：v3.5 起生成器为门店引入「城市档位（一线/二线/三线）× 品类」乘子矩阵，门店品类结构**不再由同一套权重派生** —— 本报告实测差异度平均 {{store_diff_mean}}、最高 {{store_diff_max}}，**落在真实商超门店间 0.05~0.30 的下沿**。因此 S8 的**方向与量级可被合成数据验证**（当前偏保守，属"低估"而非"无法验证"）；但门店档位与真实商圈画像仍有偏差，**绝对值必须以甲方真实门店-品类数据重算**。（v3.5 之前 15 家门店共用一套品类权重，差异度实测 <0.001，S8 确实无法验证 —— 该缺陷已修复。）

---

## 三、落地路线图

| 阶段 | 动作 | 依赖 |
|---|---|---|
| 短期（1 个月） | S1 高危挽留、S5 关联陈列调整 | 名单已产出（sheet 5 可导出），阈值由 `metrics_churn.json` 回填 |
| 中期（1 季度） | S2 分层权益、S3 复购任务、S4 促销精准化、S6 SKU 淘汰首批 | 需营销系统对接画像标签 |
| 长期（半年） | S7 季节性采购体系、S8 门店差异化、画像系统接入营销自动化 | 需甲方库存与成本数据校准测算 |

## 四、模型与方法附注

- **流失模型**：后端 {{model_backend}}，**主模型 {{model_name}}**（AUC {{model_auc}}，较 LR 基线 {{model_auc_lr}} 提升 {{model_auc_gain}}，{{model_auc_gain_verdict}}）；标准 KS {{model_ks}}；Top20% 名单捕获率 {{model_capture}}（lift {{model_lift}}，基线流失率 {{model_base_rate}} 下理论上限 {{model_lift_upper}}）。特征 {{model_n_features}} 维。对照模型：GBDT AUC {{model_gbdt_auc}}（增益 {{model_gbdt_gain}}）、随机森林 AUC {{model_rf_auc}}（增益 {{model_rf_gain}}）。选型依据：{{model_selection_reason}}。
- **时间切分**：270 天观察窗 + 90 天标签窗，5 期 cohort，训练 {{model_cohorts}} / 验证 {{model_valid_cohort}} / 测试 {{model_test_cohort}}，严格按时间先后（`tests/test_leakage.py` 有专项断言）。
- **阈值调优**：在验证集上按业务成本最优（漏判 : 误触 = 8 : 1，与 S2 同口径）求切点，行动名单阈值 {{threshold_action}}、高危阈值 {{threshold_high}}。
- **聚类**：K-Means，特征空间 {{cluster_space}}，K={{cluster_k}}，轮廓系数 {{cluster_silhouette}}，与先验人群 ARI {{cluster_ari}}；神谕上界 {{cluster_oracle}}（说明 8 类行为型人群在消费体量特征上本质不可分）。
- **同比/环比口径**：环比 = 本期 / 上一期 − 1；同比 = 本期 / 上年同期 − 1；数据为 {{n_months}} 个月非完整年度，**{{q_last_incomplete}} 仅含 1 个月，不参与季度同比解读**。
- **关联规则**：品类级 FP-Growth，min_support 1% / min_confidence 30% / lift > 1.5，共 {{n_rules}} 条。
- **全部测算为区间估算 + 敏感性声明**，标注「假设」者为行业经验值，不作为承诺值；上线后需以 A/B 实验回收真实参数。
