PY ?= python
SCALE ?= 0.3
PORT ?= 8501

.PHONY: install gen etl rfm rules rules_sku rules_store churn profile cluster \
        warehouse reports report cards segment_compare dashboard all test cov lint format clean

# 主依赖已包含全链路必需的 duckdb / plotly / shap（v3.4 修复漏声明）；
# lightgbm 仍是可选：不装会自动退 sklearn HistGB 并在 metrics_churn.json 留痕。
install:
	$(PY) -m pip install -e ".[dev]"

gen:
	$(PY) -m retailinsight.cli run --stage gen --scale $(SCALE)

etl:
	$(PY) -m retailinsight.cli run --stage etl

rfm:
	$(PY) -m retailinsight.cli run --stage rfm

rules:
	$(PY) -m retailinsight.cli run --stage rules

rules_sku:
	$(PY) -m retailinsight.cli run --stage rules_sku

rules_store:
	$(PY) -m retailinsight.cli run --stage rules_store

churn:
	$(PY) -m retailinsight.cli run --stage churn

profile:
	$(PY) -m retailinsight.cli run --stage profile

cluster:
	$(PY) -m retailinsight.cli run --stage cluster

warehouse:
	$(PY) -m retailinsight.cli run --stage warehouse

reports:
	$(PY) -m retailinsight.cli run --stage reports

report:
	$(PY) -m retailinsight.cli run --stage report

cards:
	$(PY) -m retailinsight.cli run --stage cards

segment_compare:
	$(PY) -m retailinsight.cli run --stage segment_compare

# 交互式看板（会阻塞终端，Ctrl+C 退出）；不纳入 all
dashboard:
	$(PY) -m retailinsight.cli run --stage dashboard --port $(PORT)

all:
	$(PY) -m retailinsight.cli run --stage all --scale $(SCALE)

# 覆盖率门禁 ≥60% 只作用于**全量**运行（单文件跑测试不该被 0% 覆盖率判红）
test:
	$(PY) -m pytest --cov-fail-under=60

cov:
	$(PY) -m pytest --cov=retailinsight --cov-report=term-missing --cov-fail-under=60

# lint 是**门禁**：ruff check 严格规则集必须全绿（当前状态：全绿）。
# ruff format 是**建议项**：仓库尚未统一格式化（基线 50/56 文件会被重排），
# 因此不放在 lint 里，避免门禁长期红灯；需要时用 `make format` 单独执行。
lint:
	$(PY) -m ruff check .

format:
	$(PY) -m ruff format .

clean:
	rm -rf .pytest_cache .ruff_cache .coverage htmlcov
	find . -type d -name __pycache__ -prune -exec rm -rf {} +
