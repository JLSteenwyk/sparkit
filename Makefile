VENV ?= ./venv
PY ?= $(VENV)/bin/python
PIP ?= $(VENV)/bin/pip
ALEMBIC ?= $(VENV)/bin/alembic

.PHONY: venv install db-upgrade db-downgrade db-current db-history test-compile test eval-sample eval-benchmark eval-benchmark-full benchmark-generate benchmark-generate-hle-biochem-20 benchmark-repeated-slices baseline-capture baseline-capture-official baseline-capture-quick baseline-capture-hle-biochem-20 baseline-capture-direct-calls-hle20 drift-check-sample drift-check-manifest

venv:
	python3 -m venv $(VENV)
	$(PIP) install --upgrade pip setuptools wheel

install:
	$(PIP) install -e .

db-upgrade:
	$(ALEMBIC) upgrade head

db-downgrade:
	$(ALEMBIC) downgrade -1

db-current:
	$(ALEMBIC) current

db-history:
	$(ALEMBIC) history --verbose

test-compile:
	$(PY) -m compileall services shared alembic

test:
	$(PY) -m pytest -q

eval-sample:
	$(PY) scripts_run_eval.py --questions benchmarks/stem_exam_200_sample/questions.json --predictions benchmarks/stem_exam_200_sample/predictions_sample.json

eval-benchmark:
	$(PY) scripts_run_eval.py --questions benchmarks/stem_exam_200_sample/questions.json --mode routed --providers openai,anthropic,gemini

eval-benchmark-full:
	$(PY) scripts_run_eval.py --questions benchmarks/stem_exam_200/questions.json --mode routed --providers openai,anthropic,gemini

benchmark-generate:
	$(PY) scripts_generate_stem_exam_200.py

benchmark-generate-hle-biochem-20:
	$(PY) scripts_generate_hle_biochem_subset.py --output benchmarks/hle_gold_bio_chem/questions_bio10_chem10.json --bio-count 10 --chem-count 10 --seed 7

benchmark-repeated-slices:
	$(PY) scripts_benchmark_repeated_slices.py --questions benchmarks/hle_gold_bio_chem/questions_bio10_chem10.json --configs single_openai,single_anthropic,routed_frontier --slice-size 10 --repeats 5 --parallel-workers 2 --output benchmarks/results/repeated_slices_report.json

baseline-capture:
	$(PY) scripts_capture_baselines.py --questions benchmarks/stem_exam_200/questions.json --skip-missing-keys

baseline-capture-official:
	$(PY) scripts_capture_baselines.py --questions benchmarks/stem_exam_200/questions.json --label official

baseline-capture-quick:
	$(PY) scripts_capture_baselines.py --questions benchmarks/stem_exam_200/questions.json --label quick --max-questions 1 --configs single_openai,routed_frontier --min-sources 1 --max-latency-s 45 --max-cost-usd 0.8

baseline-capture-hle-biochem-20:
	$(PY) scripts_capture_baselines.py --questions benchmarks/hle_gold_bio_chem/questions_bio10_chem10.json --label hle_bio_chem_20 --configs single_openai,single_anthropic,single_gemini,single_kimi,routed_frontier --min-sources 1 --max-latency-s 90 --max-cost-usd 2.0 --skip-missing-keys

baseline-capture-direct-calls-hle20:
	$(PY) scripts_capture_direct_baselines.py --questions benchmarks/hle_gold_bio_chem/questions_bio10_chem10.json --label hle_bio_chem_20_direct --providers openai,anthropic,gemini,kimi --skip-missing-keys

drift-check-sample:
	$(PY) scripts_run_eval.py --questions benchmarks/stem_exam_200_sample/questions.json --predictions benchmarks/stem_exam_200_sample/predictions_sample.json --save-report /tmp/sparkit_sample_report.json
	$(PY) scripts_drift_check.py --report /tmp/sparkit_sample_report.json --thresholds benchmarks/drift_thresholds.json --output /tmp/sparkit_drift_sample.json

drift-check-manifest:
	$(PY) scripts_drift_check.py --candidate-manifest $(CANDIDATE_MANIFEST) --baseline-manifest $(BASELINE_MANIFEST) --thresholds benchmarks/drift_thresholds.json
