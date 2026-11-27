# Reproduce the results of the article.
#   From the stored raw results:  make install test analysis
#   From scratch:                 make install test validate tune experiments diagnostics theory analysis
PY      ?= python3
WORKERS ?= 2

.PHONY: install test smoke configs validate tune experiments diagnostics theory analysis all clean

install:
	$(PY) -m pip install -r requirements.txt

test:                           ## unit tests
	$(PY) -m pytest -q tests

smoke:                          ## one short simulation per method
	$(PY) experiments/smoke.py

configs:                        ## JSON copy of the configuration in experiments/configs/
	$(PY) experiments/export_configs.py

validate:                       ## simulator sanity checks on hand-built topologies
	$(PY) experiments/validate_mac.py

tune:                           ## equal-budget random search on the validation seeds
	$(PY) experiments/tune.py --workers $(WORKERS)

experiments:                    ## test seeds, frozen configurations (E1-E8)
	$(PY) experiments/run_experiments.py E1 E5 E4 E3 E8 E6 E7 E2 --workers $(WORKERS)

diagnostics:                    ## instrumented re-runs on test seeds 101-103 (timings depend on the machine)
	cd analysis && $(PY) route_timing.py && $(PY) mechanism.py --rerun && $(PY) lifetime_deaths.py

theory:                         ## numerical checks of Lemma 1, Theorem 1 and Proposition 3
	cd analysis && $(PY) theory_checks.py && $(PY) fig_theory.py

analysis:                       ## tables, figures and results/test2/summary_v2.json from the raw results
	cd analysis && $(PY) v2.py && $(PY) scorecard.py && $(PY) mechanism.py

all: test validate tune experiments diagnostics theory analysis

clean:                          ## remove generated tables and figures
	rm -rf output
