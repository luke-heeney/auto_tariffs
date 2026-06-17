PYTHON ?= python

.PHONY: downstream replication paper paper-fast paper-strict post-est cost-side cost-side-report check-downstream

downstream replication:
	PYTHON=$(PYTHON) bash make.sh

paper:
	PYTHON=$(PYTHON) bash paper/make.sh

paper-fast:
	PYTHON=$(PYTHON) bash paper/make.sh --skip-render

paper-strict:
	PYTHON=$(PYTHON) bash paper/make.sh --strict-canonical-pdf

post-est:
	PYTHON=$(PYTHON) bash post_est/make.sh

cost-side:
	PYTHON=$(PYTHON) bash cost_side/make.sh

cost-side-report:
	bash cost_side/build_robustness_report.sh

check-downstream:
	$(PYTHON) post_est/check_downstream_consistency.py
