# Commands and the parameters they accept (pass as VAR=value):
#   make download                          fetch all models into the local HF cache
#   make eval       [MODEL=] [LIMIT=] [FRESH=1]
#                                          full run: 3 models x 200 images, verify, results/eval.html
#   make fast-eval  [MODEL=] [DOGS=] [FRESH=1]
#                                          fast run: all 100 non-dogs + DOGS sampled dogs
#                                          (default 10), verify, results/fast/eval.html
#   make ai-log     [LOG_DIR=] [LABEL=]    write logs/<time>-<LABEL>.md from results on disk
#   make web | serve [HOST=] [PORT=]       web app / API only
#
# Parameters:
#   MODEL=lfm3-vl-2b|lfm2-vl-450m|qwen3-vl-2b   one model only (default: all three)
#   LABEL=r2   name of the AI log written to logs/ after eval and fast-eval (default: run)
#   LIMIT=N    only the first N images (smoke test)
#   DOGS=N     dogs to sample in fast-eval (default 10)
#   FRESH=1    delete earlier paired results of that run first
#   PYTHON=/path/to/python   interpreter if the venv lives elsewhere
#
# Examples:
#   make fast-eval MODEL=lfm3-vl-2b        ~110 images, a few minutes
#   make eval FRESH=1 LIMIT=5

SHELL := /bin/bash
PYTHON ?= .venv/bin/python
PNPM ?= pnpm
FRONTEND_DIR := frontend
# HOST and PORT default to server.host / server.port in configs/default.yaml
HOST ?=
PORT ?=

.DEFAULT_GOAL := download

# Run a command, keep its console output, then write logs/<time>-<LABEL>.md for an AI
# reviewer even when the run fails. After Ctrl-C use: make ai-log. Usage: $(call LOGGED,<results dir>,<command>)
LABEL ?= run
define LOGGED
mkdir -p logs; set +e; $(2) 2>&1 | tee logs/.console.txt; status=$${PIPESTATUS[0]}; \
$(PYTHON) scripts/ai_log.py --dir $(1) --label $(LABEL) --console logs/.console.txt --exit-code $$status; \
exit $$status
endef

.PHONY: download eval fast-eval ai-log frontend-build serve web

# 1) Download the three Vision Models + the embedding model into the local HF cache.
#    This is the only command that uses the network; inference and evaluation read
#    the local cache only and fail with an error when a model is missing.
download:
	$(PYTHON) scripts/download-models.py

# 2) 3 models x 200 images x paired First Pass / RAG Second Pass -> verify -> report (one script).
#    Rerunning resumes (failed images run again); FRESH=1 starts over.
eval:
	$(call LOGGED,results,$(PYTHON) scripts/evaluate-models.py $(if $(FRESH),--fresh,) $(if $(LIMIT),--limit $(LIMIT),) $(if $(MODEL),--model $(MODEL),))

# Fast check for non-dog detection: every model, all 100 non-dogs + DOGS sampled dogs,
# then verify and write results/fast/eval.html. The full results stay untouched.
DOGS ?= 10
fast-eval:
	$(call LOGGED,results/fast,$(PYTHON) scripts/evaluate-models.py --dogs $(DOGS) $(if $(FRESH),--fresh,) $(if $(MODEL),--model $(MODEL),))

# Write the AI log from results already on disk (e.g. after an interrupted run).
#   make ai-log LOG_DIR=results/fast LABEL=r2
LOG_DIR ?= results/fast
ai-log:
	$(PYTHON) scripts/ai_log.py --dir $(LOG_DIR) --label $(LABEL)

# Build the React production bundle, then serve it and the API from FastAPI.
frontend-build:
	cd $(FRONTEND_DIR) && $(PNPM) install --frozen-lockfile && $(PNPM) run build

serve:
	$(PYTHON) -m app.cli serve $(if $(HOST),--host $(HOST),) $(if $(PORT),--port $(PORT),)

web: frontend-build
	$(MAKE) serve
