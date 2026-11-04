# Commands and the parameters they accept (pass as VAR=value):
#   make download                          fetch all models into the local HF cache
#   make eval       [MODEL=] [LIMIT=] [FRESH=1]
#                                          full run: 3 models x 200 images, verify, results/eval.html
#   make fast-eval  [MODEL=] [DOGS=] [FRESH=1]
#                                          fast run: all 100 non-dogs + DOGS sampled dogs
#                                          (default 10), verify, results/fast/eval.html
#   make lab        NAME= [SPLIT=] [MODEL=]
#                                          self-test loop: real pipeline on the dev (even) or
#                                          test (odd) images, cached outputs, results/lab/
#   make control    [MODEL=]               no-notes control: the recorded Second Pass again
#                                          without retrieved notes, one model at a time
#   make web | serve [HOST=] [PORT=]       web app / API only
#
# Parameters:
#   MODEL=lfm3-vl-2b|lfm2-vl-450m|qwen3-vl-2b   one model only (default: all three)
#   LIMIT=N    only the first N images (smoke test)
#   DOGS=N     dogs to sample in fast-eval (default 10)
#   NAME=x     name of a lab run; SPLIT=dev|test|all (default dev)
#   FRESH=1    delete earlier paired results of that run first
#   PYTHON=/path/to/python   interpreter if the venv lives elsewhere
#
# Examples:
#   make fast-eval MODEL=lfm3-vl-2b        ~110 images, a few minutes
#   make eval FRESH=1 LIMIT=5

PYTHON ?= .venv/bin/python
PNPM ?= pnpm
FRONTEND_DIR := frontend
# HOST and PORT default to server.host / server.port in configs/default.yaml
HOST ?=
PORT ?=

.DEFAULT_GOAL := download

.PHONY: download eval fast-eval lab control frontend-build serve web

# 1) Download the three Vision Models + the embedding model into the local HF cache.
#    This is the only command that uses the network; inference and evaluation read
#    the local cache only and fail with an error when a model is missing.
download:
	$(PYTHON) scripts/download-models.py

# 2) 3 models x 200 images x paired First Pass / RAG Second Pass -> verify -> report (one script).
#    Rerunning resumes (failed images run again); FRESH=1 starts over.
eval:
	$(PYTHON) scripts/evaluate-models.py $(if $(FRESH),--fresh,) $(if $(LIMIT),--limit $(LIMIT),) $(if $(MODEL),--model $(MODEL),)

# Fast check for non-dog detection: every model, all 100 non-dogs + DOGS sampled dogs,
# then verify and write results/fast/eval.html. The full results stay untouched.
DOGS ?= 10
fast-eval:
	$(PYTHON) scripts/evaluate-models.py --dogs $(DOGS) $(if $(FRESH),--fresh,) $(if $(MODEL),--model $(MODEL),)

# Try a prompt, knowledge or retrieval change: tune on SPLIT=dev, confirm on SPLIT=test.
SPLIT ?= dev
lab:
	$(PYTHON) scripts/rag-lab.py $(NAME) --split $(SPLIT) $(if $(MODEL),--model $(MODEL),)

# Does the knowledge itself help? Reruns the Second Pass of every image make eval scored,
# without the retrieved notes, and compares it with the recorded RAG Second Pass.
control:
	$(PYTHON) scripts/control-no-notes.py $(if $(MODEL),--model $(MODEL),)

# Build the React production bundle, then serve it and the API from FastAPI.
frontend-build:
	cd $(FRONTEND_DIR) && $(PNPM) install --frozen-lockfile && $(PNPM) run build

serve:
	$(PYTHON) -m app.cli serve $(if $(HOST),--host $(HOST),) $(if $(PORT),--port $(PORT),)

web: frontend-build
	$(MAKE) serve
