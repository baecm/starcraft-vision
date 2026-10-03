COMPOSE_FILE=infra/docker-compose.yml
PID_DIR=pids

# The container mounts only src/, so it has no .git and cannot answer "which
# commit is this". Read the host checkout here and pass it in, so every run
# records the code that produced it. See _write_run_provenance in src/train.py.
GIT_COMMIT := $(shell git rev-parse HEAD 2>/dev/null)
GIT_DIRTY  := $(shell test -n "$$(git status --porcelain 2>/dev/null)" && echo 1 || echo 0)
export GIT_COMMIT
export GIT_DIRTY

# Which machine a run was launched on. The container cannot work this out for
# itself - its hostname is the container id - so it is read here alongside the
# commit. NVIDIA_VISIBLE_DEVICES already reaches the container via compose, and
# it has to: inside, the selected card is always cuda:0, so torch cannot say
# which physical GPU is in use either.
HOST_NAME := $(shell hostname -s 2>/dev/null || hostname 2>/dev/null)
export HOST_NAME
JUPYTER_PORT ?= 8888
DEBUG_PORT   ?= 5678
ENABLE_SYNOLOGY_CHAT ?= false
export ENABLE_SYNOLOGY_CHAT

ifeq ($(OS),Windows_NT)
    # Auto-mount network share into WSL2 docker-desktop host (direct UNC path or via drive Z:)
    $(shell wsl -d docker-desktop -e sh -c "if [ -d /mnt/host ] && ! mountpoint -q /mnt/host/z; then mkdir -p /mnt/host/z && (mount -t drvfs '\\\172.27.186.243\homes\baecm' /mnt/host/z 2>/dev/null || mount -t drvfs 'Z:' /mnt/host/z 2>/dev/null); fi" 2>/dev/null)
    # Use Git Bash on Windows (8.3 short path avoids space issues in Make)
    SHELL := C:/PROGRA~1/Git/bin/bash.exe
    .SHELLFLAGS := -c
    # Disable MSYS automatic path conversion so /workspace paths are preserved for Docker
    export MSYS_NO_PATHCONV := 1
    export MSYS2_ARG_CONV_EXCL := *
endif

# 컨테이너 실행 uid/gid는 infra/.env 의 UID/GID 에서 온다 (워크스테이션별, gitignore).
# 마운트된 디렉터리의 소유자와 반드시 일치해야 한다. CIFS/NAS 마운트라면
# `id -u` 가 아니라 그 마운트의 uid= 옵션 값이다.

PYDEBUG_SERVICE  ?= pydebug

.PHONY: pydebug-up pydebug-logs stop-pydebug \
        debugger tunnel-debug stop-tunnel-debug \
		zeppelin benchmark mode-disagreement mode-disagreement-fg inference-fg train-fg test probe probe-resolution \
		budget-allocation qualitative-figures-fg figure-fg run-registry

# 디렉토리 생성
ifneq ($(OS),Windows_NT)
    $(shell mkdir -p $(PID_DIR) logs results models predictions .torch_cache)
endif

define run_or_parallel
	@REPLAY_COUNT=$(shell echo $(ARGS) | sed -n 's/.*--replays\([^"]*\).*/\1/p' | wc -w); \
	CONTAINER_NAME=$(1)_$$(date +%Y%m%d_%H%M%S); \
	if [ "$$REPLAY_COUNT" -le 1 ]; then \
		echo "[Makefile] Running $(1) as single task"; \
		nohup docker compose -f infra/docker-compose.yml run --name $$CONTAINER_NAME --rm preprocessor $(1) $(ARGS) \
			> logs/$$CONTAINER_NAME.log 2>&1 & \
		echo $$CONTAINER_NAME > $(PID_DIR)/$(1).cid; \
	else \
		echo "[Makefile] Running $(1) in parallel"; \
		nohup bash -c '\
			COMMAND="$$1"; shift; \
			REPLAY_IDS=(); ARGS=(); \
			while [[ $$# -gt 0 ]]; do \
				if [[ "$$1" == "--replays" ]]; then \
					shift; \
					while [[ $$# -gt 0 && "$$1" != --* ]]; do REPLAY_IDS+=("$$1"); shift; done; \
				else \
					ARGS+=("$$1"); shift; \
				fi; \
			done; \
			MAX_PARALLEL=2; CURRENT_PARALLEL=0; \
			for REPLAY_ID in "$${REPLAY_IDS[@]}"; do \
				LOG_FILE="logs/$${COMMAND}_$${REPLAY_ID}.log"; \
				docker compose -f infra/docker-compose.yml run --rm preprocessor "$$COMMAND" --replays "$$REPLAY_ID" "$${ARGS[@]}" > "$$LOG_FILE" 2>&1 & \
				CURRENT_PARALLEL=$$((CURRENT_PARALLEL + 1)); \
				if [ "$$CURRENT_PARALLEL" -ge "$$MAX_PARALLEL" ]; then wait -n; CURRENT_PARALLEL=$$((CURRENT_PARALLEL - 1)); fi; \
			done; \
			wait; \
		' dummy $(1) $(ARGS) > logs/$$CONTAINER_NAME.log 2>&1 & \
		echo $$CONTAINER_NAME > $(PID_DIR)/$(1).cid; \
	fi
endef

build:
	docker compose -f $(COMPOSE_FILE) build

rebuild:
	docker compose -f $(COMPOSE_FILE) build --no-cache

train:
	CONTAINER_NAME=train_$(shell date +%Y%m%d_%H%M%S); \
	nohup docker compose -f $(COMPOSE_FILE) run --name $$CONTAINER_NAME --rm trainer train $(ARGS) \
		> logs/$$CONTAINER_NAME.log 2>&1 & \
	echo $$CONTAINER_NAME > $(PID_DIR)/train.cid

# Foreground training, so a shell can queue several runs onto one GPU with
# `&&`. `train` backgrounds itself, so a chain built on it starts every run at
# once and they fight over the card. Same reasoning as inference-fg.
train-fg:
	docker compose -f $(COMPOSE_FILE) run --rm trainer train $(ARGS)

# One training forward pass of Mask R-CNN + KBRS from the Hydra config, checking
# the weights the scorer received, that every component is non-zero, and that
# loss_kbrs reaches the backbone. Run it before launching KBRS training. ARGS
# are extra Hydra overrides, e.g. ARGS="plugins/kbrs/score=mixture/000".
kbrs-smoke:
	docker compose -f $(COMPOSE_FILE) run --rm debugger debug -c "PYTHONPATH=/workspace:/workspace/src python3 /workspace/scripts/kbrs_smoke.py $(ARGS)"

inference:
	CONTAINER_NAME=inference_$(shell date +%Y%m%d_%H%M%S); \
	nohup docker compose -f $(COMPOSE_FILE) run --name $$CONTAINER_NAME --rm inferencer inference $(ARGS) \
		> logs/$$CONTAINER_NAME.log 2>&1 & \
	echo $$CONTAINER_NAME > $(PID_DIR)/inference.cid

# Foreground inference, so a shell can wait for it and chain the evaluation
# that reads its predictions. Same reasoning as mode-disagreement-fg.
inference-fg:
	docker compose -f $(COMPOSE_FILE) run --rm inferencer inference $(ARGS)

evaluate:
	CONTAINER_NAME=evaluate_$$(date +%Y%m%d_%H%M%S); \
	docker compose -f $(COMPOSE_FILE) run --name $$CONTAINER_NAME --rm evaluator evaluate $(ARGS) \
		> logs/$$CONTAINER_NAME.log 2>&1; \
	echo $$CONTAINER_NAME > $(PID_DIR)/evaluate.cid

estimate:
	CONTAINER_NAME=estimate_$$(date +%Y%m%d_%H%M%S); \
	docker compose -f $(COMPOSE_FILE) run --name $$CONTAINER_NAME --rm evaluator estimate $(ARGS) \
		> logs/$$CONTAINER_NAME.log 2>&1; \
	echo $$CONTAINER_NAME > $(PID_DIR)/estimate.cid

cache:
	CONTAINER_NAME=cache_$(shell date +%Y%m%d_%H%M%S); \
	nohup docker compose -f $(COMPOSE_FILE) run --name $$CONTAINER_NAME --rm evaluator cache $(ARGS) \
		> logs/$$CONTAINER_NAME.log 2>&1 & \
	echo $$CONTAINER_NAME > $(PID_DIR)/cache.cid

lookup:
	CONTAINER_NAME=lookup_$(shell date +%Y%m%d_%H%M%S); \
	nohup docker compose -f $(COMPOSE_FILE) run --name $$CONTAINER_NAME --rm evaluator lookup $(ARGS) \
		> logs/$$CONTAINER_NAME.log 2>&1 & \
	echo $$CONTAINER_NAME > $(PID_DIR)/lookup.cid

precheck:
	CONTAINER_NAME=precheck_$(shell date +%Y%m%d_%H%M%S); \
	nohup docker compose -f $(COMPOSE_FILE) run --name $$CONTAINER_NAME --rm preprocessor precheck $(ARGS) \
		> logs/$$CONTAINER_NAME.log 2>&1 & \
	echo $$CONTAINER_NAME > $(PID_DIR)/precheck.cid

profile_kbrs:
	CONTAINER_NAME=profile_kbrs_$(shell date +%Y%m%d_%H%M%S); \
	nohup docker compose -f $(COMPOSE_FILE) run --name $$CONTAINER_NAME --rm evaluator profile_kbrs $(ARGS) \
		> logs/$$CONTAINER_NAME.log 2>&1 & \
	echo $$CONTAINER_NAME > $(PID_DIR)/profile_kbrs.cid

run:
	CONTAINER_NAME=run_$(shell date +%Y%m%d_%H%M%S); \
	nohup docker compose -f $(COMPOSE_FILE) run --name $$CONTAINER_NAME --rm trainer run $(ARGS) \
		> logs/$$CONTAINER_NAME.log 2>&1 & \
	echo $$CONTAINER_NAME > $(PID_DIR)/run.cid

debug:
	CONTAINER_NAME=debug_$(shell date +%Y%m%d_%H%M%S); \
	docker compose -f $(COMPOSE_FILE) run --name $$CONTAINER_NAME --rm debugger debug $(ARGS);

test:
	docker compose -f $(COMPOSE_FILE) run --rm debugger debug -c "PYTHONPATH=/workspace:/workspace/src python3 /workspace/tests/test_director_losses.py"

probe:
	docker compose -f $(COMPOSE_FILE) run --rm debugger debug -c "PYTHONPATH=/workspace:/workspace/src python3 /workspace/scripts/probe_director.py"


probe-resolution:
	docker compose -f $(COMPOSE_FILE) run --rm debugger debug -c "PYTHONPATH=/workspace:/workspace/src python3 /workspace/scripts/probe_resolution.py"

benchmark:
	CONTAINER_NAME=benchmark_$$(date +%Y%m%d_%H%M%S); \
	nohup docker compose -f $(COMPOSE_FILE) run --name $$CONTAINER_NAME --rm debugger debug -c "PYTHONPATH=/workspace:/workspace/src python3 /workspace/scripts/run_benchmark.py $(ARGS)" \
		> logs/$$CONTAINER_NAME.log 2>&1 & \
	echo $$CONTAINER_NAME > $(PID_DIR)/benchmark.cid

# Example:
# make mode-disagreement ARGS="--replays 275 1725 3613 4520 4664 --label-method all_correct \
#   --model maskrcnn=maskrcnn_win4_vanilla_fold1_s123_20251201_072032 \
#   --model maskrcnn_kbrs=maskrcnn_win4_kbrs_fold1_s123_kbrs025_base_score_20260203_055642 \
#   --model centernet=centernet_vanilla_win1_fold1_s123_20260818_071116 \
#   --epoch 30 --outdir /workspace/results/mode_disagreement/fold1"
mode-disagreement:
	CONTAINER_NAME=mode_disagreement_$$(date +%Y%m%d_%H%M%S); \
	nohup docker compose -f $(COMPOSE_FILE) run --name $$CONTAINER_NAME --rm debugger debug -c "PYTHONPATH=/workspace:/workspace/src python3 /workspace/scripts/mode_disagreement.py $(ARGS)" \
		> logs/$$CONTAINER_NAME.log 2>&1 & \
	echo $$CONTAINER_NAME > $(PID_DIR)/mode_disagreement.cid

# Same as mode-disagreement but in the foreground, so the shell waits for it.
# The backgrounded target returns as soon as the container is launched, which
# makes "a && b" start both at once - they then compete for the GPU and the
# second overwrites the first CID file. Chain this one instead, and wrap the
# whole chain in a single nohup if the terminal should be free:
#   nohup bash -c 'make mode-disagreement-fg ARGS="..." && \
#                   make mode-disagreement-fg ARGS="..."' > logs/chain.log 2>&1 &
mode-disagreement-fg:
	docker compose -f $(COMPOSE_FILE) run --rm debugger debug -c "PYTHONPATH=/workspace:/workspace/src python3 /workspace/scripts/mode_disagreement.py $(ARGS)"

# Reads the frames_*.csv that mode-disagreement wrote. Example:
# make budget-allocation ARGS="--dir /workspace/results/mode_disagreement/th_sweep \
#   --methods director prior_th07 --k-cap director=3"
budget-allocation:
	docker compose -f $(COMPOSE_FILE) run --rm debugger debug -c "PYTHONPATH=/workspace:/workspace/src python3 /workspace/scripts/budget_allocation.py $(ARGS)"

# Are the primary region's cuts prepared by an auxiliary region? See the
# script's docstring. Reads predictions only, so it needs no GPU. Example:
# make switch-preparation ARGS="--replays 275 1725 3613 4520 4664 \
#   --model director=dc_full_b16_f1_s456_v6 --epoch 30"
switch-preparation:
	docker compose -f $(COMPOSE_FILE) run --rm debugger debug -c "PYTHONPATH=/workspace:/workspace/src python3 /workspace/scripts/switch_preparation.py $(ARGS)"

# Re-derive single-region numbers reported from earlier evaluator versions and
# separate the fixes behind them (missing-prediction frames, edge-clamped
# centroids, the sentinel scored as a corner viewport). Reads predictions only,
# so it needs no GPU. Output goes to results/reeval/<set>_<date>/; see the
# script's docstring. --set kbrs is the saliency-prior paper, --set ch9 the
# Director-CenterNet comparison. Check one run first:
# make reeval-missing-frames ARGS="--set kbrs --runs maskrcnn_win4_vanilla_fold1_s123_20251201_072032 --out /workspace/results/reeval/check"
reeval-missing-frames:
	CONTAINER_NAME=reeval_$$(date +%Y%m%d_%H%M%S); \
	nohup docker compose -f $(COMPOSE_FILE) run --name $$CONTAINER_NAME --rm debugger debug -c "PYTHONPATH=/workspace:/workspace/src python3 /workspace/scripts/reeval_missing_frames.py $(ARGS)" \
		> logs/$$CONTAINER_NAME.log 2>&1 & \
	echo $$CONTAINER_NAME > $(PID_DIR)/reeval.cid; \
	echo "log: logs/$$CONTAINER_NAME.log"

# The paper's qualitative figures (1)-(3); see the script's docstring. The
# heatmap subcommand re-runs a checkpoint, so it wants the GPU. Example:
# make qualitative-figures-fg ARGS="select #   --baseline-csv /workspace/results/mode_disagreement/fold1/frames_maskrcnn.csv #   --director-csv /workspace/results/mode_disagreement/fold1/frames_director.csv"
qualitative-figures-fg:
	docker compose -f $(COMPOSE_FILE) run --rm debugger debug -c "PYTHONPATH=/workspace:/workspace/src python3 /workspace/scripts/qualitative_figures.py $(ARGS)"

# One figure, by the name of its script in scripts/figures/ (the figures of
# qualitative-figures-fg, one script each; see the script's docstring).
# scripts/figures/render_all.sh lists the commands behind every published
# figure. Example:
# make figure-fg FIG=single_failure ARGS="--replay 4664 --frame 10331 \
#   --tie-replay 1725 --start 11970 --end 12097 --pad 10 \
#   --baseline maskrcnn=maskrcnn_win4_vanilla_f1_s456_v6:30@0.5 --label Mask_R-CNN"
figure-fg:
	@test -n "$(FIG)" || { echo "usage: make figure-fg FIG=<script in scripts/figures> ARGS=\"...\""; exit 1; }
	docker compose -f $(COMPOSE_FILE) run --rm debugger debug -c "PYTHONPATH=/workspace:/workspace/src python3 /workspace/scripts/figures/$(FIG).py $(ARGS)"

notebook:
	CONTAINER_NAME=notebook_$$(date +%Y%m%d_%H%M%S); \
	docker compose -f $(COMPOSE_FILE) run --name $$CONTAINER_NAME -p 8888:8888 --rm notebook debug -c "jupyter lab --ip=0.0.0.0 --port=8888 --no-browser --allow-root --NotebookApp.token='' --NotebookApp.password=''"

# Entry points
preprocess_input:
	$(call run_or_parallel,preprocess_input)
preprocess_label:
	docker compose -f $(COMPOSE_FILE) run --rm preprocessor preprocess_label $(ARGS)

stop:
	@echo "Stopping all tracked containers..."
	@for cidfile in $(PID_DIR)/*.cid; do \
		[ -f $$cidfile ] && docker stop $$(cat $$cidfile) 2>/dev/null || true; \
		rm -f $$cidfile; \
	done

# Which trained runs are still comparable to the current code, read from the
# run_provenance.json that train.py writes next to each checkpoint. SINCE names
# the commit after which a run counts as current; the ancestry is resolved here
# because the containers do not mount .git. Examples:
#   make run-registry
#   make run-registry SINCE=ed86508 ARGS="--out /workspace/results/run_registry.md"
SINCE ?=
run-registry:
	@EXTRA=""; if [ -n "$(SINCE)" ]; then EXTRA="--ok-commits $$(git rev-list --abbrev-commit --abbrev=12 $(SINCE)..HEAD | tr '\n' ' ') $$(git rev-parse --short=12 HEAD)"; fi; docker compose -f $(COMPOSE_FILE) run --rm debugger debug -c "PYTHONPATH=/workspace:/workspace/src python3 /workspace/scripts/run_registry.py --models-root /workspace/models --pred-root /workspace/predictions $$EXTRA $(ARGS)"
