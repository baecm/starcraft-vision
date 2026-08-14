COMPOSE_FILE=infra/docker-compose.yml
PID_DIR=pids
JUPYTER_PORT ?= 8888
DEBUG_PORT   ?= 5678
ENABLE_SYNOLOGY_CHAT ?= false
export ENABLE_SYNOLOGY_CHAT

PYDEBUG_SERVICE  ?= pydebug

.PHONY: pydebug-up pydebug-logs stop-pydebug \
        debugger tunnel-debug stop-tunnel-debug \
		zeppelin

# 디렉토리 생성
$(shell mkdir -p $(PID_DIR))
$(shell mkdir -p logs)

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

inference:
	CONTAINER_NAME=inference_$(shell date +%Y%m%d_%H%M%S); \
	nohup docker compose -f $(COMPOSE_FILE) run --name $$CONTAINER_NAME --rm inferencer inference $(ARGS) \
		> logs/$$CONTAINER_NAME.log 2>&1 & \
	echo $$CONTAINER_NAME > $(PID_DIR)/inference.cid

evaluate:
	CONTAINER_NAME=evaluate_$(shell date +%Y%m%d_%H%M%S); \
	nohup docker compose -f $(COMPOSE_FILE) run --name $$CONTAINER_NAME --rm evaluator evaluate $(ARGS) \
		> logs/$$CONTAINER_NAME.log 2>&1 & \
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

zeppelin:
	docker compose -f $(COMPOSE_FILE) up zeppelin