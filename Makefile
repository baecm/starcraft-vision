COMPOSE_FILE=infra/docker-compose.yml
PID_DIR=pids
JUPYTER_PORT ?= 8888
DEBUG_PORT   ?= 5678

PYDEBUG_SERVICE  ?= pydebug


.PHONY: pydebug-up pydebug-logs stop-pydebug \
        debugger tunnel-debug stop-tunnel-debug

# 디렉토리 생성
$(shell mkdir -p $(PID_DIR))
$(shell mkdir -p logs)

define run_or_parallel
	@REPLAY_COUNT=$(shell echo $(ARGS) | sed -n 's/.*--replays\([^"]*\).*/\1/p' | wc -w); \
	CONTAINER_NAME=$(1)_$$(date +%Y%m%d_%H%M%S); \
	if [ "$$REPLAY_COUNT" -le 1 ]; then \
		echo "[Makefile] Running $(1) as single task"; \
		nohup docker compose -f infra/docker-compose.yml run --name $$CONTAINER_NAME --rm dispatcher $(1) $(ARGS) \
			> logs/$$CONTAINER_NAME.log 2>&1 & \
		echo $$CONTAINER_NAME > $(PID_DIR)/$(1).cid; \
	else \
		echo "[Makefile] Running $(1) in parallel"; \
		nohup bash scripts/preprocess_batch.sh $(1) $(ARGS) \
			> logs/$$CONTAINER_NAME.log 2>&1 & \
		echo $$CONTAINER_NAME > $(PID_DIR)/$(1).cid; \
	fi
endef

build:
	docker compose -f $(COMPOSE_FILE) build

rebuild:
	docker compose -f $(COMPOSE_FILE) build --no-cache

up:
	docker compose -f $(COMPOSE_FILE) up -d

down:
	docker compose -f $(COMPOSE_FILE) down --remove-orphans

train:
	CONTAINER_NAME=train_$(shell date +%Y%m%d_%H%M%S); \
	nohup docker compose -f $(COMPOSE_FILE) run --name $$CONTAINER_NAME --rm dispatcher train $(ARGS) \
		> logs/$$CONTAINER_NAME.log 2>&1 & \
	echo $$CONTAINER_NAME > $(PID_DIR)/train.cid

inference:
	CONTAINER_NAME=inference_$(shell date +%Y%m%d_%H%M%S); \
	nohup docker compose -f $(COMPOSE_FILE) run --name $$CONTAINER_NAME --rm dispatcher inference $(ARGS) \
		> logs/$$CONTAINER_NAME.log 2>&1 & \
	echo $$CONTAINER_NAME > $(PID_DIR)/inference.cid

evaluate:
	CONTAINER_NAME=evaluate_$(shell date +%Y%m%d_%H%M%S); \
	nohup docker compose -f $(COMPOSE_FILE) run --name $$CONTAINER_NAME --rm dispatcher evaluate $(ARGS) \
		> logs/$$CONTAINER_NAME.log 2>&1 & \
	echo $$CONTAINER_NAME > $(PID_DIR)/evaluate.cid

estimate:
	CONTAINER_NAME=estimate_$(shell date +%Y%m%d_%H%M%S); \
	nohup docker compose -f $(COMPOSE_FILE) run --name $$CONTAINER_NAME --rm dispatcher estimate $(ARGS) \
		> logs/$$CONTAINER_NAME.log 2>&1 & \
	echo $$CONTAINER_NAME > $(PID_DIR)/estimate.cid

precheck:
	CONTAINER_NAME=precheck_$(shell date +%Y%m%d_%H%M%S); \
	nohup docker compose -f $(COMPOSE_FILE) run --name $$CONTAINER_NAME --rm dispatcher precheck $(ARGS) \
		> logs/$$CONTAINER_NAME.log 2>&1 & \
	echo $$CONTAINER_NAME > $(PID_DIR)/precheck.cid

run:
	CONTAINER_NAME=run_$(shell date +%Y%m%d_%H%M%S); \
	nohup docker compose -f $(COMPOSE_FILE) run --name $$CONTAINER_NAME --rm dispatcher run $(ARGS) \
		> logs/$$CONTAINER_NAME.log 2>&1 & \
	echo $$CONTAINER_NAME > $(PID_DIR)/run.cid
	
debug:
	CONTAINER_NAME=debug_$(shell date +%Y%m%d_%H%M%S); \
	docker compose -f $(COMPOSE_FILE) run --name $$CONTAINER_NAME --rm dispatcher debug $(ARGS);

# Entry points
preprocess_input:
# 	docker compose -f $(COMPOSE_FILE) run --rm dispatcher preprocess_input $(ARGS)
	$(call run_or_parallel,preprocess_input)
preprocess_label:
	docker compose -f $(COMPOSE_FILE) run --rm dispatcher preprocess_label $(ARGS)

# 상태 확인
status:
	@echo "Running container IDs:"; \
	ls $(PID_DIR) | xargs -I {} sh -c 'echo "{} -> $$(cat $(PID_DIR)/{})"'

# 종료 명령
stop-train:
	-@docker stop $$(cat $(PID_DIR)/train.cid) 2>/dev/null || true
	@rm -f $(PID_DIR)/train.cid

stop-inference:
	-@docker stop $$(cat $(PID_DIR)/inference.cid) 2>/dev/null || true
	@rm -f $(PID_DIR)/inference.cid

stop-evaluate:
	-@docker stop $$(cat $(PID_DIR)/evaluate.cid) 2>/dev/null || true
	@rm -f $(PID_DIR)/evaluate.cid

stop-run:
	-@docker stop $$(cat $(PID_DIR)/run.cid) 2>/dev/null || true
	@rm -f $(PID_DIR)/run.cid

stop:
	@echo "Stopping all tracked containers..."
	@for cidfile in $(PID_DIR)/*.cid; do \
		[ -f $$cidfile ] && docker stop $$(cat $$cidfile) 2>/dev/null || true; \
		rm -f $$cidfile; \
	done
