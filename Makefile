COMPOSE_FILE=infra/docker-compose.yml
PID_DIR=pids

# PID 디렉토리 생성
$(shell mkdir -p $(PID_DIR))
$(shell mkdir -p logs)

define run_or_parallel
	@REPLAY_COUNT=$(shell echo $(ARGS) | sed -n 's/.*--replays\([^"]*\).*/\1/p' | wc -w); \
	if [ "$$REPLAY_COUNT" -le 1 ]; then \
		echo "[Makefile] Running $(1) as single task"; \
		nohup docker compose -f infra/docker-compose.yml run --rm dispatcher $(1) $(ARGS) \
			> logs/$(1)_$$(date +%Y%m%d_%H%M%S).log 2>&1 & \
		echo $$! > $(PID_DIR)/$(1).pid; \
	else \
		echo "[Makefile] Running $(1) in parallel"; \
		nohup bash scripts/preprocess_batch.sh $(1) $(ARGS) \
			> logs/$(1)_$$(date +%Y%m%d_%H%M%S).log 2>&1 & \
		echo $$! > $(PID_DIR)/$(1).pid; \
	fi
endef

build:
	nohup docker compose -f $(COMPOSE_FILE) build \
		> logs/build_$(shell date +%Y%m%d_%H%M%S).log 2>&1 &
	echo $$! > $(PID_DIR)/build.pid

rebuild:
	nohup docker compose -f $(COMPOSE_FILE) build --no-cache \
		> logs/rebuild_$(shell date +%Y%m%d_%H%M%S).log 2>&1 &
	echo $$! > $(PID_DIR)/rebuild.pid

up:
	nohup docker compose -f $(COMPOSE_FILE) up -d \
		> logs/up_$(shell date +%Y%m%d_%H%M%S).log 2>&1 &
	echo $$! > $(PID_DIR)/up.pid

down:
	nohup docker compose -f $(COMPOSE_FILE) down --remove-orphans \
		> logs/down_$(shell date +%Y%m%d_%H%M%S).log 2>&1 &
	echo $$! > $(PID_DIR)/down.pid

run:
	nohup docker compose -f $(COMPOSE_FILE) run --rm dispatcher $(CMD) $(ARGS) \
		> logs/run_$(shell date +%Y%m%d_%H%M%S).log 2>&1 &
	echo $$! > $(PID_DIR)/run.pid

train:
	nohup docker compose -f $(COMPOSE_FILE) run --rm dispatcher train $(ARGS) \
		> logs/train_$(shell date +%Y%m%d_%H%M%S).log 2>&1 &
	echo $$! > $(PID_DIR)/train.pid

inference:
	nohup docker compose -f $(COMPOSE_FILE) run --rm dispatcher inference $(ARGS) \
		> logs/inference_$(shell date +%Y%m%d_%H%M%S).log 2>&1 &
	echo $$! > $(PID_DIR)/inference.pid

evaluate:
	nohup docker compose -f $(COMPOSE_FILE) run --rm dispatcher evaluate $(ARGS) \
		> logs/evaluate_$(shell date +%Y%m%d_%H%M%S).log 2>&1 &
	echo $$! > $(PID_DIR)/evaluate.pid

# Entry points
preprocess_input:
	$(call run_or_parallel,preprocess_input)

preprocess_label:
	$(call run_or_parallel,preprocess_label)

preprocess_pair:
	$(call run_or_parallel,preprocess_pair)

# PID 관리용 타겟
status:
	@echo "Running PIDs:"; \
	ls $(PID_DIR) | xargs -I {} sh -c 'echo "{} -> $$(cat $(PID_DIR)/{})"'

stop:
	@echo "Stopping all tasks..."; \
	for pidfile in $(PID_DIR)/*.pid; do \
		kill $$(cat $$pidfile) 2>/dev/null || true; \
		rm -f $$pidfile; \
	done
