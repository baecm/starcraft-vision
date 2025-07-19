COMPOSE_FILE=infra/docker-compose.yml
PID_DIR=pids

# PID 디렉토리 생성
$(shell mkdir -p $(PID_DIR))

define run_or_parallel
	@REPLAY_COUNT=$(shell echo $(ARGS) | sed -n 's/.*--replays\([^"]*\).*/\1/p' | wc -w); \
	if [ "$$REPLAY_COUNT" -le 1 ]; then \
		echo "[Makefile] Running $(1) as single task"; \
		docker compose -f infra/docker-compose.yml run --rm dispatcher $(1) $(ARGS) & \
		echo $$! > $(PID_DIR)/$(1).pid; \
	else \
		echo "[Makefile] Running $(1) in parallel"; \
		bash scripts/preprocess_batch.sh $(1) $(ARGS) & \
		echo $$! > $(PID_DIR)/$(1).pid; \
	fi
endef

build:
	docker compose -f $(COMPOSE_FILE) build & \
	echo $$! > $(PID_DIR)/build.pid

rebuild:
	docker compose -f $(COMPOSE_FILE) build --no-cache & \
	echo $$! > $(PID_DIR)/rebuild.pid

up:
	docker compose -f $(COMPOSE_FILE) up -d & \
	echo $$! > $(PID_DIR)/up.pid

down:
	docker compose -f $(COMPOSE_FILE) down --remove-orphans & \
	echo $$! > $(PID_DIR)/down.pid

run:
	docker compose -f $(COMPOSE_FILE) run --rm dispatcher $(CMD) $(ARGS) & \
	echo $$! > $(PID_DIR)/run.pid

train:
	mkdir -p logs
	nohup script -q -f -c "make run CMD=train ARGS='$(ARGS)'" logs/train_$(shell date +%Y%m%d_%H%M%S).log > /dev/null 2>&1 &
	echo $$! > $(PID_DIR)/train.pid

inference:
	mkdir -p logs
	script -q -f -c "make run CMD=inference ARGS='$(ARGS)'" logs/inference_$(shell date +%Y%m%d_%H%M%S).log & \
	echo $$! > $(PID_DIR)/inference.pid

evaluate:
	mkdir -p logs
	script -q -f -c "make run CMD=evaluate ARGS='$(ARGS)'" logs/evaluate_$(shell date +%Y%m%d_%H%M%S).log & \
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
