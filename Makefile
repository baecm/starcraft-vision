COMPOSE_FILE=infra/docker-compose.yml

define run_or_parallel
	@REPLAY_COUNT=$(shell echo $(ARGS) | sed -n 's/.*--replays\([^"]*\).*/\1/p' | wc -w); \
	if [ "$$REPLAY_COUNT" -le 1 ]; then \
		echo "[Makefile] Running $(1) as single task"; \
		docker compose -f infra/docker-compose.yml run --rm dispatcher $(1) $(ARGS); \
	else \
		echo "[Makefile] Running $(1) in parallel"; \
		bash scripts/preprocess_batch.sh $(1) $(ARGS); \
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

run:
	docker compose -f $(COMPOSE_FILE) run --rm dispatcher $(CMD) $(ARGS)

train:
	mkdir -p logs
	script -q -f -c "make run CMD=train ARGS='$(ARGS)'" logs/train_$(shell date +%Y%m%d_%H%M%S).log

inference:
	mkdir -p logs
	script -q -f -c "make run CMD=inference ARGS='$(ARGS)'" logs/inference_$(shell date +%Y%m%d_%H%M%S).log

evaluate:
	mkdir -p logs
	script -q -f -c "make run CMD=evaluate ARGS='$(ARGS)'" logs/evaluate_$(shell date +%Y%m%d_%H%M%S).log

# Entry points
preprocess_input:
	$(call run_or_parallel,preprocess_input)

preprocess_label:
	$(call run_or_parallel,preprocess_label)

preprocess_pair:
	$(call run_or_parallel,preprocess_pair)