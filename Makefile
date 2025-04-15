COMPOSE_FILE=infra/docker-compose.yml

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

preprocess_input:
	make run CMD=preprocess_input ARGS="$(ARGS)"

preprocess_label:
	make run CMD=preprocess_label ARGS="$(ARGS)"

preprocess_pair:
	make run CMD=preprocess_pair ARGS="$(ARGS)"

train:
	make run CMD=train ARGS="$(ARGS)"

evaluate:
	make run CMD=evaluate ARGS="$(ARGS)"
