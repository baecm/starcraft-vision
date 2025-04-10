build:
	docker compose build

rebuild:
	docker compose build --no-cache

exec:
	docker compose exec pytorch-app bash

up: 
	docker compose up -d pytorch-app

down:
	docker compose down --remove-orphans

preprocess_input:
	docker compose run --rm preprocess_input $(ARGS)

preprocess_label:
	docker compose run --rm preprocess_label $(ARGS)

preprocess_pair:
	docker compose run --rm preprocess_pair $(ARGS)

run:
	docker compose run --rm pytorch-app

preprocess_all:
	$(foreach replay,$(ARGS),\
		make preprocess_input ARGS="--replays $(replay)" && \
		make preprocess_label ARGS="--replays $(replay)" && \
		make preprocess_pair ARGS="--replays $(replay)";)