build:
	docker compose build

rebuild:
	docker compose build --no-cache

run:
	docker compose run --rm pytorch-app

exec:
	docker compose exec pytorch-app bash