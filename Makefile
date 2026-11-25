# NeuroGenesis developer commands. Python commands use $(PY) (default: the active python).
PY ?= python
PYTEST = PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PY) -m pytest
COMPOSE = docker compose -f docker/mc/compose.yaml

.PHONY: help mod proto dev-server up down logs py test itest bench watch clean

help:
	@echo "mod         build the Fabric mod (mod/build/dist/neurogenesis-*.jar) and run its unit tests"
	@echo "proto       regenerate the Python protobuf modules from proto/"
	@echo "dev-server  run a development server from Gradle (ports 25565/25570/25575)"
	@echo "up/down     start/stop the Docker server; logs follows its output"
	@echo "py          install the Python package (editable) with test dependencies"
	@echo "test        unit tests (Java + Python)"
	@echo "itest       integration tests against a running server (dev-server or up)"
	@echo "bench       lockstep throughput benchmark against a running server"
	@echo "watch       run smell-following agents at real-time speed (watch with a 26.3 client)"

mod:
	cd mod && ./gradlew build

proto:
	scripts/gen_proto.sh

dev-server:
	cd mod && ./gradlew runServer

up: mod
	$(COMPOSE) up -d
	@echo "Waiting for the control port..."
	@python3 -c "import socket, time; d = time.time() + 300; \
		exec('while True:\n try:\n  socket.create_connection((\"127.0.0.1\", 25570), 2).close(); break\n except OSError:\n  assert time.time() < d, \"timed out\"; time.sleep(1)')" \
		&& echo "Server ready."

down:
	$(COMPOSE) down

logs:
	$(COMPOSE) logs -f mc

py:
	$(PY) -m pip install -e "python[dev]"

test:
	cd mod && ./gradlew test
	cd python && $(PYTEST)

itest:
	cd python && $(PYTEST) -m server -v

bench:
	$(PY) scripts/bench_steps.py

watch:
	$(PY) scripts/watch.py --agents 3

clean:
	cd mod && ./gradlew clean
