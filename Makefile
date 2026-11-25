# NeuroGenesis developer commands. Python commands use $(PY) (default: the active python).
PY ?= python
PYTEST = PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 $(PY) -m pytest
COMPOSE = docker compose -f docker/mc/compose.yaml

.PHONY: help mod proto dev-server up down logs py test itest bench clean

help:
	@echo "mod         build the Fabric mod (mod/build/dist/neurogenesis-*.jar) and run its unit tests"
	@echo "proto       regenerate the Python protobuf modules from proto/"
	@echo "dev-server  run a development server from Gradle (ports 25565/25570/25575)"
	@echo "up/down     start/stop the Docker server; logs follows its output"
	@echo "py          install the Python package (editable) with test dependencies"
	@echo "test        unit tests (Java + Python)"
	@echo "itest       integration tests against a running server (dev-server or up)"
	@echo "bench       lockstep throughput benchmark against a running server"

mod:
	cd mod && ./gradlew build

proto:
	scripts/gen_proto.sh

dev-server:
	cd mod && ./gradlew runServer

up: mod
	$(COMPOSE) up -d
	@echo "Waiting for the control port..."
	@$(PY) -c "from neurogenesis.mc import McClient; McClient(connect_timeout=300).close()" && echo "Server ready."

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

clean:
	cd mod && ./gradlew clean
