SHELL := /bin/bash

PYTHON_BIN := python
VENV := library-ve
PYTHON := $(VENV)/Scripts/python
PIP := $(VENV)/Scripts/pip
NOTEBOOK_DIR := theory

.PHONY: help venv install freeze run test-finetuned clean

help:
	@echo "Comandos disponibles:"
	@echo "  make run             Ejecuta main.py levantando un venv e instalando dependencias"
	@echo "  make test-finetuned  Ejecuta test_finetunned.py levantando un venv e instalando dependencias"
	@echo "  make freeze          Vuelca las dependencias instaladas a requirements.txt"
	@echo "  make clean           Elimina $(VENV)"

venv:
	@test -d "$(VENV)" || $(PYTHON_BIN) -m venv "$(VENV)"

install: venv
	@$(PYTHON) -m pip install --upgrade pip
	@$(PIP) install -r requirements.txt

freeze: venv
	@$(PIP) freeze > requirements.txt

run: venv install
	@$(PYTHON) src/main.py

test-finetuned: venv install
	@$(PYTHON) src/test_finetunned.py

clean:
	@rm -rf "$(VENV)"
