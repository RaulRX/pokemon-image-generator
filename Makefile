SHELL := /bin/bash

PYTHON_BIN := python
VENV := library-ve
PYTHON := $(VENV)/Scripts/python
PIP := $(VENV)/Scripts/pip
KERNEL_NAME := library-ve-kernel
NOTEBOOK := oldbookillustration_generator.ipynb

.PHONY: help venv install freeze run test-finetuned kernel notebook clean

help:
	@echo "Comandos disponibles:"
	@echo "  make run             Ejecuta main.py levantando un venv e instalando dependencias"
	@echo "  make test-finetuned  Ejecuta test_finetunned.py levantando un venv e instalando dependencias"
	@echo "  make kernel          Registra el kernel Jupyter $(KERNEL_NAME) con las dependencias instaladas"
	@echo "  make notebook        Ejecuta $(NOTEBOOK) usando el kernel $(KERNEL_NAME)"
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

kernel: install
	@$(PYTHON) -m ipykernel install --user --name "$(KERNEL_NAME)" --display-name "$(KERNEL_NAME)"

notebook: kernel
	@$(PYTHON) -m jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.kernel_name="$(KERNEL_NAME)" "$(NOTEBOOK)"

clean:
	@rm -rf "$(VENV)"
