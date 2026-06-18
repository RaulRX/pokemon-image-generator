SHELL := /bin/bash

PYTHON_BIN := python
VENV := pokemon-ve
PYTHON := $(VENV)/Scripts/python
PIP := $(VENV)/Scripts/pip
KERNEL_NAME := pokemon-generator
NOTEBOOK_DIR := theory

.PHONY: help setup venv install kernel check notebook freeze clean

help:
	@echo "Comandos disponibles:"
	@echo "  make setup     Crea el venv, instala dependencias y registra el kernel de Jupyter"
	@echo "  make check     Verifica imports y versiones principales"
	@echo "  make notebook  Abre JupyterLab sobre $(NOTEBOOK_DIR)/"
	@echo "  make freeze    Vuelca las dependencias instaladas a requirements.txt"
	@echo "  make clean     Elimina $(VENV)"

setup: venv install kernel check

venv:
	@test -d "$(VENV)" || $(PYTHON_BIN) -m venv "$(VENV)"

install: venv
	@$(PYTHON) -m pip install --upgrade pip
	@$(PIP) install -r requirements.txt

kernel: venv
	@$(PYTHON) -m ipykernel install --user --name "$(KERNEL_NAME)" --display-name "Python ($(KERNEL_NAME))"

check: venv
	@$(PYTHON) -c 'import importlib, platform; mods = ["torch", "torchvision", "diffusers", "transformers", "accelerate", "datasets", "PIL", "cv2", "matplotlib", "numpy"]; [importlib.import_module(m) for m in mods]; print("Python:", platform.python_version()); print("Todos los imports principales funcionan.")'

notebook: venv
	@$(PYTHON) -m jupyter lab "$(NOTEBOOK_DIR)"

freeze: venv
	@$(PIP) freeze > requirements.txt

clean:
	@rm -rf "$(VENV)"
