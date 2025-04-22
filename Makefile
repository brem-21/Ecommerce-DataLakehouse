install:
	pip install --upgrade pip && \
	pip install -r requirements.txt

format:
	black scripts/*.py tests/*.py glue_script/*.py

test:
	python -m pytest -vv --cov=tests tests/test_*.py

refactor: format lint

all: install format test 