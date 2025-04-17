install:
	pip install --upgrade pip && \
	pip install -r requirements.txt

format:
	black *.py 

lint:
	pylint --disable=R,C *.py

refactor: format lint

test:
	python -m pytest -vv --cov=tests test_*.py

all: install refactor test