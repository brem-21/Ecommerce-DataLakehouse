install:
	pip install --upgrade pip && \
	pip install -r requirements.txt

format:
	black *.py 

test:
	python -m pytest -vv --cov=tests test_*.py

all: install format test