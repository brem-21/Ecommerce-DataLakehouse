install:
	pip install --upgrade pip && \
	pip install -r requirements.txt

format:
	black *.py 

lint:
	pylint --disable=R,C *.py

refactor: format lint

pytest:
	pytest --disable-warnings

all: install refactor pytest