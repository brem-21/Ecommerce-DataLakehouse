install:
	pip install --upgrade pip && \
	pip install -r requirements.txt

format:
	black scripts/*.py tests/*.py 

test:
	python -m pytest -vv --cov=tests test_*.py

refactor: format lint

push-to-main:
	push-to-main:
	@git config user.name "brem-21" 
	@git config user.email "brempong.dankwah@amalitech.com"
	@git fetch origin
	@git checkout main
	@git merge dev --no-edit
	@git push origin main
	@git checkout dev


all: install format test push-to-main