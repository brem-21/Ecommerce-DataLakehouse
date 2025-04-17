install:
	pip install --upgrade pip && \
	pip install -r requirements.txt

format:
	black scripts/*.py tests/*.py 

test:
	python -m pytest -vv --cov=tests tests/test_*.py

refactor: format lint

push-to-main:
	@git config user.name "brem-21" 
	@git config user.email "brempong.dankwah@amalitech.com"
	@git add -A && git commit -m "Auto commit before pushing to main" || echo "No changes to commit"
	@git fetch origin
	@git checkout main
	@git merge dev --no-edit --allow-unrelated-histories
	@git push origin main
	@git checkout dev


all: install format test push-to-main