find . -name "*py"|xargs python3 -m autoflake -i --remove-all-unused-imports --remove-unused-variables --ignore-init-module-imports
black .
isort .
