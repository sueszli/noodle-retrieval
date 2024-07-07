.PHONY: help decompress fmt sec up reqs conda-run conda-clean docker-run docker-clean

help:
	@printf "Usage: make [target]\n"
	@printf "Targets:\n"
	@printf "\thelp - show this help message\n"
	@printf "\tdecompress - decompress data/* files\n"
	@printf "\tfmt - run formatter\n"
	@printf "\tsec - run security checks\n"
	@printf "\tup - git pull, add, commit, push\n"
	@printf "\treqs - generate requirements.txt\n"
	@printf "\tconda-run - create conda environment\n"
	@printf "\tconda-clean - remove conda environment\n"
	@printf "\tdocker-run - run docker container\n"
	@printf "\tdocker-clean - remove docker container\n"

decompress:
	# validate ./data/* files
	if [ ! -d data ]; then echo "data/ directory not found"; exit 1; fi
	if ! ls data/*-chunk-* &> /dev/null && ! ls data/*.md5 &> /dev/null; then echo "invalid files found in data/"; exit 1; fi

	# create data-merged directory
	rm -rf data-merged
	mkdir data-merged
	echo "created data-merged directory"

	# merge chunks into data-merged directory
	cat data/*-chunk-* > data-merged/merged.tar.gz
	echo "merged chunks into data-merged/merged.tar.gz"

	# validate checksum
	expected_checksum=$(cat data/*.md5)
	actual_checksum=$(md5sum data-merged/merged.tar.gz | awk '{ print $1 }')
	if [ $expected_checksum != $actual_checksum ]; then echo "checksum mismatch"; exit 1; fi
	echo "checksum matched: $expected_checksum == $actual_checksum"

	# untar in data-merged
	tar -xzf data-merged/merged.tar.gz -C data-merged
	rm data-merged/merged.tar.gz
	echo "untarred data-merged/merged.tar.gz"

fmt:
	# sort and remove unused imports
	pip install isort
	isort .
	pip install autoflake
	autoflake --remove-all-unused-imports --recursive --in-place .

	# format
	pip install ruff
	ruff format --config line-length=500 .

sec:
	pip install bandit
	pip install safety
	
	bandit -r .
	safety check --full-report

up:
	git pull
	git add .
	git commit -m "up"
	git push

reqs:
	pip install pipreqs
	rm -rf requirements.txt
	pipreqs .

conda-run:
	# to emulate x86_64 on M1
	conda config --env --set subdir osx-64

	conda deactivate
	conda config --set auto_activate_base false
	conda activate base

	# see: https://blog.balasundar.com/install-older-versions-of-python-using-miniconda-on-mac-m1
	conda create --yes --name main python=3.6.12 anaconda
	conda activate main

	# setuptools requirements (order matters)
	pip install 'pyqt5<5.13'
	pip install pyls-black
	pip install 'pyqtwebengine<5.13'
	pip install --no-deps ruamel.yaml # kind of a hack

	# allennlp requirements (order matters)
	pip install --upgrade pip
	pip install --upgrade setuptools
	pip install --upgrade wheel
	pip install --upgrade thinc
	pip install Cython==0.29.36
	pip install think --no-build-isolation
	pip install spacy --no-build-isolation
	pip install jsonnet --no-build-isolation

	# project requirements
	pip install allennlp==1.2.2 --no-build-isolation
	pip install blingfire==0.1.7
	pip install PyYAML==5.4
	pip install transformers==3.5.1
	pip install --find-links https://download.pytorch.org/whl/torch_stable.html torch==1.6.0
	pip install overrides

	# convenience
	pip install black isort flake8 mypy
	pip install numpy pandas
	pip install matplotlib seaborn

	# take snapshot
	conda env export > conda-environment.yml

conda-clean:
	conda deactivate
	conda remove --yes --name main --all
	conda env list

docker-run:
	docker-compose up
	docker ps --all
	docker exec -it main /bin/bash

docker-clean:
	docker-compose down

	# wipe docker
	docker stop $(docker ps -a -q)
	docker rm $(docker ps -a -q)
	docker rmi $(docker images -q)
	yes | docker container prune
	yes | docker image prune
	yes | docker volume prune
	yes | docker network prune
	yes | docker system prune
	
	# check if successful
	docker ps --all
	docker images
	docker system df
	docker volume ls
	docker network ls