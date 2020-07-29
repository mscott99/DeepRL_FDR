#!/usr/bin/env bash
docker run --rm --detach -it\
	--name Ngan \
	-v `pwd`:/root/DeepRL\
	--entrypoint ./my_docker_commands/run_hyper_tuning.sh\
	first_build:0.1\
	> ./current_container
