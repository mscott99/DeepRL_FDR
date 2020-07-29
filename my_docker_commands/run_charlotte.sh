#!/usr/bin/env bash
docker run --rm -it\
	--name hello_Ngan_second \
	-v `pwd`:/root/DeepRL\
	--entrypoint /bin/bash\
	first_build:0.1
