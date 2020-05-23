#!/usr/bin/env bash
docker run --rm -it\
	-e DISPLAY\
	--name hello_Ngan \
	-v /tmp/.X11-unix:/tmp/.X11-unix\
	-v $XAUTHORITY:/shaang/.Xauthority\
	-v `pwd`:/shaang/DeepRL\
	--entrypoint /bin/bash\
	opt_params:0.1
