#!/usr/bin/env bash
docker run --rm -it\
	-e DISPLAY\
	--name hello_Ngan \
	-v /tmp/.X11-unix:/tmp/.X11-unix\
	-v $XAUTHORITY:/shaang/.Xauthority\
	-v ~/programming/machine_learning/paper_actor_critic/DeepRL/:/root/DeepRL\
	-v `pwd`:/shaang/DeepRL\
	--entrypoint /bin/bash\
	fresh_root:0.1
