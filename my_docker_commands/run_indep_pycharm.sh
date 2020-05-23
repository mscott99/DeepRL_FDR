#!/usr/bin/env bash
#xhost +local:root
docker run --rm --detach\
       -e DISPLAY \
       --name Ngan\
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v `pwd`:/shaang/DeepRL \
        opt_params:0.1\

        #--net=host \
#xhost -local:root
exit
