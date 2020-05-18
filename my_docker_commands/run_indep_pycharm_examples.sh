#!/usr/bin/env bash
xhost +local:root
docker run --rm -it\
       -e DISPLAY \
       --name hello_Ngan\
        --net=host \
        -v /tmp/.X11-unix:/tmp/.X11-unix \
        -v `pwd`:/shaang/DeepRL \
        with_screen:7.2
