#!/usr/bin/env bash

docker run --rm -v `pwd`:/shaang/DeepRL --entrypoint /bin/bash -it indep_pycharm:1.2
