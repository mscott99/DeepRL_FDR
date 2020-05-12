#!/usr/bin/env bash


docker run --rm -v `pwd`:/shaang/DeepRL --entrypoint python3 indep_pycharm:1.2 /shaang/DeepRL/examples.py
