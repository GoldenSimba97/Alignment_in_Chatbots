#! /bin/sh

clear

export PYTHONPATH=../../src/

python3 ../../src/programy/clients/events/console/client1.py --config ./config.yaml --cformat yaml --logging ./logging.yaml
