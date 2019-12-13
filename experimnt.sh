#!/usr/bin/env bash
PYTHON=\Users\robotics\Anaconda3\envs\stage_py37\python.exe
SCRIPT_DIR=$(cd $(dirname $0); pwd)
echo $SCRIPT_DIR

for i in {0..9}
do
  PYTHON main_exp_system.py
  echo "OK"
done