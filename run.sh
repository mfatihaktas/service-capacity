#!/bin/bash
echo $1 $2 $3

PY=python3

if [ $1 = 'c' ]; then
  $PY cap_finder.py
elif [ $1 = 'p' ]; then
  # $PY popularity.py
  $PY paper_plotting.py
elif [ $1 = 'e' ]; then
  $PY exp.py
elif [ $1 = 'b' ]; then
  $PY bucket_model.py
elif [ $1 = 'bs' ]; then
  $PY bucket_sim.py
else
  echo "Arg did not match!"
fi