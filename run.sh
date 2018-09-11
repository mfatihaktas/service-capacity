#!/bin/bash
echo $1 $2 $3

PY=python3

if [ $1 = 'c' ]; then
  $PY cap_finder.py
elif [ $1 = 'p' ]; then
  $PY popularity.py
else
  echo "Arg did not match!"
fi