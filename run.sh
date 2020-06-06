#!/bin/bash
echo $1 $2 $3

PY=python3

if [ $1 = 'i' ]; then
  source /home/mfa51/service-capacity-virtualenv/bin/activate
elif [ $1 = 'c' ]; then
  $PY cap_finder.py
elif [ $1 = 't' ]; then
  $PY tcom_plot.py
elif [ $1 = 'p' ]; then
  # $PY popularity.py
  $PY paper_plotting.py
elif [ $1 = 'e' ]; then
  $PY exp.py
elif [ $1 = 'b' ]; then
  # $PY bucket_model.py
  # $PY bucket_viz.py
  $PY bucket_wchoice.py
  # $PY bucket_conjecture.py
elif [ $1 = 'l' ]; then
  $PY load_imbalance.py
elif [ $1 = 'v' ]; then
  $PY bucket_wcode.py
elif [ $1 = 'bs' ]; then
  $PY bucket_sim.py
else
  echo "Arg did not match!"
fi