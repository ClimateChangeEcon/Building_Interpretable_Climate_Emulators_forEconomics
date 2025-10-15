#!/usr/bin/env bash

PY=python3.12

# run relative to this script's directory (assumes no spaces in path)
cd $(dirname $0) || exit 1

echo "Working dir: $(pwd)"
echo "Using Python: $(which $PY)"

# Generate models only if 'result/' does not exist
if [ -d result ]; then
  echo "'result/' exists"
else
  echo "'result/' not found — running model generation..."
  echo "############################################################"
  echo "# NOTICE"
  echo "# This is a non-convex optimization problem solved using a"
  echo "# differential-evolution algorithm."
  echo "# Stochastic perturbations and round-off errors may cause"
  echo "# slight variability across runs."
  echo "# Observed variability is below 1% and is insignificant for simulation purposes."
  echo "############################################################"
  read -p 'Press ENTER to continue...'
    
  $PY solver.py PI 250 MMM
  $PY solver.py PD 250 MMM
  $PY solver.py PI 250 MESMO
  $PY solver.py PD 250 MESMO
  $PY solver.py PI 250 CLIMBER2
  $PY solver.py PD 250 CLIMBER2
  mv result_NEW result
fi

mkdir -p fig/PD fig/PI figs_replication

$PY fig_analysis.py
$PY fig_RCP_RF.py
$PY fig_benchmark_pulse.py
$PY fig_sim.py