#!/bin/bash

SCRIPTS_ROOT=$SCRATCH/decoupled-contact/scripts

SCRIPTS=(
    # "$SCRIPTS_ROOT/friction/armadillo-roller.json"
    # "$SCRIPTS_ROOT/friction/armadillo-roller-baseline.json"
    # "$SCRIPTS_ROOT/stress-tests/trash-compactor-octocat.json"
    # "$SCRIPTS_ROOT/stress-tests/trash-compactor-octocat-baseline.json"
    # "$SCRIPTS_ROOT/rigid/screw.json"
    # "$SCRIPTS_ROOT/rigid/screw-baseline.json"
    # "$SCRIPTS_ROOT/stress-tests/mat-twist-P1-coarse.json"
    # "$SCRIPTS_ROOT/stress-tests/mat-twist-P1.json"
    # "$SCRIPTS_ROOT/stress-tests/mat-twist-P2.json"
    "$SCRIPTS_ROOT/graspsim/grasp-gargoyle.json"
    # "$SCRIPTS_ROOT/graspsim/grasp-gargoyle-baseline.json"
    # "$SCRIPTS_ROOT/baseline/full-full.json"
    # "$SCRIPTS_ROOT/baseline/full-reduced.json"
    # "$SCRIPTS_ROOT/baseline/reduced-full.json"
    # "$SCRIPTS_ROOT/baseline/reduced-reduced.json"
)

LOGS_DIR=$SCRATCH/decoupled-contact/results/logs
mkdir -p $LOGS_DIR

JOB=$SCRATCH/decoupled-contact/tools/hpc/job.sh

for SCRIPT in ${SCRIPTS[@]}; do
    JOB_NAME=$(basename -- "$SCRIPT")
    JOB_NAME="${JOB_NAME%.*}"
    sbatch \
        -J "$JOB_NAME" \
        -o "$LOGS_DIR/$JOB_NAME.out" -e "$LOGS_DIR/$JOB_NAME.err" \
        "$JOB" "$SCRIPT"
done
