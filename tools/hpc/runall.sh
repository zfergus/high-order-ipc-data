#!/bin/bash

SCRIPTS_ROOT=$SCRATCH/decoupled-contact/scripts

SCRIPTS=(
    # "$SCRIPTS_ROOT/balance/armadillo-balance-coarse-optimized.json"
    # "$SCRIPTS_ROOT/balance/armadillo-balance-coarse.json"
    # "$SCRIPTS_ROOT/balance/armadillo-balance-fine.json"
    # "$SCRIPTS_ROOT/friction/armadillo-roller-P2-wCM.json"
    # "$SCRIPTS_ROOT/friction/armadillo-roller-P2.json"
    # "$SCRIPTS_ROOT/friction/armadillo-roller.json"
    # "$SCRIPTS_ROOT/friction/armadillo-roller-baseline.json"
    # "$SCRIPTS_ROOT/stress-tests/trash-compactor-octocat-P2-r2.json"
    # "$SCRIPTS_ROOT/stress-tests/trash-compactor-octocat-P2-wCM-r2.json"
    # "$SCRIPTS_ROOT/stress-tests/trash-compactor-octocat.json"
    # "$SCRIPTS_ROOT/stress-tests/trash-compactor-octocat-baseline.json"
    # "$SCRIPTS_ROOT/rigid/screw.json"
    # "$SCRIPTS_ROOT/rigid/screw-baseline.json"
    # "$SCRIPTS_ROOT/stress-tests/mat-twist-P1-coarse.json"
    # "$SCRIPTS_ROOT/stress-tests/mat-twist-P1-40x40.json"
    # "$SCRIPTS_ROOT/stress-tests/mat-twist-P1-60x60.json"
    # "$SCRIPTS_ROOT/stress-tests/mat-twist-P1-70x70.json"
    # "$SCRIPTS_ROOT/stress-tests/mat-twist-P1-80x80.json"
    # "$SCRIPTS_ROOT/stress-tests/mat-twist-P1-90x90.json"
    # "$SCRIPTS_ROOT/stress-tests/mat-twist-P1-92x92.json"
    # "$SCRIPTS_ROOT/stress-tests/mat-twist-P1-94x94.json"
    # "$SCRIPTS_ROOT/stress-tests/mat-twist-P1-96x96.json"
    # "$SCRIPTS_ROOT/stress-tests/mat-twist-P1-98x98.json"
    # "$SCRIPTS_ROOT/stress-tests/mat-twist-P1-100x100.json"
    # "$SCRIPTS_ROOT/stress-tests/mat-twist-P1-150x150.json"
    # "$SCRIPTS_ROOT/stress-tests/mat-twist-P1-225x225.json"
    # "$SCRIPTS_ROOT/stress-tests/mat-twist-P1.json"
    # "$SCRIPTS_ROOT/stress-tests/mat-twist-P2.json"
    # "$SCRIPTS_ROOT/graspsim/grasp-gargoyle.json"
    # "$SCRIPTS_ROOT/graspsim/grasp-gargoyle-baseline.json"
    # "$SCRIPTS_ROOT/baseline/full-full.json"
    # "$SCRIPTS_ROOT/baseline/full-reduced.json"
    # "$SCRIPTS_ROOT/baseline/reduced-full.json"
    # "$SCRIPTS_ROOT/baseline/reduced-reduced.json"
    # "$SCRIPTS_ROOT/microstructure/microstructure-P1-coarse.json"
    # "$SCRIPTS_ROOT/microstructure/microstructure-P1-medium-fixAL.json"
    # "$SCRIPTS_ROOT/microstructure/microstructure-P1-medium-E=6e7.json"
    # "$SCRIPTS_ROOT/microstructure/microstructure-P1-medium-E=6e7-r2.json"
    # "$SCRIPTS_ROOT/microstructure/microstructure-P1-fine-fixAL.json"
    # "$SCRIPTS_ROOT/microstructure/microstructure-P1.json"
    # "$SCRIPTS_ROOT/microstructure/microstructure-P2.json"
    # "$SCRIPTS_ROOT/microstructure/microstructure-P2-E=6e7.json"
    # "$SCRIPTS_ROOT/microstructure/microstructure-P3.json"
    # "$SCRIPTS_ROOT/microstructure/microstructure-P4.json"
    # "$SCRIPTS_ROOT/bounce/bounce-P1-baseline-coarse.json"
    # "$SCRIPTS_ROOT/bounce/bounce-P1-baseline-fine.json"
    # "$SCRIPTS_ROOT/bounce/bounce-P1.json"
    # "$SCRIPTS_ROOT/bounce/bounce-P4.json"
    # "$SCRIPTS_ROOT/higher-order/squish-beam-P1-ref.json"
    # "$SCRIPTS_ROOT/higher-order/squish-beam-P1.json"
    # "$SCRIPTS_ROOT/higher-order/squish-beam-P2.json"
    # "$SCRIPTS_ROOT/higher-order/squish-beam-P3.json"
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
