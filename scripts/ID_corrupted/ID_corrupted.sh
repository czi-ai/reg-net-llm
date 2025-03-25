#!/bin/bash

CACHE_DIR="$1"

sbatch scan_train.sh "$CACHE_DIR"
sbatch scan_SG.sh "$CACHE_DIR"
sbatch scan_HOG.sh "$CACHE_DIR"
