#!/bin/bash

DIRECTORY="/burg/pmg/collab/scGraphLLM/data/cellxgene/cell_type_005"
FILE_COUNT=$(ls -1 "$DIRECTORY" | wc -l)

sbatch --array=1-${FILE_COUNT} array_preprocess.sh