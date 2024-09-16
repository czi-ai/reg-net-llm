# Data Preprocessing Scripts (SLURM Parallelization)

## start_slurm_preprocess.sh

This file gets the total number of files to process from the correct directories. Then makes a parallelized SLURM call on `array_preprocess.sh`, according to how many files there are. This parallelizes on each cell-type, and nested parallelization takes place later for each file in each cell-type. (Tissue)

_Example Command:_ `source start_slurm_preprocess.sh "initial"`

_Example Command:_ `source start_slurm_preprocess.sh "run_timed_out" "786994"`

The **RUN_TYPE** argument lets the script run the pipeline from scratch or on specific cell-types that weren't successfully processed _(options: initial, run_env_fail, run_timed_out, run_failed, and run_unaccounted)_. For any non-"initial" runs, you must also include the **JOB_ID** from which you've identified the cell-types you would like to re-run. These specific cell-types are identified from the job .out files and are parsed & organized by the `./check_out.py` script: `python check_out.py`.

## array_preprocess.sh

Creates the `slurm_out` directory. Lists and sorts the cell-types in the data directory. Calls `preprocess_cellxgene.sh` for each cell-type. (Cell-type)

_Note: #SBATCH --time=2:00:00 was enough time to run most cell-types on our cluster._

## preprocess_cellxgene.sh

This processes the desired cell-types through the established pipeline.