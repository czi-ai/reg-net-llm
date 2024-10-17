# Data Preprocessing Scripts (SLURM Parallelization) & Caching

_NOTE: To avoid preprocessing issues (`ValueError: cannot specify integer "bins" when input data contains infinity`), please add `X = X.astype(np.float64)` under `X = _get_obs_rep(adata, layer=layer)` at the start of the `_highly_variable_genes_single_batch()` function in `/scanpy/preprocessing/_highly_variavle_genes.py`. Otherwise __inf__ values likely be encountered and interrupt the preprocessing on some cell-types._

## Preprocess

### start_slurm_preprocess.sh

_NOTE: `start_slurm_preprocess.sh` will automatically run `array_preprocess.sh` & `preprocess_cellxgene.sh`. Please run `python checkout.py` to obtain the run's success/failure summary._

`start_slurm_preprocess.sh` gets the total number of files to process from the correct directories. Then makes a parallelized SLURM call on `array_preprocess.sh`, according to how many files there are. This parallelizes on each cell-type, and nested parallelization takes place later for each file in each cell-type. (Tissue)

_Example Command:_ `source start_slurm_preprocess.sh "initial" "1024"`

_Example Command:_ `source start_slurm_preprocess.sh "run_timed_out" "1024" "786994"`

1. `RUN_TYPE`: The **RUN_TYPE** argument lets the script run the pipeline from scratch _or_ on specific cell-types that weren't successfully processed _(options: initial, run_env_fail, run_timed_out, run_failed, and run_unaccounted)_. 
1. `ARACNE_TOP_N_HVG`: There is also the **ARACNE_TOP_N_HVG** parameter which is the number of top highly variable genes on which to generate the ARACNe networks (1024, 2048, 4096, or unspecified "": the entire set of genes). 
1. `JOB_ID`: For any non-"initial" runs, you must also include the **JOB_ID** from which you've identified the cell-types you would like to re-run. These specific cell-types are identified from the job .out files and are parsed & organized by the `./checkout.py` script: `python checkout.py`.

### array_preprocess.sh

Creates the `slurm_out` directory. Lists and sorts the cell-types in the data directory. Calls `preprocess_cellxgene.sh` for each cell-type. (Cell-type)

_Note: #SBATCH --time=2:00:00 was enough time to run most cell-types on our cluster._

### preprocess_cellxgene.sh

This processes the desired cell-types through the established pipeline: `/scGraphLLM/scGraphLLM/preprocess.py` followed by ARACNe network generation.

## Debug SLURM

After running `start_slurm_preprocess.sh`, please run `python checkout.py` to obtain the run's success/failure summary. Use the output of `checkout.py` to diagnose issues and rerun `start_slurm_preprocess.sh` as needed.

Running `python checkout.py` is necessary to identifying the successfully run folders to then be used in the cache generation (following step) for the training/validation model data.

## Generate Cache for Model Use

Once the preprocessing is successfully completed, you first have to organize the directories into __train__, __ValSG__, and __ValHG__ assignments. This is done by running `python outdir_gen.py` in `/scGraphLLM/scripts/`. This file generates an `aracne_<ARACNE_TOP_N_HVG>_outdir.csv` file (wherever specified). Make sure to change `ARACNE_TOP_N_HVG`, `run`, and any directory variable to the correct value for your run. The output csv is used in the next step to organize the cache files in the correct directories.

You can now run `sbatch generate_cache_data.sh <ARACNE_TOP_N_HVG>` from `/scGraphLLM` to generate the cache files that will be used to train the model. `<ARACNE_TOP_N_HVG>` is the number of top highly variable genes on which to generate the ARACNe networks. Make sure to change any cluster-specific variables (`--ntasks` (# cpus) & `--num-proc`) in `generate_cache_data.sh` before running.