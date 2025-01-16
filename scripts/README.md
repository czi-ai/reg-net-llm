# Data Preprocessing & Caching (SLURM)

## ⚠️ Before starting, please check the following ⚠️

1. This pipeline is to be run AFTER the CellxGene dataset is downloaded and organized by cell-type (not tissue type) in `./data/cellxgene/data/cell_type_all`.  

    A. To complete the previous step, please refer to the scGPT pipeline for downloading the data (https://github.com/bowang-lab/scGPT/tree/main/data/cellxgene).  

    B. Following this, run `sbatch ./data/cellxgene/cxg_by_cell_type.sh` - replacing the `$DIRECTORY` variable with your directory path to the `./data/cellxgene/tissue` folder created in the previous step.

1. Make sure the following directory structures are respected:

```md
./data
    ├── cellxgene
    │   ├── ...
    │   └── data
    │       └── cell_type_all
    ├── cellxgene_gene2index.csv
    ├── gene-name-map.csv
    └── regulators.txt
```

```md
./scGraphLLM
    ├── ...
    ├── scGraphLLM
    │   ├── ...
    │   ├── _globals.py
    │   ├── data.py
    │   └── preprocess.py
    └── scripts
        ├── ARACNe3_app_release
        ├── array_preprocess.sh  
        ├── checkout.py
        ├── generate_cache_data.sh
        ├── ID_corrupted
        │   ├── corrupted_files
        │   │   └── ...
        │   ├── corrupt_HOG.py
        │   ├── corrupt_SG.py
        │   ├── corrupt_train.py
        │   ├── delete_corrupted.sh   <- 4. Run this
        │   ├── ID_corrupted.sh       <- 3. Run this
        │   ├── scan_HOG.sh
        │   ├── scan_SG.sh
        │   └── scan_train.sh
        ├── outdir_gen.py
        ├── pipeline.sh               <- 1. Run preprocess | 2. Run cache
        ├── preprocess_cellxgene.sh
        ├── README.md
        ├── start_slurm_cache.sh
        ├── start_slurm_preprocess.sh
        └── success.py
```

3. Make sure to look at the troubleshooting section before running the following pipeline to preemptively fix & recognize potential problems after running.

## What is the Pipeline?

The data pipeline consists of two steps: __Preprocessing__ & __Caching__, both of which can be completed from `./scGraphLLM/scripts/pipeline.sh`. 

1. A. The preprocessing step runs Quality Control (QC) on the `./data/cellxgene/data/cell_type_all` cells in each cell-type. These __QC metrics__ can be adjusted to your specifications in `./scGraphLLM/scGraphLLM/_globals.py`. After filtering out the undesireable cells in a cell-type, the remaining cells are clustered and metacells are created within each cluster. Each metacell is an aggregate of 5 single cells from the cluster. 

     B. Then, the ARACNe algorithm, developed by the Califano Lab, is used to generate Gene Regulatory Networks (GRNs) on the metacells of each cluster for each cell-type. Finally, the ARACNe graphs (a.k.a. GRNs) of each cluster are combined into one main consolidated GRN for that cell-type.

1. Once the preprocessing step is completed, the data is cached to expedite dataloading during model training. Some further QC takes place here. The data is partitioned into three datasets: __train__, __valSG__ (~17%), and __valHOG__. 

    __train (68%):__ Corresponds to your typical training dataset.

    __valSG (17%):__ Validation dataset for "Seen Graph". In other words, it is the validation dataset for cell-types that are included in the training dataset. This is similar to your typical validation dataset.

    __valHOG (15%):__ Validation dataset for "Held Out Graph". This means it consists only of cell-types the model has never seen before (not included in the training dataset). This is used to evaluate the model's zero-shot learning capabilities.
    
    _NOTE: The dataset proportions aren't exactly (70%, 15%, 15%) as we first make the SG/HOG split (85% & 15%, respectively) in `./scGraphLLM/scripts/outdir_gen.py` and we then make the train/valSG split on the previous 85% SG assignment (80% & 20%, respectively) in `./scGraphLLM/scGraphLLM/data.py`. These proportions can be adjusted in the previously mentioned files._

    After caching is complete, each metacell will have its own cached file. This means millions of files in `train` and at least hundreds of thousands in `valSG` & `valHOG`. __Please ensure that your filesystem can handle such loads.__

## Running the Pipeline

__⚠️ Preprocessing & caching must be run separately. Cache only once preprocessing is fully completed!__

Preprocessing & caching can individually be run with the `./scGraphLLM/scripts/pipeline.sh` file given the following required arguments:

`--preprocess`: Preprocess the CellxGene raw cell-type data (include: true | exclude: false).  

`--cache`: Cache the preprocessed CellxGene cell-type data (include: true | exclude: false).  

`--data-dir`: Directory to create the cache folder(s) - different folders are automatically created for each ARACNE_TOP_N_HVG (defined below) configuration (i.e. `./data`).  

`--cxg-dir`: Path to the CellxGene cell-type dataset (i.e. `./data/cellxgene/data/cell_type_all`).

`--out-dir`: Path to store the processed data. The same path as cell_type_all (above) is recommended (i.e. `./data/cellxgene/data/`).

`--rank-by-z`: When creating the expression rank bins, would you like to rank by the expression z-score of each gene? or just the raw expression value (include: true | exclude: false). 

`--aracne-top-n-hvg`: top "n" highly variable genes (HVGs) to preprocess (1024, 2048, or 4096).

`--aracne-path`: Path to the compiled ARACNe3 algorithm. We currently recommend the one provided in our repository as this version is customized for our larger datasets (i.e. `./scGraphLLM/scripts/ARACNe3_app_release`).

`--job-out-dir`: Path to the slurm_out file where all SLURM logs from preprocessing and caching steps will be stored. These files will are used to track job status and also identify which cell-types were successfully preprocessed and are ready to be cached (i.e. `./scGraphLLM/scripts/slurm_out`).

### 1. Preprocessing

__Completion time on our cluster: ~12h__

To run the preprocessing step, you will need to specify the `--preprocess`, `--data-dir`, `--cxg-dir`, `--out-dir`, `--rank-by-z` __(OPTIONAL)__, `--aracne-top-n-hvg`, `--aracne-path`, and `--job-out-dir` arguments. For instance, assuming you are in `./` and `data` & `scGraphLLM` are in `./`:


```
source pipeline.sh \
    --preprocess \
    --data-dir "./data" \
    --cxg-dir "./data/cellxgene/data/cell_type_all" \
    --out-dir "./data/cellxgene/data/" \
    --rank-by-z \
    --aracne-top-n-hvg "1024" \
    --aracne-path "./ARACNe3/build/src/app/ARACNe3_app_release" \
    --job-out-dir "./scGraphLLM/scripts/slurm_out"
```

This creates the `./data/cellxgene/data/complete_data_z_scored` directory (`./data/cellxgene/data/complete_data` if not `--rank-by-z`). Here, the filtered metacells and their respective ARACNe GRNs are stored for each cell-type.

### 2.a. Caching

__Completion time on our cluster: ~2h__

To run the caching step, you will need to specify the `--cache`, `--data-dir`, `--out-dir`, `--rank-by-z` __(OPTIONAL)__, `--aracne-top-n-hvg`, and `--job-out-dir` arguments. For instance:

```
source pipeline.sh \
    --cache \
    --data-dir "./data" \
    --out-dir "./data/cellxgene/data/" \
    --rank-by-z \
    --aracne-top-n-hvg "1024" \
    --job-out-dir "./scGraphLLM/scripts/slurm_out"
```

This creates the `./data/cxg_cache_1024` directory (`./data/cxg_cache_1024_base` if not `--rank-by-z`) with subdirectories `train`, `valSG`, `valHOG`. Each file in these subdirectories corresponds to an individual metacell in the form of a pytorch _.pt_ file. Files are saved as `<cell_type_name>_<idx>.pt`. The data is now cached; however, sometimes files get corrupted in the process and have to be removed.

### 2.b. Identify & Remove Corrupted Files

__Completion time on our cluster: ~12h__ (Can be cut drastically with parallelization)

In `./scGraphLLM/scripts/ID_corrupted`, you will find the `ID_corrupted.sh` script. You can run `source ./scGraphLLM/scripts/ID_corrupted/ID_corrupted.sh <PATH_TO_CACHE_DIR>` where, building on __2.a.__, `<PATH_TO_CACHE_DIR>` is `./data/cxg_cache_1024`. 

```
source ./scGraphLLM/scripts/ID_corrupted/ID_corrupted.sh ./data/cxg_cache_1024
```

This script will individually attempt to load each cached file in all three cache subdirectories (`train`, `valSG`, `valHOG`), all files where this fails are recorded in the `./scGraphLLM/scripts/ID_corrupted/corrupted_files` directory in a _.txt_ file corresponding to the cached directory and dataset. Check that not too many files were corrupted before continuing as you may want to re-cache.

Once the identification of corrupted files process is completed, you can delete these with the `./scGraphLLM/scripts/ID_corrupted/delete_corrupted.sh` script by specifying the path to the _.txt_ file with the corrupted _.pt_ files that you would like to delete. For example:

```
source ./scGraphLLM/scripts/ID_corrupted/delete_corrupted.sh ./scGraphLLM/scripts/ID_corrupted/corrupted_files/cxg_cache_1024_corrupt_train.txt
```

__Congrats!__ The cached data is now ready to be used. 🎉

## ⚠️ Troubleshooting ⚠️

- `ValueError: cannot specify integer "bins" when input data contains infinity` (preprocessing). Add `X = X.astype(np.float64)` under `X = _get_obs_rep(adata, layer=layer)` at the start of the `_highly_variable_genes_single_batch()` function in `/scanpy/preprocessing/_highly_variavle_genes.py`. Otherwise __inf__ values likely be encountered and interrupt the preprocessing on some cell-types.

- `ARACNe3 Error` (preprocessing: ARACNe step). Make sure you are using our pre-compiled version of ARACNe3 at `./scGraphLLM/scripts/ARACNe3_app_release` as we have specifically modified it for the use of large datasets. You may otherwise experience unexpected errors or an indefinite run-time.

<br />
<br />
<br />
<br />
<br />
<br />
<br />
<br />

# For a more granular understanding (DEPRICATED)
## Preprocess

### start_slurm_preprocess.sh

_NOTE: `start_slurm_preprocess.sh` will automatically run `array_preprocess.sh` & `preprocess_cellxgene.sh`. Please run `python checkout.py` to obtain the run's success/failure summary._

`start_slurm_preprocess.sh` gets the total number of files to process from the correct directories. Then makes a parallelized SLURM call on `array_preprocess.sh`, according to how many files there are. This parallelizes on each cell-type, and nested parallelization takes place later for each file in each cell-type. (Tissue)

_Example Command:_ `source start_slurm_preprocess.sh "initial" "1024" "false"` 

_Example Command:_ `source start_slurm_preprocess.sh "run_timed_out" "1024" "false" "786994"`

1. `RUN_TYPE`: The **RUN_TYPE** argument lets the script run the pipeline from scratch _or_ on specific cell-types that weren't successfully processed _(options: initial, run_env_fail, run_timed_out, run_failed, and run_unaccounted)_. 
1. `ARACNE_TOP_N_HVG`: There is also the **ARACNE_TOP_N_HVG** parameter which is the number of top highly variable genes on which to generate the ARACNe networks (1024, 2048, 4096, or unspecified "": the entire set of genes). 
1. `RANK_BY_Z_SCORE`: Boolean value where true will lead to the ranking of gene expression within the cell to be based on the statistical significance (z-score) of a given gene's expression relative to the overall expression of this gene across the population. The ranking bins are thereby made on the z-score "expression" of each gene.
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