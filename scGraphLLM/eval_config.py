from os.path import join

from scGraphLLM.config import Config

# Human Immune Cell Dataset
# train_cell_types = [
#     "cd14_monocytes",
#     "cd20_b_cells",
#     "cd8_t_cells",
#     "nkt_cells"
# ]
# val_cell_types = [
#     "erythrocytes",
#     "cd16_monocytes"
# ]
# test_cell_types = [
#     "cd4_t_cells",
#     "monocyte-derived_dendritic_cells",
#     "nk_cells"
# ]

# if args.model == "scglm":
#     if args.task == "link":
#         embedding_path_format = "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/{}/embeddings/scglm/embedding.npz"
#     elif args.task == "mgm":
#         embedding_path_format = "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/{}/embeddings/scglm/aracne_4096_masked_0.45/embedding.npz"
# elif args.model == "scgpt":
#     embedding_path_format = "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/{}/embeddings/scgpt/embedding.npz"
# elif args.model == "scf":
#     embedding_path_format = "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/{}/embeddings/scfoundation/aracne_4096/embedding.npz"
# elif args.model == "gf":
#     embedding_path_format = "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/{}/embeddings/geneformer/aracne_4096/embedding.npz"




### Myeloid Dataset for Cell Annotation

MYE_REF_DIR = "/hpc/mydata/rowan.cassius/data/scGPT/mye/ref/cell_type"
MYE_QUERY_DIR = "/hpc/mydata/rowan.cassius/data/scGPT/mye/query/cell_type"


EMBEDDING_DATASETS = ({
    "debug": [
        f"/hpc/mydata/rowan.cassius/data/scGPT/mye/all/cell_type/{cell_type}/embeddings/geneformer/test_cache/cached_embeddings"
        for cell_type in ("Macro_NLRP3", "Macro_LYVE1", "cDC2_CXCR4hi")
    ]
})

SPLIT_CONFIGS = {
    "random": Config({
        "metadata_config": None,
        "ratio_config": (0.7, 0.1, 0.2)
    }),
    "debug": Config({
        "metadata_config": ("cancer_type", (None, None, ['MYE', 'UCEC', 'cDC2'])),
        "ratio_config": (0.9, 0.1, None)
    }),
    "mye": Config({
        "metadata_config": ("cancer_type", (None, None, ['MYE', 'UCEC', 'cDC2'])),
        "ratio_config": (0.8, 0.2, None)
    })
}
