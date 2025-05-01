from os.path import join

from scGraphLLM.config import Config


CELL_TYPES = [
    'Macro_C1QC',
    'Mono_CD16',
    'Mono_CD14',
    'cDC2_CD1C',
    'Macro_LYVE1',
    'Macro_SPP1',
    'Macro_NLRP3',
    'Macro_GPNMB',
    'Macro_INHBA',
    'Macro_IL1B',
    'cDC2_CXCR4hi',
    # 'cDC2_CD1A', # exclude due to insufficient edge count (< 500)
    # 'Macro_FN1',
    # 'pDC_LILRA4',
    # 'cDC2_IL1B',
    # 'cDC2_FCN1',
    # 'Macro_ISG15',
    # 'cDC3_LAMP3',
    # 'cDC2_CXCL9',
    # 'cDC1_CLEC9A',
    # 'cDC2_ISG15'
]

EMBEDDING_DATASETS = ({
    "gf_debug": [
        f"/hpc/mydata/rowan.cassius/data/scGPT/mye/all/cell_type/{cell_type}/embeddings/geneformer/test_cache/cached_embeddings"
        for cell_type in ("Macro_NLRP3", "Macro_LYVE1", "cDC2_CXCR4hi")
    ],
    "gf": [
        f"/hpc/mydata/rowan.cassius/data/scGPT/mye/all/cell_type/{cell_type}/embeddings/geneformer/cached_embeddings"
        for cell_type in CELL_TYPES
    ],
    "scglm": [
        f"/hpc/mydata/rowan.cassius/data/scGPT/mye/all/cell_type/{cell_type}/embeddings/scglm/cached_embeddings"
        for cell_type in CELL_TYPES
    ],

    "gf_seq_len_2048": [
        f"/hpc/mydata/rowan.cassius/data/scGPT/mye/all/cell_type/{cell_type}/embeddings/geneformer/seq_len_2048/cached_embeddings"
        for cell_type in CELL_TYPES
    ],
    "scglm_seq_len_2048": [
        f"/hpc/mydata/rowan.cassius/data/scGPT/mye/all/cell_type/{cell_type}/embeddings/scglm/seq_len_2048/cached_embeddings"
        for cell_type in CELL_TYPES
    ],
    "scgpt_seq_len_2048": [
        f"/hpc/mydata/rowan.cassius/data/scGPT/mye/all/cell_type/{cell_type}/embeddings/scgpt/seq_len_2048/cached_embeddings"
        for cell_type in CELL_TYPES
    ],

    "gf_seq_len_512": [
        f"/hpc/mydata/rowan.cassius/data/scGPT/mye/all/cell_type/{cell_type}/embeddings/geneformer/seq_len_512/cached_embeddings"
        for cell_type in CELL_TYPES
    ],
    "scglm_seq_len_512": [
        f"/hpc/mydata/rowan.cassius/data/scGPT/mye/all/cell_type/{cell_type}/embeddings/scglm/seq_len_512/cached_embeddings"
        for cell_type in CELL_TYPES
    ],
    "scgpt_seq_len_512":[
        f"/hpc/mydata/rowan.cassius/data/scGPT/mye/all/cell_type/{cell_type}/embeddings/scgpt/seq_len_512/cached_embeddings"
        for cell_type in CELL_TYPES
    ]
})

SPLIT_CONFIGS = {
    "random": Config({
        "metadata_config": None,
        "ratio_config": (0.7, 0.1, 0.2)
    }),
    "debug": Config({
        "metadata_config": ("cancer_type", (None, None, ['MYE', 'UCEC'])),
        "ratio_config": (0.9, 0.1, None)
    }),
    "mye": Config({
        "metadata_config": ("cancer_type", (None, None, ['MYE', 'OV-FTC', 'ESCA'])),
        "ratio_config": (0.85, 0.15, None)
    }),
    "mye_cancer_type": Config({
        "metadata_config": ("cancer_type", (None, ['THCA', 'LYM'], ['MYE', 'OV-FTC', 'ESCA'])),
        "ratio_config": (1.0, None, None)
    })
}
