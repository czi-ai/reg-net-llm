from os.path import join

from scGraphLLM.config import Config


CELL_TYPES = [
    'Macro_C1QC',
    'Mono_CD16',
    'Mono_CD14',
    'cDC2_CD1C',
    'Macro_LYVE1',
    # 'Macro_SPP1',
    'Macro_NLRP3',
    # 'Macro_GPNMB',
    # 'Macro_INHBA',
    # 'Macro_IL1B',
    # 'cDC2_CXCR4hi',
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

PANCREAS_CELL_TYPES = [
    'acinar',
    'delta',
    'beta',
    'PSC',
    'alpha',
    'ductal',
    # 'epsilon',
    'PP',
    # 'endothelial',
    # 'macrophage',
    # 'schwann',
    # 'mast',
    # 't_cell',
    # 'MHC class II'
]

IMMUNE_CELL_TYPES = [
    "cd14_monocytes", # 'CD14+ Monocytes'
    "cd20_b_cells", # 'CD20 B cells'
    "cd8_t_cells", # 'CD8+ T cells'
    "nkt_cells", # NKT cells
    # Val
    "erythrocytes", # 'Erythrocytes'
    "cd16_monocytes", # 'CD16+ Monocytes'
    # Test
    "cd4_t_cells", # 'CD4+ T cells'
    "monocyte-derived_dendritic_cells", # 'Monocyte-derived dendritic cells'
    "nk_cells" # 'NK cells'
]

human_immune_cell_names = [
    'CD4+ T cells',
    'CD14+ Monocytes',
    'CD20+ B cells',
    'NKT cells',
    'NK cells',
    'CD8+ T cells',
    'Erythrocytes',
    'Monocyte-derived dendritic cells',
    'CD16+ Monocytes',
    # 'HSPCs',
    # 'Erythroid progenitors',
    # 'Plasmacytoid dendritic cells',
    # 'Monocyte progenitors',
    # 'Megakaryocyte progenitors',
    # 'CD10+ B cells',
    # 'Plasma cells'
]

NETWORK_SETS = {
    "human_immune_metacells_networks": {
        'CD4+ T cells': "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/cd4_t_cells/aracne_4096/consolidated-net_defaultid.tsv",
        'CD14+ Monocytes': "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/cd14_monocytes/aracne_4096/consolidated-net_defaultid.tsv",
        'CD20+ B cells': "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/cd20_b_cells/aracne_4096/consolidated-net_defaultid.tsv",
        'NKT cells': "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/nkt_cells/aracne_4096/consolidated-net_defaultid.tsv",
        'NK cells': "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/nk_cells/aracne_4096/consolidated-net_defaultid.tsv",
        'CD8+ T cells': "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/cd8_t_cells/aracne_4096/consolidated-net_defaultid.tsv",
        'Erythrocytes': "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/erythrocytes/aracne_4096/consolidated-net_defaultid.tsv",
        'Monocyte-derived dendritic cells': "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/monocyte-derived_dendritic_cells/aracne_4096/consolidated-net_defaultid.tsv",
        'CD16+ Monocytes': "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/cd16_monocytes/aracne_4096/consolidated-net_defaultid.tsv"
    },
    "t_cell_network": {
        'CD4+ T cells':    "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/cd4_t_cells/aracne_4096/consolidated-net_defaultid.tsv",
        'CD14+ Monocytes': "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/cd4_t_cells/aracne_4096/consolidated-net_defaultid.tsv",
        'CD20+ B cells':   "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/cd4_t_cells/aracne_4096/consolidated-net_defaultid.tsv",
        'NKT cells':       "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/cd4_t_cells/aracne_4096/consolidated-net_defaultid.tsv",
        'NK cells':        "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/cd4_t_cells/aracne_4096/consolidated-net_defaultid.tsv",
        'CD8+ T cells':    "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/cd4_t_cells/aracne_4096/consolidated-net_defaultid.tsv",
        'Erythrocytes':    "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/cd4_t_cells/aracne_4096/consolidated-net_defaultid.tsv",
        'Monocyte-derived dendritic cells': "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/cd4_t_cells/aracne_4096/consolidated-net_defaultid.tsv",
        'CD16+ Monocytes': "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/cd4_t_cells/aracne_4096/consolidated-net_defaultid.tsv",
    },
    "mye_prune_networks":{
        'Macro_C1QC': '/hpc/mydata/rowan.cassius/data/scGPT/mye/all/cell_type/Macro_C1QC/aracne_metacells_pruned/consolidated-pruned-no-elim-net_defaultid.tsv',
        'Mono_CD16': '/hpc/mydata/rowan.cassius/data/scGPT/mye/all/cell_type/Mono_CD16/aracne_metacells_pruned/consolidated-pruned-no-elim-net_defaultid.tsv',
        'Mono_CD14': '/hpc/mydata/rowan.cassius/data/scGPT/mye/all/cell_type/Mono_CD14/aracne_metacells_pruned/consolidated-pruned-no-elim-net_defaultid.tsv',
        'cDC2_CD1C': '/hpc/mydata/rowan.cassius/data/scGPT/mye/all/cell_type/cDC2_CD1C/aracne_metacells_pruned/consolidated-pruned-no-elim-net_defaultid.tsv',
        'Macro_LYVE1': '/hpc/mydata/rowan.cassius/data/scGPT/mye/all/cell_type/Macro_LYVE1/aracne_metacells_pruned/consolidated-pruned-no-elim-net_defaultid.tsv',
        'Macro_NLRP3': '/hpc/mydata/rowan.cassius/data/scGPT/mye/all/cell_type/Macro_NLRP3/aracne_metacells_pruned/consolidated-pruned-no-elim-net_defaultid.tsv'
    },
    "mye_networks": {
        'Macro_C1QC': '/hpc/mydata/rowan.cassius/data/scGPT/mye/all/cell_type/Macro_C1QC/aracne_metacells/consolidated-net_defaultid.tsv',
        'Mono_CD16': '/hpc/mydata/rowan.cassius/data/scGPT/mye/all/cell_type/Mono_CD16/aracne_metacells/consolidated-net_defaultid.tsv',
        'Mono_CD14': '/hpc/mydata/rowan.cassius/data/scGPT/mye/all/cell_type/Mono_CD14/aracne_metacells/consolidated-net_defaultid.tsv',
        'cDC2_CD1C': '/hpc/mydata/rowan.cassius/data/scGPT/mye/all/cell_type/cDC2_CD1C/aracne_metacells/consolidated-net_defaultid.tsv',
        'Macro_LYVE1': '/hpc/mydata/rowan.cassius/data/scGPT/mye/all/cell_type/Macro_LYVE1/aracne_metacells/consolidated-net_defaultid.tsv',
        'Macro_NLRP3': '/hpc/mydata/rowan.cassius/data/scGPT/mye/all/cell_type/Macro_NLRP3/aracne_metacells/consolidated-net_defaultid.tsv'
    }


}



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
    "scglm_graph_metacells": [
        f"/hpc/mydata/rowan.cassius/data/scGPT/mye/all/cell_type/{cell_type}/embeddings/scglm/cls_3L_12000_steps_MLM_001_seq_len_2048_graph_metacells/cached_embeddings"
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
    ],

    "mye_scglm_cls_3L_12000_steps_MLM_001_infer_network_pruned": [
        "/hpc/mydata/rowan.cassius/data/scGPT/mye/embeddings/scglm/aracne__cls_3L_12000_steps_MLM_001_infer_network_pruned/cached_embeddings"
    ],
    # new scGLM model
    "mye_scglm_cls_3L_12000_steps_MLM_001_seq_len_2048": [
        f"/hpc/mydata/rowan.cassius/data/scGPT/mye/all/cell_type/{cell_type}/embeddings/scglm/cls_3L_12000_steps_MLM_001_seq_len_2048/cached_embeddings"
        for cell_type in CELL_TYPES
    ],
    "mye_scglm_cls_3L_12000_steps_MLM_001_seq_len_2048_pruned_graph_002": [
        f"/hpc/mydata/rowan.cassius/data/scGPT/mye/all/cell_type/{cell_type}/embeddings/scglm/cls_3L_12000_steps_MLM_001_seq_len_2048_graph_pruned_002/cached_embeddings"
        for cell_type in CELL_TYPES
    ],
    "mye_scglm_cls_3L_12000_steps_MLM_001_seq_len_2048_pruned_graph_50": [
        f"/hpc/mydata/rowan.cassius/data/scGPT/mye/all/cell_type/{cell_type}/embeddings/scglm/cls_3L_12000_steps_MLM_001_seq_len_2048_graph_pruned_50/cached_embeddings"
        for cell_type in CELL_TYPES
    ],

    "mye_scglm_cls_3L_12000_steps_MLM_001_seq_len_2048_mask_0.15": [
        f"/hpc/mydata/rowan.cassius/data/scGPT/mye/all/cell_type/{cell_type}/embeddings/scglm/cls_3L_12000_steps_MLM_001_seq_len_2048_mask_0.15/cached_embeddings"
        for cell_type in CELL_TYPES
    ],
    "mye_scglm_cls_3L_12000_steps_MLM_001_infer_networks_soft": [
        "/hpc/mydata/rowan.cassius/data/scGPT/mye/embeddings/scglm/aracne__cls_3L_12000_steps_MLM_001_infer_networks_soft/cached_embeddings"
    ],
    "mye_scglm_cls_3L_12000_steps_MLM_001_infer_networks_soft_unpruned": [
        "/hpc/mydata/rowan.cassius/data/scGPT/mye/embeddings/scglm/aracne_4096_cls_3L_12000_steps_MLM_001_infer_networks_soft_unpruned/cached_embeddings"
    ],
    # even newer scGLM model
    "mye_scglm_cls_3L_12000_steps_MLM_002_seq_len_2048_pruned_graph_50": [
        f"/hpc/mydata/rowan.cassius/data/scGPT/mye/all/cell_type/{cell_type}/embeddings/scglm/cls_3L_12000_steps_MLM_002_seq_len_2048_graph_pruned_50/cached_embeddings"
        for cell_type in CELL_TYPES
    ],

    
    # PANCREAS
    "pancreas_scglm_cls_3L_12000_steps_MLM_001": [
        f"/hpc/mydata/rowan.cassius/data/scGPT/annotation_pancreas/cell_type/{cell_type}/embeddings/scglm/cls_3L_12000_steps_MLM_001_seq_len_2048_graph_metacells/cached_embeddings"
        for cell_type in PANCREAS_CELL_TYPES
    ],
    "pancreas_scgpt_seq_len_2048": [
        f"/hpc/mydata/rowan.cassius/data/scGPT/annotation_pancreas/cell_type/{cell_type}/embeddings/scgpt/seq_len_2048/cached_embeddings"
        for cell_type in PANCREAS_CELL_TYPES
    ],
    "pancreas_scglm_cls_3L_12000_steps_MLM_001_pruned_graph": [
        f"/hpc/mydata/rowan.cassius/data/scGPT/annotation_pancreas/cell_type/{cell_type}/embeddings/scglm/cls_3L_12000_steps_MLM_001_seq_len_2048_pruned_graph/cached_embeddings"
        for cell_type in PANCREAS_CELL_TYPES
    ],
    "pancreas_scglm_cls_3L_12000_steps_MLM_001_pruned_graph_002": [
        f"/hpc/mydata/rowan.cassius/data/scGPT/annotation_pancreas/cell_type/{cell_type}/embeddings/scglm/cls_3L_12000_steps_MLM_001_seq_len_2048_pruned_graph_002/cached_embeddings"
        for cell_type in PANCREAS_CELL_TYPES
    ],

    # Adamson
    "adamson_scglm_cls_3L_12000_steps_MLM_001":[
        "/hpc/projects/group.califano/GLM/data/adamson/embeddings/scglm/scglm_cls_3L_12000_steps_MLM_001/cached_embeddings"
    ],
    # Adamson
    "adamson_scglm_cls_3L_12000_steps_MLM_001_pruned_graph_100":[
        "/hpc/projects/group.califano/GLM/data/adamson/embeddings/scglm/scglm_cls_3L_12000_steps_MLM_001_pruned_graph_100/cached_embeddings"
    ],

    # HUMAN IMMUNE
    # scGraphLLM
    "human_immune_scglm_cls_3L_12000_steps_MLM_001_correct_expression": [
        f"/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/{cell_type}/embeddings/scglm/aracne_4096_cls_3L_12000_steps_MLM_001_correct_expression/cached_embeddings"
        for cell_type in IMMUNE_CELL_TYPES
    ],
    "human_immune_scglm_cls_3L_12000_steps_MLM_001": [
        f"/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/{cell_type}/embeddings/scglm/aracne_4096_cls_3L_12000_steps_MLM_001/cached_embeddings"
        for cell_type in IMMUNE_CELL_TYPES
    ],
    "human_immune_scglm_cls_3L_12000_steps_MLM_001_repro": [
        f"/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/{cell_type}/embeddings/scglm/aracne_4096_cls_3L_12000_steps_MLM_001_repro/cached_embeddings"
        for cell_type in IMMUNE_CELL_TYPES
    ],
    "human_immune_scglm_cls_3L_12000_steps_MLM_001_integrated_network": [
        f"/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/{cell_type}/embeddings/scglm/aracne_4096_cls_3L_12000_steps_MLM_001_integrated_network/cached_embeddings"
        for cell_type in IMMUNE_CELL_TYPES
    ],
    "human_immune_scglm_cls_3L_12000_steps_MLM_001_infer_network": [
        "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/embeddings/scglm/aracne_4096_cls_3L_12000_steps_MLM_001_infer_network/cached_embeddings"
    ],
    # latest model
    "human_immune_scglm_cls_3L_12000_steps_MLM_002_infer_network": [
        "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/embeddings/scglm/aracne_4096_cls_3L_12000_steps_MLM_002_infer_network/cached_embeddings"
    ],

    "human_immune_scglm_cls_3L_12000_steps_MLM_001_infer_network_sample": [
        "/hpc/mydata/rowan.cassius/data/scGPT/human_immune/embeddings/test_infer_network/cached_embeddings"
    ],
    "human_immune_scglm_cls_3L_12000_steps_MLM_001_edge_mask_0.15": [
        f"/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/{cell_type}/embeddings/scglm/aracne_4096_cls_3L_12000_steps_MLM_001_edge_mask_0.15/cached_embeddings"
        for cell_type in IMMUNE_CELL_TYPES
    ],

    # Geneformer
    "human_immune_geneformer_seq_len_2048": [
        f"/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/{cell_type}/embeddings/geneformer/aracne_4096_seq_len_2048/cached_embeddings"
        for cell_type in IMMUNE_CELL_TYPES
    ],
    # scGPT
    "human_immune_scgpt_seq_len_2048": [
        f"/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/{cell_type}/embeddings/scgpt/aracne_4096_seq_len_2048/cached_embeddings"
        for cell_type in IMMUNE_CELL_TYPES
    ],
    # scFoundation




    #### BRAIN #####

    "scglm_brain": [
        "/hpc/mydata/mingxuan.zhang/data/brain/embeddings/GREmLN/aracne__inferred_network_not_pruned/cached_embeddings"
    ],
    "scgpt_brain": [
        "/hpc/mydata/mingxuan.zhang/data/brain/embeddings/scgpt/seq_len_2048/cached_embeddings"
    ],
    "gf_brain": [
        "/hpc/mydata/mingxuan.zhang/data/brain/embeddings/gfm/seq_len_2048/cached_embeddings"
    ],


    # Held out graph
    "gf_hog": [
        "/hpc/projects/group.califano/GLM/data/HOG_benchmarking/embeddings/geneformer/seq_len_2048/cached_embeddings"
    ],
    "scgpt_hog":[
        "/hpc/projects/group.califano/GLM/data/HOG_benchmarking/embeddings/scgpt/seq_len_2048/cached_embeddings"
    ],
    "scglm_hog":[
        f"/hpc/projects/group.califano/GLM/data/HOG_benchmarking/embeddings/scglm_PARALLEL_{i}/aracne__cls_3L_12000_steps_MLM_001_infer_network_pruned/cached_embeddings"
        for i in range(10)
    ],

    # scf late to the party
    "scf_human_immune": [
        f"/hpc/mydata/rowan.cassius/data/scGPT/human_immune/cell_type/{cell_type}/embeddings/scfoundation/aracne_4096_seq_len_2048/cached_embeddings"
        for cell_type in IMMUNE_CELL_TYPES
    ],
    "scf_mye":[
        f"/hpc/mydata/rowan.cassius/data/scGPT/mye/all/cell_type/{cell_type}/embeddings/scfoundation/seq_len_2048/cached_embeddings"
        for cell_type in CELL_TYPES
    ]

})

SPLIT_CONFIGS = {
    "train_test_set": Config({
        "metadata_config": ("set", (None, None, ["test"])),
        "ratio_config": (0.9, 0.1, None)
    }),
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
    }),
    "human_immune": Config({
        "metadata_config": ("final_annotation", (None, ['Erythrocytes', 'CD16+ Monocytes'], ['CD4+ T cells', 'Monocyte-derived dendritic cells', 'NK cells'])),
        "ratio_config": (1.0, None, None)
    }),
    "adamson": Config({
        "metadata_config": None,
        "ratio_config": (0.7, 0.1, 0.2),
        "filter_config": {
            "condition": {
                "values": [
                    'CREB1+ctrl',
                    'ZNF326+ctrl',
                    'BHLHE40+ctrl',
                    'DDIT3+ctrl',
                    'HARS+ctrl',
                    'SLMO2+ctrl',
                    'IARS2+ctrl',
                    'QARS+ctrl',
                    'NEDD8+ctrl',
                    'HSPA9+ctrl',
                    'SARS+ctrl',
                    'HSPA5+ctrl',
                    'PSMD4+ctrl',
                    'EIF2S1+ctrl',
                    'TIMM44+ctrl',
                    'AARS+ctrl',
                    'PPWD1+ctrl',
                    'COPZ1+ctrl',
                    'EIF2B3+ctrl',
                    'CARS+ctrl'
                ], 
                "mode": "include"
            }
        }
    })
}



