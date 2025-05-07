import pandas as pd
import numpy as np
import anndata as ad
import torch
import os
from pathlib import Path
import unittest

from scGraphLLM.data import run_cache, run_cache_perturbation

# class TestRunCache(unittest.TestCase):
#     def setUp(self):
#         global_gene_to_node = pd.read_csv("/hpc/projects/group.califano/GLM/data/cellxgene_gene2index.csv")
#         self.global_gene_to_node = {row.gene_name:row.idx for _,row in global_gene_to_node.iterrows()}
#         self.data_dir = "/hpc/projects/group.califano/GLM/data/_cellxgene/data/tmp_out/ct"
#         self.cache_dir = "/hpc/mydata/leo.dupire/GLM/scGraphLLM/test/test_cache_cxg"
#         self.min_genes_per_graph = 2
        
#         self.network = pd.DataFrame(
#             [["ENSG00000000003", "ENSG00000000938", 0.000362889, 0.0417306, 1],
#             ["ENSG00000000419", "ENSG00000000971", 0.000119578, -0.0274501, 1],
#             ["ENSG00000000460", "ENSG00000001036", 0.540816, 0.436078, 1],
#             ["ENSG00000000457", "ENSG00000001084", 0.0205539, 0.0688922, 1],
#             ["ENSG00000000005", "ENSG00000001167", 0.0258143, 0.0739799, 1],
#             ["ENSG00000000005", "ENSG00000001460", 0.000234215, 0.0558651, 3],
#             ["ENSG00000000005", "ENSG00000001461", 0.0275552, 0.101647, 1],
#             ["ENSG00000000005", "ENSG00000001497", 0.0378205, 0.0443026, 2],
#             ["ENSG00000000005", "ENSG00000001561", 0.0368367, 0.117863, 6],
#             ["ENSG00000000005", "ENSG00000001617", 0.000639771, 0.0815333, 4]],
#             columns=["regulator.values", "target.values", "mi.values", "scc.values", "count.values"]
#         )

#         self.expression = pd.DataFrame(
#             [[10, 0, 0, 0, 0, 0, 2, 0],
#             [0, 8, 4, 6, 2, 0, 3, 0], 
#             [3, 0, 1, 0, 4, 0, 0, 6], 
#             [2, 40, 1, 3, 3, 8, 0, 23]], 
#             index=['Cell1', 'Cell2', 'Cell3', 'Cell4'], 
#             columns=['ENSG00000000460', 'ENSG00000000005', 'ENSG00000001084', 'ENSG00000001497', 'ENSG00000000938', 'ENSG00000000457', 'ENSG00000001460', 'ENSG00000001036']
#         )


#     # Check consistent formatting (CxG)
#     def test_format(self):
#         folder = "format"
#         Path(self.cache_dir + "/" + folder).mkdir(parents=True, exist_ok=True)
        
#         skipped, ncells = run_cache(
#             network=self.network, 
#             expression=self.expression, 
#             global_gene_to_node=self.global_gene_to_node, 
#             cache_dir=self.cache_dir, 
#             overwrite=True, 
#             msplit=folder, 
#             valsg_split_ratio=0.2, 
#             cell_type="CXG",
#             min_genes_per_graph=self.min_genes_per_graph
#         )
        
#         for f in os.listdir(self.cache_dir + "/" + folder):
#             file = torch.load(self.cache_dir + "/" + folder + "/" + f)
        
#         # self.assertTrue(True)


class TestRunCachePERTURBED(unittest.TestCase):
    def setUp(self):
        global_gene_to_node = pd.read_csv("/hpc/projects/group.califano/GLM/data/cellxgene_gene2index.csv")
        self.global_gene_to_node = {row.gene_name:row.idx for _,row in global_gene_to_node.iterrows()}
        self.data_dir = "/hpc/projects/group.califano/GLM/data/_cellxgene/data/tmp_out/ct"
        self.cache_dir = "/hpc/mydata/leo.dupire/GLM/scGraphLLM/test/test_cache_rep"
        self.min_genes_per_graph = 2
        
        self.network = pd.DataFrame(
            [["ENSG00000000003", "ENSG00000000938", 0.000362889, 0.0417306, 1],
            ["ENSG00000000419", "ENSG00000000971", 0.000119578, -0.0274501, 1],
            ["ENSG00000000460", "ENSG00000001036", 0.540816, 0.436078, 1],
            ["ENSG00000000457", "ENSG00000001084", 0.0205539, 0.0688922, 1],
            ["ENSG00000000005", "ENSG00000001167", 0.0258143, 0.0739799, 1],
            ["ENSG00000000005", "ENSG00000001460", 0.000234215, 0.0558651, 3],
            ["ENSG00000000005", "ENSG00000001461", 0.0275552, 0.101647, 1],
            ["ENSG00000000005", "ENSG00000001497", 0.0378205, 0.0443026, 2],
            ["ENSG00000000005", "ENSG00000001561", 0.0368367, 0.117863, 6],
            ["ENSG00000000005", "ENSG00000001617", 0.000639771, 0.0815333, 4]],
            columns=["regulator.values", "target.values", "mi.values", "scc.values", "count.values"]
        )

        X = np.array([[10, 0, 0, 0, 0, 0, 2, 0],
            [0, 8, 4, 6, 2, 0, 3, 0], 
            [3, 0, 1, 0, 4, 0, 0, 6], 
            [2, 40, 1, 3, 3, 8, 0, 23]])
        
        obs = pd.DataFrame(index=['Cell1', 'Cell2', 'Cell3', 'Cell4'])
        obs["gene_id"] = pd.Series(["ENSG00000000419", "non-targeting", "ENSG00000000460", "ENSG00000000005"], index=obs.index)
        var = pd.DataFrame(index=['ENSG00000000460', 'ENSG00000000005', 'ENSG00000001084', 'ENSG00000001497', 'ENSG00000000938', 'ENSG00000000457', 'ENSG00000001460', 'ENSG00000001036'])
        
        self.expression = ad.AnnData(X=X, obs=obs, var=var)


    # Check consistent formatting (CxG)
    def test_format(self):
        folder = "format"
        path = self.cache_dir + "/" + folder
        Path(path + "/control").mkdir(parents=True, exist_ok=True)
        Path(path + "/perturbed").mkdir(parents=True, exist_ok=True)
        
        skipped, ncells = run_cache_perturbation(
            network=self.network, 
            expression=self.expression, 
            global_gene_to_node=self.global_gene_to_node, 
            cache_dir=self.cache_dir, 
            overwrite=True, 
            msplit=folder, 
            valsg_split_ratio=0.2, 
            cell_type="REP",
            min_genes_per_graph=self.min_genes_per_graph,
            test_mode=True
        )
        
        for f in os.listdir(path + "/control"):
            file = torch.load(path + "/control/" + f)
        
        for f in os.listdir(path + "/perturbed"):
            file = torch.load(path + "/perturbed/" + f)
        
        # self.assertTrue(True)

if __name__ == '__main__':
    unittest.main()
    # self.data_dir_cxg = "/hpc/projects/group.califano/GLM/data/_cellxgene/data/tmp_out/ct"
    # self.data_dir_rep = "/hpc/projects/group.califano/GLM/data/_replogle/data/tmp_out/K562"
    
    
    # # network = pd.read_csv(self.data_dir_rep + "/aracne/consolidated-net_defaultid.tsv", sep = "\t")
    # # expression = ad.read_h5ad(self.data_dir_rep + "/cells.h5ad", backed="r")