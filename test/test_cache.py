# test_math_utils.py
import pandas as pd
import numpy as np
import unittest

from scGraphLLM.data import run_cache

class TestRunCache(unittest.TestCase):
    def setUp(self):
        self.global_gene_to_node = pd.read_csv("/hpc/projects/group.califano/GLM/data/cellxgene_gene2index.csv")
        self.data_dir = "/hpc/projects/group.califano/GLM/data/_cellxgene/data/tmp_out/ct"
        self.cache_dir = "/hpc/projects/group.califano/GLM/scGraphLLM/test/test_cache"
        self.min_genes_per_graph = 2
        
        self.network = pd.DataFrame(
            [["ENSG00000196367", "ENSG00000259305", 0.000362889, 0.0417306, 1],
            ["ENSG00000033030", "ENSG00000162992", 0.000119578, -0.0274501, 1],
            ["ENSG00000170345", "ENSG00000137392", 0.540816, 0.436078, 1],
            ["ENSG00000131747", "ENSG00000170540", 0.0205539, 0.0688922, 1],
            ["ENSG00000131747", "ENSG00000164104", 0.0258143, 0.0739799, 1],
            ["ENSG00000131747", "ENSG00000148773", 0.000234215, 0.0558651, 3],
            ["ENSG00000131747", "ENSG00000117632", 0.0275552, 0.101647, 1],
            ["ENSG00000131747", "ENSG00000117724", 0.0378205, 0.0443026, 2],
            ["ENSG00000131747", "ENSG00000123416", 0.0368367, 0.117863, 6],
            ["ENSG00000131747", "ENSG00000137804", 0.000639771, 0.0815333, 4],
            ["ENSG00000131747", "ENSG00000139734", 4.97111e-06, 0.033628, 5],
            ["ENSG00000120738", "ENSG00000137392", 0.584849, 0.505505, 3],
            ["ENSG00000166925", "ENSG00000214491", 0.000464275, 0.0182634, 1]],
            columns=["regulator.values", "target.values", "mi.values", "scc.values", "count.values"]
        )
        
        self.expression = pd.DataFrame(
            [[10, 0, 0, 0, 0, 0, 2, 0],
            [0, 8, 4, 6, 2, 0, 3, 0], 
            [3, 0, 1, 0, 4, 0, 0, 6], 
            [2, 40, 1, 3, 3, 8, 0, 23]], 
            index=['Cell1', 'Cell2', 'Cell3', 'Cell4'], 
            columns=['ENSG00000170345', 'ENSG00000196367', 'ENSG00000162992', 'ENSG00000259305', 'ENSG00000170540', 'ENSG00000033030', 'ENSG00000131747', 'ENSG00000117724']
        )


    # Check consistent formatting (CxG)
    def test_format_cxg(self):
        skipped, ncells = run_cache(
            network=self.network, 
            expression=self.expression, 
            global_gene_to_node=self.global_gene_to_node, 
            cache_dir=self.cache_dir + "/format", 
            overwrite=False, 
            msplit="valHOG", 
            valsg_split_ratio=0.2, 
            cell_type="TEST",
            min_genes_per_graph=self.min_genes_per_graph
        )
        
        # self.assertTrue(True)


if __name__ == '__main__':
    unittest.main()
    # self.data_dir_cxg = "/hpc/projects/group.califano/GLM/data/_cellxgene/data/tmp_out/ct"
    # self.data_dir_rep = "/hpc/projects/group.califano/GLM/data/_replogle/data/tmp_out/K562"
    
    
    # # network = pd.read_csv(self.data_dir_rep + "/aracne/consolidated-net_defaultid.tsv", sep = "\t")
    # # expression = ad.read_h5ad(self.data_dir_rep + "/cells.h5ad", backed="r")