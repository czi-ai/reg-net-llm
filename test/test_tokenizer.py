import unittest
import pandas as pd
import numpy as np
import torch

from torch_geometric.data import Data as torchGeomData

from scGraphLLM.tokenizer import GraphTokenizer
from scGraphLLM.vocab import GeneVocab
from scGraphLLM.network import RegulatoryNetwork


class TestGraphTokenizer(unittest.TestCase):
    def setUp(self):
        # Expression data: 3 cells × 8 genes
        self.expression = pd.DataFrame(
            [[10, 0, 0, 0, 0, 0, 2, 0],
             [0, 8, 4, 6, 2, 0, 3, 0],
             [3, 0, 1, 0, 4, 0, 0, 6]], 
            index=['Cell1', 'Cell2', 'Cell3'], 
            columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        )

        # Gene vocabulary
        self.vocab = GeneVocab(
            genes=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
            nodes=list(range(8))
        )

        # Regulatory network with a few edges
        self.network = RegulatoryNetwork(
            regulators=['A', 'B', 'C', 'G'],
            targets=['C', 'D', 'E', 'H'],
            weights=[0.9, 0.8, 0.7, 0.6],
            likelihoods=[-3.2, -2.5, -1.8, -1.1]
        )

        self.tokenizer = GraphTokenizer(
            vocab=self.vocab,
            network=self.network,
            max_seq_length=5,
            only_expressed_genes=True,
            with_edge_weights=True,
            n_bins=3,
            method="quantile"
        )

    def test_tokenizer_output_type(self):
        cell_expr = self.expression.loc['Cell2']
        data = self.tokenizer(cell_expr)

        self.assertIsInstance(data, torchGeomData)
        self.assertTrue(hasattr(data, 'x'))
        self.assertTrue(hasattr(data, 'edge_index'))
        self.assertTrue(hasattr(data, 'edge_weight'))

    def test_expression_binning_and_gene_filtering(self):
        # Cell 1
        data = self.tokenizer(self.expression.loc['Cell1'])
        genes = [self.vocab.node_to_gene[int(i)] for i in data.x[:, 0].numpy()]
        expected_genes = ["A", "G"]
        self.assertEqual(expected_genes, genes)

        # Cell 2
        data = self.tokenizer(self.expression.loc['Cell2'])
        genes = [self.vocab.node_to_gene[int(i)] for i in data.x[:, 0].numpy()]
        expected_genes = ["B", "C", "D", "E", "G"]
        self.assertEqual(expected_genes, genes)

        # Cell 3
        data = self.tokenizer(self.expression.loc['Cell3'])
        genes = [self.vocab.node_to_gene[int(i)] for i in data.x[:, 0].numpy()]
        expected_genes = ["A", "C", "E", "H"]
        self.assertEqual(expected_genes, genes)

    def test_max_seq_length_enforced(self):
        # artificially increase expression of all to check max_seq_length enforcement
        cell_expr = self.expression.loc['Cell2'] + 10
        data = self.tokenizer(cell_expr)

        self.assertLessEqual(data.num_nodes, self.tokenizer.max_seq_length)

    def test_edge_index_shape_and_weights(self):
        cell_expr = self.expression.loc['Cell2']
        data = self.tokenizer(cell_expr)

        self.assertEqual(data.edge_index.shape[0], 2)
        self.assertEqual(data.edge_weight.shape[0], data.edge_index.shape[1])

    def test_graph_is_empty_when_all_genes_zero(self):
        cell_expr = pd.Series([0]*8, index=self.expression.columns)
        data = self.tokenizer(cell_expr)

        self.assertEqual(data.x.shape[0], 0)
        self.assertEqual(data.edge_index.shape[1], 0)

    


if __name__ == '__main__':
    unittest.main()
