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
            [[10, 0, 0, 0, 0, 7, 2, 0],
             [0, 8, 4, 6, 2, 0, 3, 0],
             [3, 0, 1, 0, 4, 6, 0, 6]], 
            index=['Cell1', 'Cell2', 'Cell3'], 
            columns=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
        )
        
        # Gene vocabulary
        self.vocab = GeneVocab(
            genes=['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'],
            nodes=list(range(8))
        )

        # Regulatory network with edges (A,C), (B,D), (C,E), (G,H), (G,A), (H,E), (H,C)
        self.network = RegulatoryNetwork(
            regulators=['A', 'B', 'C', 'G', 'G', 'H', 'H'],
            targets=   ['C', 'D', 'E', 'H', 'A', 'E', 'C'],
            weights=   [0.9, 0.8, 0.7, 0.6, 1.1, 0.2, 0.6],
            likelihoods=[-3.2, -2.5, -1.8, -1.1, -2.4, -1.3, -1.9]
        )

    def test_gene_filtering(self):
        tokenizer = GraphTokenizer(
            vocab=self.vocab,
            network=self.network,
            max_seq_length=5,
            only_expressed_genes=True,
            with_edge_weights=True,
            n_bins=3,
            method="quantile"
        )

        # Cell 1, edges = (G,A)
        data = tokenizer(self.expression.loc['Cell1'])
        genes = [tokenizer.vocab.node_to_gene[int(i)] for i in data.x.numpy()[:, 0]]
        edges = data.edge_index.numpy()
        expected_genes = ["A", "G"]
        expected_edges = [[1], [0]]

        self.assertEqual(expected_genes, genes)
        self.assertTrue(np.array_equal(expected_edges, edges))

        # Cell 2, edges = (B,D), (C,E)
        data = tokenizer(self.expression.loc['Cell2'])
        genes = [tokenizer.vocab.node_to_gene[int(i)] for i in data.x[:, 0].numpy()]
        edges = data.edge_index.numpy()
        expected_genes = ["B", "C", "D", "E", "G"]
        expected_edges = [[0, 1], [2, 3]]

        self.assertEqual(expected_genes, genes)
        self.assertTrue(np.array_equal(expected_edges, edges))

        # Cell 3, edges = (A,C), (C,E), (H,E), (H,C)
        data = tokenizer(self.expression.loc['Cell3'])
        genes = [tokenizer.vocab.node_to_gene[int(i)] for i in data.x[:, 0].numpy()]
        edges = data.edge_index.numpy()
        expected_genes = ["A", "C", "E", "H"]
        expected_edges = [[0, 1, 3, 3], [1, 2, 2, 1]]

        self.assertEqual(expected_genes, genes)
        self.assertTrue(np.array_equal(expected_edges, edges))

    def test_only_expressed_plus_neighbors(self):
        tokenizer = GraphTokenizer(
            vocab=self.vocab,
            network=self.network,
            max_seq_length=6,
            only_expressed_plus_neighbors=True,
            with_edge_weights=True,
            n_bins=10,
            method="quantile"
        )

        # Cell 1, edges = (A,C), (G,H), (G,A), (H,C) 
        data = tokenizer(self.expression.loc['Cell1'])
        genes = [tokenizer.vocab.node_to_gene[int(i)] for i in data.x.numpy()[:, 0]]
        edges = data.edge_index.numpy()
        expected_genes = ["A", "C", "G", "H"] 
        expected_edges = [[0,2,2,3], [1,3,0,1]]

        self.assertEqual(expected_genes, genes)
        self.assertTrue(np.array_equal(expected_edges, edges))

        # Cell 2, edges = (A,C), (B,D), (C,E), (G,A)
        data = tokenizer(self.expression.loc['Cell2'])
        genes = [tokenizer.vocab.node_to_gene[int(i)] for i in data.x[:, 0].numpy()]
        edges = data.edge_index.numpy()
        # max_seq_length of 6 only allows one non_expressed neighbor (A)
        expected_genes = ["B", "D", "C", "G", "E", "A"]
        expected_edges = [[5,0,2,3], [2,1,4,5]]

        self.assertEqual(expected_genes, genes)
        self.assertTrue(np.array_equal(expected_edges, edges))

        # Cell 3, edges = (A,C), (C,E), (G,H), (G,A), (H,E), (H,C)
        data = tokenizer(self.expression.loc['Cell3'])
        genes = [tokenizer.vocab.node_to_gene[int(i)] for i in data.x[:, 0].numpy()]
        edges = data.edge_index.numpy()
        expected_genes = ["A", "C", "E", "G", "H"]
        expected_edges = [[0, 1, 3, 3, 4, 4], [1, 2, 4, 0, 2, 1]]

        self.assertEqual(expected_genes, genes)
        self.assertTrue(np.array_equal(expected_edges, edges))

    def test_filtering_unknown_genes(self):
        # Create vocabulary with genes E and F removes
        reduced_vocab = GeneVocab(
            genes=['A', 'B', 'C', 'D', 'G', 'H'],
            nodes=list(range(6))
        )
        tokenizer = GraphTokenizer(
            vocab=reduced_vocab,
            network=self.network,
            max_seq_length=5,
            only_expressed_genes=True,
            with_edge_weights=True,
            n_bins=3,
            method="quantile"
        )

        # Cell 1, edges = (G,A)
        data = tokenizer(self.expression.loc['Cell1'])
        genes = [tokenizer.vocab.node_to_gene[int(i)] for i in data.x.numpy()[:, 0]]
        edges = data.edge_index.numpy()
        expected_genes = ["A", "G"] 
        expected_edges = [[1], [0]]

        self.assertEqual(expected_genes, genes)
        self.assertTrue(np.array_equal(expected_edges, edges))

        # Cell 2, edges = (B,D)
        data = tokenizer(self.expression.loc['Cell2'])
        genes = [tokenizer.vocab.node_to_gene[int(i)] for i in data.x[:, 0].numpy()]
        edges = data.edge_index.numpy()
        expected_genes = ["B", "C", "D", "G"]
        expected_edges = [[0], [2]]

        self.assertEqual(expected_genes, genes)
        self.assertTrue(np.array_equal(expected_edges, edges))

        # Cell 3, edges = (A,C), (H,C)
        data = tokenizer(self.expression.loc['Cell3'])
        genes = [tokenizer.vocab.node_to_gene[int(i)] for i in data.x[:, 0].numpy()]
        edges = data.edge_index.numpy()
        expected_genes = ["A", "C", "H"]
        expected_edges = [[0, 2], [1, 1]]

        self.assertEqual(expected_genes, genes)
        self.assertTrue(np.array_equal(expected_edges, edges))

    def test_not_only_expressed_genes(self):
        tokenizer = GraphTokenizer(
            vocab=self.vocab,
            network=self.network,
            max_seq_length=5,
            only_expressed_genes=False,
            with_edge_weights=True,
            n_bins=10,
            method="quantile"
        )

        # Cell 1, edges = (A, C), (B,D), (G,A)
        data = tokenizer(self.expression.loc['Cell1'])
        genes = [tokenizer.vocab.node_to_gene[int(i)] for i in data.x.numpy()[:, 0]]
        edges = data.edge_index.numpy()
        expected_genes = ["A", "G", "B", "C", "D"] 
        expected_edges = [[0,2,1], [3,4,0]]
        self.assertEqual(expected_genes, genes)
        self.assertTrue(np.array_equal(expected_edges, edges))

        # Cell 2, edges = (B,D), (C,E)
        data = tokenizer(self.expression.loc['Cell2'])
        genes = [tokenizer.vocab.node_to_gene[int(i)] for i in data.x[:, 0].numpy()]
        edges = data.edge_index.numpy()
        expected_genes = ["B", "D", "C", "G", "E"]
        expected_edges = [[0, 2], [1, 4]]

        self.assertEqual(expected_genes, genes)
        self.assertTrue(np.array_equal(expected_edges, edges))

        # Cell 3, edges = (A,C), (C,E), (H,E), (H,C)
        data = tokenizer(self.expression.loc['Cell3'])
        genes = [tokenizer.vocab.node_to_gene[int(i)] for i in data.x[:, 0].numpy()]
        edges = data.edge_index.numpy()
        expected_genes = ["H", "E", "A", "C", "B"]
        expected_edges = [[2, 3, 0, 0], [3, 1, 1, 3]]

        self.assertEqual(expected_genes, genes)
        self.assertTrue(np.array_equal(expected_edges, edges))

    def test_not_only_network_genes(self):
        tokenizer = GraphTokenizer(
            vocab=self.vocab,
            network=self.network,
            max_seq_length=5,
            only_expressed_genes=True,
            only_network_genes=False,
            with_edge_weights=True,
            n_bins=10,
            method="quantile"
        )
        # Cell 1, edges = (G,A)
        data = tokenizer(self.expression.loc['Cell1'])
        genes = [tokenizer.vocab.node_to_gene[int(i)] for i in data.x.numpy()[:, 0]]
        edges = data.edge_index.numpy()
        expected_genes = ["A", "F", "G"] 
        expected_edges = [[2], [0]]

        self.assertEqual(expected_genes, genes)
        self.assertTrue(np.array_equal(expected_edges, edges))

        # Cell 2, edges = (B,D), (C,E)
        data = tokenizer(self.expression.loc['Cell2'])
        genes = [tokenizer.vocab.node_to_gene[int(i)] for i in data.x[:, 0].numpy()]
        edges = data.edge_index.numpy()
        expected_genes = ["B", "C", "D", "E", "G"]
        expected_edges = [[0, 1], [2, 3]]

        self.assertEqual(expected_genes, genes)
        self.assertTrue(np.array_equal(expected_edges, edges))

        # Cell 3, edges = (A,C), (C,E), (H,E), (H,C)
        data = tokenizer(self.expression.loc['Cell3'])
        genes = [tokenizer.vocab.node_to_gene[int(i)] for i in data.x[:, 0].numpy()]
        edges = data.edge_index.numpy()
        expected_genes = ["A", "C", "E", "F", "H"]
        expected_edges = [[0, 1, 4, 4], [1, 2, 2, 1]]

        self.assertEqual(expected_genes, genes)
        self.assertTrue(np.array_equal(expected_edges, edges))

    def test_graph_is_empty_when_all_genes_zero(self):
        tokenizer = GraphTokenizer(vocab=self.vocab, network=self.network)
        cell_expr = pd.Series([0]*8, index=self.expression.columns)
        data = tokenizer(cell_expr)

        self.assertEqual(data.x.shape[0], 0)
        self.assertEqual(data.edge_index.shape[1], 0)

    def test_tokenizer_output_type(self):
        tokenizer = GraphTokenizer(vocab=self.vocab, network=self.network, with_edge_weights=True)
        cell_expr = self.expression.loc['Cell2']
        data = tokenizer(cell_expr)

        self.assertIsInstance(data, torchGeomData)
        self.assertTrue(hasattr(data, 'x'))
        self.assertTrue(hasattr(data, 'edge_index'))
        self.assertTrue(hasattr(data, 'edge_weight'))


if __name__ == '__main__':
    unittest.main()
