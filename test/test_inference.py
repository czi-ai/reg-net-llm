import unittest
import pandas as pd
import numpy as np

from scGraphLLM._globals import CLS_GENE, MASK_GENE
from scGraphLLM.tokenizer import GraphTokenizer
from scGraphLLM.vocab import GeneVocab
from scGraphLLM.network import RegulatoryNetwork
from scGraphLLM.inference import InferenceDataset, VariableNetworksInferenceDataset


class TestInferenceDatasets(unittest.TestCase):
    def setUp(self):
        self.expression = pd.DataFrame(
            [[1, 0, 3, 0, 0], [0, 2, 0, 4, 1]],
            index=["Cell1", "Cell2"],
            columns=["A", "B", "C", "D", "E"]
        )

        self.vocab = GeneVocab(
            genes=["A", "B", "C", "D", "E", CLS_GENE, MASK_GENE],
            nodes=[0, 1, 2, 3, 4, 17936, 903],
            require_special_tokens=False
        )

        self.network = RegulatoryNetwork(
            regulators=["A", "B", "E"],
            targets=["C", "D", "B"],
            weights=[1.0, 1.0, 1.0],
            likelihoods=[-2.4, -1.8, -2.0]
        )

        self.tokenizer = GraphTokenizer(
            vocab=self.vocab,
            network=self.network,
            only_expressed_plus_neighbors=True
        )

    def test_inference_dataset(self):
        dataset = InferenceDataset(expression=self.expression, tokenizer=self.tokenizer)
        self.assertEqual(len(dataset), 2)

        item = dataset[0]
        self.assertTrue(np.array_equal([17936, 0, 2], item["orig_gene_id"].numpy()))
        self.assertTrue(np.array_equal([[0],[1]], item["edge_index"].numpy()))
        self.assertTrue(np.array_equal([False, False, False], item["gene_mask"].numpy()))
        self.assertEqual(2, item["num_nodes"])
        self.assertEqual("Cell1", item["obs_name"])

        item = dataset[1]
        self.assertTrue(np.array_equal([17936, 1, 3, 4], item["orig_gene_id"].numpy()))
        self.assertTrue(np.array_equal([[0, 2],[1, 0]], item["edge_index"].numpy()))
        self.assertTrue(np.array_equal([False, False, False, False], item["gene_mask"].numpy()))
        self.assertEqual(3, item["num_nodes"])
        self.assertEqual("Cell2", item["obs_name"])

    def test_variable_networks_inference_dataset(self):
        all_edges = np.array([["A", "C"], ["E", "B"], ["B", "D"], ["E", "A"]])
        edge_ids_list = [np.array([0, 2]), np.array([0, 2, 3])]
        weights_list = [np.array([0.9, 0.8]), np.array([1.2, 0.7,  1.5])]

        dataset = VariableNetworksInferenceDataset(
            expression=self.expression,
            tokenizer=self.tokenizer,
            edge_ids_list=edge_ids_list,
            all_edges=all_edges,
            weights_list=weights_list
        )
        self.assertEqual(len(dataset), 2)

        item = dataset[0]
        self.assertTrue(np.array_equal([17936, 0, 2], item["orig_gene_id"].numpy()))
        self.assertTrue(np.array_equal([[0], [1]], item["edge_index"].numpy()))  # A→C, B→D
        self.assertTrue(np.array_equal([False, False, False], item["gene_mask"].numpy()))
        self.assertEqual(2, item["num_nodes"])
        self.assertEqual("Cell1", item["obs_name"])

        item = dataset[1]
        self.assertTrue(np.array_equal([17936, 0, 1, 3, 4], item["orig_gene_id"].numpy()))
        self.assertTrue(np.array_equal([[1, 3],[2, 0]], item["edge_index"].numpy()))  # E→B, E→D
        self.assertTrue(np.array_equal([False, False, False, False, False], item["gene_mask"].numpy()))
        self.assertEqual(4, item["num_nodes"])
        self.assertEqual("Cell2", item["obs_name"])


    def test_network_pruning(self):
        all_edges = np.array([["B", "A"], ["B", "C"], ["B", "D"], ["B", "E"], ["E", "B"]])
        edge_ids_list = [np.array([0, 1, 2]), np.array([0,1,2,3,4])]
        weights_list = [np.array([0.5, 0.4, 0.3]), np.array([0.5, 0.4, 0.3, 0.9, 0.1])]

        dataset = VariableNetworksInferenceDataset(
            expression=self.expression,
            tokenizer=self.tokenizer,
            edge_ids_list=edge_ids_list,
            all_edges=all_edges,
            weights_list=weights_list,
            limit_regulon=2
        )
        
        item = dataset[1]
        self.assertTrue(np.array_equal([17936, 0, 1, 4], item["orig_gene_id"].numpy()))
        self.assertTrue(np.array_equal([[1, 1, 2], [2, 0, 1]], item["edge_index"].numpy()))
        self.assertTrue(np.array_equal([False, False, False, False], item["gene_mask"].numpy()))
        self.assertEqual(3, item["num_nodes"])
        self.assertEqual("Cell2", item["obs_name"])


if __name__ == "__main__":
    unittest.main()
