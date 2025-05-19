import pandas as pd
import numpy as np
import unittest

from scGraphLLM.inference import *
from pyviper import Interactome

REG_VALS = "regulator.values"
TAR_VALS = "target.values"
MI_VALS = "mi.values"
LOGP_VALS = "log.p.values"
SCC_VALS = "scc.values"


def make_undirected(df, reg_col=REG_VALS, tar_col=TAR_VALS, mi_col=MI_VALS, logp_col=LOGP_VALS):
    """
    Takes a directed graph dataframe and returns an undirected version,
    where for every (A, B), the reverse (B, A) is added if missing.
    """
    reversed_edges = []

    # Track existing edges as a set of (reg, tar) tuples
    existing_edges = set(zip(df[reg_col], df[tar_col]))

    for idx, row in df.iterrows():
        src, tgt = row[reg_col], row[tar_col]
        if (tgt, src) not in existing_edges:
            reversed_edges.append({
                reg_col: tgt,
                tar_col: src,
                mi_col: row[mi_col],
                logp_col: row[logp_col]
            })

    # Create a DataFrame of the reversed edges
    if reversed_edges:
        reversed_df = pd.DataFrame(reversed_edges)
        df = pd.concat([df, reversed_df], ignore_index=True)

    return df

def formulate_network(data, columns=[REG_VALS, TAR_VALS, MI_VALS, LOGP_VALS]):
    return pd.DataFrame(data=data, columns=columns)\
        .pipe(make_undirected)\
        .set_index([REG_VALS, TAR_VALS]) \
        .sort_index()


class TestGraphIntegration(unittest.TestCase):

    def setUp(self):
        self.network1 = formulate_network([
            ["A", "B", 0.6, np.log(0.01)],
            ["F", "G", 0.7, np.log(0.05)],
            ["B", "I", 0.2, np.log(0.02)]
        ])

        self.network2 = formulate_network([
            ["A", "B", 1.6, np.log(0.03)],
            ["H", "G", 0.4, np.log(0.02)],
            ["C", "A", 2.0, np.log(0.01)],
            ["J", "D", 2.0, np.log(0.01)]
        ])

        self.network3 = formulate_network([
            ["C", "B", 1.2, np.log(0.03)],
            ["C", "A", 0.8, np.log(0.01)],
            ["C", "I", 3.0, np.log(0.04)],
            ["J", "D", 0.4, np.log(0.01)]
        ])

        self.network4 = formulate_network([
            ["H", "A", 0.2, np.log(0.03)],
            ["C", "A", 1.0, np.log(0.02)],
            ["F", "A", 1.1, np.log(0.01)],
            ["H", "B", 0.5, np.log(0.02)]
        ])
        self.classes = ["class1", "class2", "class3", "class4"]
        self.class_networks = {
            self.classes[0]: self.network1,
            self.classes[1]: self.network2, 
            self.classes[2]: self.network3,
            self.classes[3]: self.network4
        }
        self.all_edges = np.unique(np.concatenate([network.index for _, network in self.class_networks.items()]))
        

    def test_edge_matrix(self):
        E, MI, _ = build_class_edge_matrix(self.class_networks, self.classes, default_alpha=(0.05 + 1)/2)     
        total_edges = self.all_edges.size
        total_classes = len(self.class_networks)
        self.assertEqual((total_edges, total_classes), E.shape)
        self.assertEqual((total_edges, total_classes), MI.shape)

    def test_implicit_hard_assignment(self):
        E, MI, all_edges = build_class_edge_matrix(self.class_networks, self.classes, default_alpha=(0.05 + 1.)/2)
        for i, c in enumerate(self.classes):
            probs = np.zeros_like(self.classes, dtype=float)
            probs[i] = 1.0
            edge_ids, pvals, mis = infer_cell_edges_(probs, E, MI, alpha=0.05 + 1e-4)
            cell_network = get_cell_network_df(edge_ids, pvals, mis, all_edges)\
                .set_index([REG_VALS, TAR_VALS])\
                .sort_index()
            class_network = self.class_networks[c]
            pd.testing.assert_frame_equal(class_network, cell_network)

    def test_default_alpha(self):
        # uniform soft class assignment
        probs = np.ones_like(self.classes, dtype=float) / len(self.classes)

        # high default alpha
        E, MI, all_edges = build_class_edge_matrix(self.class_networks, self.classes, default_alpha=0.99)
        edge_ids, pvals, mis = infer_cell_edges_(probs, E, MI, alpha=0.05 + 1e-4)
        cell_network = get_cell_network_df(edge_ids, pvals, mis, all_edges).set_index([REG_VALS, TAR_VALS])
        self.assertEqual(0, len(cell_network))

        # medium default alpha
        E, MI, all_edges = build_class_edge_matrix(self.class_networks, self.classes, default_alpha=(0.05 + 1.)/2)
        edge_ids, pvals, mis = infer_cell_edges_(probs, E, MI, alpha=0.2)
        cell_network = get_cell_network_df(edge_ids, pvals, mis, all_edges).set_index([REG_VALS, TAR_VALS])
        cell_edges = set(cell_network.index)
        expected_edges = {("A", "C"), ("C", "A")}
        self.assertEqual(expected_edges, cell_edges)

        # low default alpha
        E, MI, all_edges = build_class_edge_matrix(self.class_networks, self.classes, default_alpha=0.05)
        edge_ids, pvals, mis = infer_cell_edges_(probs, E, MI, alpha=0.05 + 1e-4)
        cell_network = get_cell_network_df(edge_ids, pvals, mis, all_edges).set_index([REG_VALS, TAR_VALS])
        cell_edges = set(cell_network.index)
        expected_edges = set(self.all_edges)
        self.assertEqual(expected_edges, cell_edges)

    def test_same_networks(self):
        class_networks = {
            self.classes[0]: self.network2,
            self.classes[1]: self.network2, 
            self.classes[2]: self.network2,
            self.classes[3]: self.network2
        }
        E, MI, all_edges = build_class_edge_matrix(class_networks, self.classes, default_alpha=(0.05 + 1.)/2)
        for _ in range(100):
            probs = np.random.rand(len(self.classes))  # Uniform random values in [0, 1)
            probs /= probs.sum()
            edge_ids, pvals, mis = infer_cell_edges_(probs, E, MI, alpha=0.2)
            cell_network = get_cell_network_df(edge_ids, pvals, mis, all_edges).set_index([REG_VALS, TAR_VALS])
            pd.testing.assert_frame_equal(self.network2, cell_network)
    
    def test_limit_regulon(self):
        E, MI, all_edges = build_class_edge_matrix(self.class_networks, self.classes, default_alpha=(0.05 + 1.)/2)
        probs = [0.07, 0.0, 0.9, 0.03]
        edge_ids, pvals, mis = infer_cell_edges_(probs, E, MI, alpha=0.2)
        cell_network = get_cell_network_df(edge_ids, pvals, mis, all_edges, limit_regulon=2).set_index([REG_VALS, TAR_VALS])
        self.assertEqual(7, len(cell_network))
        

if __name__ == "__main__":
    unittest.main()