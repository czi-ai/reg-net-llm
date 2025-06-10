import pandas as pd
import numpy as np
import unittest
from itertools import chain

from scGraphLLM.infer_graph import *
from scGraphLLM.network import RegulatoryNetwork


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

        self.network5 = formulate_network([
            ["H", "L", 0.5, np.log(0.03)],
            ["H", "A", 2.8, np.log(0.02)],
            ["H", "B", 1.9, np.log(0.04)],
            ["H", "I", 0.5, np.log(0.02)],
            ["H", "C", 5.0, np.log(0.03)],
            ["H", "D", 0.1, np.log(0.01)],
            ["C", "A", 1.7, np.log(0.01)],
            ["C", "D", 1.7, np.log(0.01)]
        ])

        self.classes = [
            "class1", 
            "class2", 
            "class3", 
            "class4", 
            "class5"
        ]

        self.class_networks = {
            self.classes[0]: self.network1,
            self.classes[1]: self.network2, 
            self.classes[2]: self.network3,
            self.classes[3]: self.network4,
            self.classes[4]: self.network5
        }
 
        self.all_edges = list({
            e for net in self.class_networks.values() for e in net.edges
        })

    def test_edge_matrix(self):
        E, MI, _ = build_class_edge_matrix(self.class_networks, self.classes, default_alpha=(0.05 + 1)/2)     
        total_edges = len(self.all_edges)
        total_classes = len(self.class_networks)
        self.assertEqual((total_edges, total_classes), E.shape)
        self.assertEqual((total_edges, total_classes), MI.shape)

    def test_implicit_hard_assignment(self):
        E, MI, all_edges = build_class_edge_matrix(self.class_networks, self.classes, default_alpha=(0.05 + 1.)/2)
        for i, c in enumerate(self.classes):
            probs = np.zeros_like(self.classes, dtype=float)
            probs[i] = 1.0
            edge_ids, pvals, mis = infer_cell_edges_(probs, E, MI, alpha=0.05 + 1e-4)
            regulators, targets = zip(*all_edges[edge_ids])
            cell_network = RegulatoryNetwork(regulators, targets, mis, np.log(pvals))
            class_network = self.class_networks[c]
            self.assertEqual(class_network, cell_network)

    def test_default_alpha(self):
        # uniform soft class assignment
        probs = np.ones_like(self.classes, dtype=float) / len(self.classes)

        # high default alpha
        E, MI, all_edges = build_class_edge_matrix(self.class_networks, self.classes, default_alpha=0.99)
        edge_ids, pvals, mis = infer_cell_edges_(probs, E, MI, alpha=0.05 + 1e-4)
        edges = all_edges[edge_ids]
        regulators = [e[0] for e in edges]
        targets = [e[1] for e in edges]
        cell_network = RegulatoryNetwork(regulators, targets, mis, np.log(pvals))
        self.assertEqual(0, len(cell_network))

        # medium default alpha
        E, MI, all_edges = build_class_edge_matrix(self.class_networks, self.classes, default_alpha=(0.05 + 1.)/2)
        edge_ids, pvals, mis = infer_cell_edges_(probs, E, MI, alpha=0.2)
        regulators, targets = zip(*all_edges[edge_ids])
        cell_network = RegulatoryNetwork(regulators, targets, mis, np.log(pvals))
        cell_edges = set(cell_network.edges)
        expected_edges = {("A", "C"), ("C", "A")}
        self.assertEqual(expected_edges, cell_edges)

        # low default alpha
        E, MI, all_edges = build_class_edge_matrix(self.class_networks, self.classes, default_alpha=0.05)
        edge_ids, pvals, mis = infer_cell_edges_(probs, E, MI, alpha=0.05 + 1e-4)
        regulators, targets = zip(*all_edges[edge_ids])
        cell_network = RegulatoryNetwork(regulators, targets, mis, np.log(pvals))
        cell_edges = set(cell_network.edges)
        expected_edges = set(self.all_edges)
        self.assertEqual(expected_edges, cell_edges)

    def test_same_networks(self):
        class_networks = {
            self.classes[0]: self.network2,
            self.classes[1]: self.network2, 
            self.classes[2]: self.network2,
            self.classes[3]: self.network2,
            self.classes[4]: self.network2
        }
        E, MI, all_edges = build_class_edge_matrix(class_networks, self.classes, default_alpha=(0.05 + 1.)/2)
        for _ in range(100):
            probs = np.random.rand(len(self.classes))
            probs /= probs.sum()
            edge_ids, pvals, mis = infer_cell_edges_(probs, E, MI, alpha=0.2)
            regulators, targets = zip(*all_edges[edge_ids])
            cell_network = RegulatoryNetwork(regulators, targets, mis, np.log(pvals))
            self.assertEqual(self.network2, cell_network)
    
    def test_limit_regulon(self):
        E, MI, all_edges = build_class_edge_matrix(self.class_networks, self.classes, default_alpha=(0.05 + 1.)/2)
        
        # likely class 3, aggressive pruning with symmetry
        probs = [0.07, 0.0, 0.9, 0.03, 0.0]
        edge_ids, pvals, mis = infer_cell_edges_(probs, E, MI, alpha=0.2)
        regulators, targets = zip(*all_edges[edge_ids])
        cell_network = RegulatoryNetwork(regulators, targets, mis, np.log(pvals))\
            .prune(limit_regulon=2)\
            .make_undirected(drop_unpaired=True)
        # C's regulon is originally 3, so one of its targets (C, A) should be pruned
        self.assertFalse(("C", "A") in cell_network.edges)
        self.assertFalse(("A", "C") in cell_network.edges)
        self.assertEqual(6, len(cell_network))

        # likely class 5, aggressive pruning with symmetry
        probs = [0.0, 0.05, 0.0, 0.30, 0.65]
        edge_ids, pvals, mis = infer_cell_edges_(probs, E, MI, alpha=0.25)
        regulators, targets = zip(*all_edges[edge_ids])
        cell_network = RegulatoryNetwork(regulators, targets, mis, np.log(pvals))\
            .prune(limit_regulon=2)\
            .make_undirected(drop_unpaired=True)
        cell_edges = set(cell_network.edges)
        expected_edges = [
            ("A", "H"), 
            ("C", "H"), ("C", "D"), 
            ("D", "C"), 
            ("H", "C"), ("H", "A")
        ]
        expected_edges = set(expected_edges + [(t, r) for r, t in expected_edges])
        self.assertEqual(expected_edges, cell_edges)
        
        # likely class 5, pruning with reinstallation of edges for symmetry
        edge_ids, pvals, mis = infer_cell_edges_(probs, E, MI, alpha=0.25)
        regulators, targets = zip(*all_edges[edge_ids])
        cell_network = RegulatoryNetwork(regulators, targets, mis, np.log(pvals))\
            .prune(limit_regulon=2)\
            .make_undirected(drop_unpaired=False)
        cell_edges = set(cell_network.edges)
        expected_edges = [
            ("A", "H"), ("A", "C"), 
            ("B", "H"), 
            ("C", "H"), ("C", "D"), 
            ("D", "C"), ("D", "H"), 
            ("H", "C"), ("H", "A"), ("H", "B"), ("H", "D"), ("H", "I"), ("H", "L"), 
            ("I", "H"), 
            ("L", "H")
        ]
        expected_edges = set(expected_edges + [(t, r) for r, t in expected_edges])
        self.assertEqual(expected_edges, cell_edges)

        # likely class 5, pruning without required symmetry
        edge_ids, pvals, mis = infer_cell_edges_(probs, E, MI, alpha=0.25)
        regulators, targets = zip(*all_edges[edge_ids])
        cell_network = RegulatoryNetwork(regulators, targets, mis, np.log(pvals))\
            .prune(limit_regulon=2)
        cell_edges = set(cell_network.edges)
        expected_edges = set([
            ("A", "H"), ("A", "C"), 
            ("B", "H"), 
            ("C", "H"), ("C", "D"), 
            ("D", "C"), ("D", "H"), 
            ("H", "C"), ("H", "A"),
            ("I", "H"), 
            ("L", "H")
        ])
        self.assertEqual(expected_edges, cell_edges)

        # likely class 5, graph-based pruning instead of regulon-based pruning
        edge_ids, pvals, mis = infer_cell_edges_(probs, E, MI, alpha=0.25)
        regulators, targets = zip(*all_edges[edge_ids])
        cell_network = RegulatoryNetwork(regulators, targets, mis, np.log(pvals))\
            .prune(limit_graph=6)
        cell_edges = set(cell_network.edges)
        expected_edges = set([
            ("A", "H"),
            ("C", "H"),
            ("C", "D"),
            ("D", "C"),
            ("H", "C"),
            ("H", "A"),
        ])
        self.assertEqual(expected_edges, cell_edges)

        # likely class 5, regulon-based pruning followed by graph-based pruning
        edge_ids, pvals, mis = infer_cell_edges_(probs, E, MI, alpha=0.25)
        regulators, targets = zip(*all_edges[edge_ids])
        cell_network = RegulatoryNetwork(regulators, targets, mis, np.log(pvals))\
            .prune(limit_regulon=1, limit_graph=5)
        cell_edges = set(cell_network.edges)
        expected_edges = set([
            ("A", "H"),
            ("B", "H"), 
            ("C", "H"),
            ("D", "C"),
            ("H", "C"),
        ])
        self.assertEqual(expected_edges, cell_edges)

        # likely class 5, regulon-based pruning followed by graph-based pruning, requiring undirected graph
        edge_ids, pvals, mis = infer_cell_edges_(probs, E, MI, alpha=0.25)
        regulators, targets = zip(*all_edges[edge_ids])
        cell_network = RegulatoryNetwork(regulators, targets, mis, np.log(pvals))\
            .prune(limit_regulon=1, limit_graph=5)\
            .make_undirected(drop_unpaired=True)
        cell_edges = set(cell_network.edges)
        expected_edges = set([
            ("C", "H"),
            ("H", "C"),
        ])
        self.assertEqual(expected_edges, cell_edges)


def formulate_network(data):
    data = np.array(data)
    return RegulatoryNetwork(
        regulators=data[:,0], 
        targets=data[:,1], 
        weights=data[:,2],
        likelihoods=data[:,3]
    ).make_undirected(drop_unpaired=False).sort()
        

if __name__ == "__main__":
    unittest.main()