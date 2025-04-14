# test_math_utils.py
import pandas as pd
import numpy as np
import unittest

from scGraphLLM.preprocess import rank_cells


class TestRank(unittest.TestCase):

    def setUp(self):
        self.expression = pd.DataFrame(
            [[10, 0, 0, 0, 0, 0, 2, 0],
             [0, 8, 0, 6, 2, 0, 3, 0],
             [3, 0, 1, 0, 4, 0, 0, 6]], 
            index=['Cell1', 'Cell2', 'Cell3'], 
            columns=['GeneA', 'GeneB', 'GeneC', 'GeneD', 'GeneE', 'GeneF', 'GeneG', 'GeneH']
        )

    def test_rank(self):
        ranks = rank_cells(self.expression, n_bins=5, rank_by_z_score=False)
        print("Ranks...")
        print(ranks)

        zero_expression = self.expression.to_numpy() == 0 
        zero_rank = ranks.to_numpy() == 0
        self.assertTrue(np.array_equal(zero_expression, zero_rank))

    def test_rank_with_z_score(self):
        ranks = rank_cells(self.expression, n_bins=5, rank_by_z_score=True)
        print("Ranks with z-score...")
        print(ranks)
      
    def test_n_bins_greater_than_n_genes(self):
        pass

    def test_n_bins_equals_n_genes(self):
        pass

    def test_equal_expression(self):
        expr = pd.DataFrame(
            [np.repeat(2, 8),  # 1, 1, 1, 1
             np.repeat(3, 8),  # 1, 1, 1, 1
             np.repeat(0, 8)], # 0, 0, 0, 0
            index=['Cell1', 'Cell2', 'Cell3'], 
            columns=['GeneA', 'GeneB', 'GeneC', 'GeneD', 'GeneE', 'GeneF', 'GeneG', 'GeneH']
        )
        ranks = rank_cells(expr, n_bins=5, rank_by_z_score=False)
        print(ranks)
        pass

    # testing ideas
    # 1 bin
    # many many bins


if __name__ == '__main__':
    unittest.main()
