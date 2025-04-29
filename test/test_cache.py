# test_math_utils.py
import pandas as pd
import numpy as np
import unittest

from scGraphLLM.data import quantize_cells

class TestRunSave(unittest.TestCase):

    def setUp(self):
        self.n_bins = 5
        self.expression = pd.DataFrame(
            [[10, 0, 0, 0, 0, 0, 2, 0],
             [0, 8, 4, 6, 2, 0, 3, 0],
             [3, 0, 1, 0, 4, 0, 0, 6]], 
            index=['Cell1', 'Cell2', 'Cell3'], 
            columns=['GeneA', 'GeneB', 'GeneC', 'GeneD', 'GeneE', 'GeneF', 'GeneG', 'GeneH']
        )
    
        
    # Check that the shape is consistent
    # Also check that the rows & columns are formatted the same
    def test_format(self):
        # Shape
        bins = quantize_cells(self.expression, n_bins=self.n_bins)
        self.assertTrue(self.expression.shape == bins.shape)

        # Axes
        check_rows = self.expression.index == bins.index
        check_cols = self.expression.columns == bins.columns
        self.assertTrue(check_rows.all() and check_cols.all())
        

# Test different graphs - make sure the result is different
# Test all zeros expression
# Test equivalent expression
# Make sure file isn't corrupted
# Split into smaller functions

if __name__ == '__main__':
    unittest.main()
