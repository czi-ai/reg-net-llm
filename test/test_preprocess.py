# test_math_utils.py
import pandas as pd
import numpy as np
import unittest

from scGraphLLM.preprocess import quantize_cells

class TestQuantization(unittest.TestCase):

    def setUp(self):
        self.n_bins = 5
        self.expression = pd.DataFrame(
            [[10, 0, 0, 0, 0, 0, 2, 0],
             [0, 8, 4, 6, 2, 0, 3, 0],
             [3, 0, 1, 0, 4, 0, 0, 6]], 
            index=['Cell1', 'Cell2', 'Cell3'], 
            columns=['GeneA', 'GeneB', 'GeneC', 'GeneD', 'GeneE', 'GeneF', 'GeneG', 'GeneH']
        )
        
    
    # Runs all the tests not including those where n_bins is changed
    def run_tests(self):
        self.test_format()
        self.test_bin_range()
        self.test_zeros()
        self.test_unique_bins()
        self.test_min_max_pos()
        self.test_equal_expression()
        self.test_uniform_expression()
        self.test_various_expression()
        
    # Runs all the bin tests (and subsequently all run_tests() tests)
    def run_bin_tests(self):
        self.test_n_bins_greater_than_n_genes()
        self.test_too_many_bins()
        self.test_n_bins_equals_n_genes()
        self.test_2_bin()
    
        
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
        
    
    # Make sure the bin values are within the n_bins range
    def test_bin_range(self):
        bins = quantize_cells(self.expression, n_bins=self.n_bins)
        mn = bins.min().min()
        mx = bins.max().max()
        self.assertTrue((mn >= 0) and (mx < self.n_bins))
        
    
    # Check that all zero-expression genes are assigned to zero bin
    # Also checks that no expressed genes are assigned to zero bin
    def test_zeros(self):
        bins = quantize_cells(self.expression, n_bins=self.n_bins)
        zero_mask_input = self.expression == 0
        zero_mask_output = bins == 0
        self.assertTrue(np.array_equal(zero_mask_input, zero_mask_output))
        
        
    # Check if the number of unique bins corresponds correctly to the number of unique expression values in a given cell
    def test_unique_bins(self):
        bins = quantize_cells(self.expression, n_bins=self.n_bins)
        
        for i in range(bins.shape[0]): # Iterate through cells
            cell = self.expression.iloc[i]
            cell_bins = bins.iloc[i]
            
            n_unique_expressed_genes = cell[cell != 0].unique().shape[0] # Get number of genes with unique non-zero expression (identical expressions get mapped to the same bins)
            n_unique_bins = cell_bins[cell_bins != 0].unique().shape[0] # Get number of uniquenon-zero bins
            
            if n_unique_expressed_genes <= self.n_bins-1: # Less expressed genes than requested n_bins (-1 as we are not accounting for the zero expression bin)
                # Then number of unique bins should be the same as the number of expressed genes -> one bin per unique gene expression
                self.assertTrue(n_unique_bins == n_unique_expressed_genes) # Check no two different expressions are in the same bin
            else: # More expressed genes than there are requested n_bins
                # Then all the bins must be in use (with multiple expressions in the same bin)
                self.assertTrue(n_unique_bins == self.n_bins-1)
            

    # Ensure that the bin assignment is in ascending order
    def test_min_max_pos(self):
        bins = quantize_cells(self.expression, n_bins=self.n_bins)
        
        input_mins = self.expression[self.expression != 0].min(axis=1)
        input_mins_mask = self.expression.eq(input_mins, axis=0) & (self.expression != 0)
        output_mins = bins[bins != 0].min(axis=1)
        output_mins_mask = bins.eq(output_mins, axis=0) & (bins != 0)
        
        input_maxs = self.expression[self.expression != 0].max(axis=1)
        input_maxs_mask = self.expression.eq(input_maxs, axis=0) & (self.expression != 0)
        output_maxs = bins[bins != 0].max(axis=1)
        output_maxs_mask = bins.eq(output_maxs, axis=0) & (bins != 0)
        
        check_mins = input_mins_mask[input_mins_mask].equals(output_mins_mask[input_mins_mask])
        check_maxs = input_maxs_mask[input_maxs_mask].equals(output_maxs_mask[input_maxs_mask])

        self.assertTrue(check_mins and check_maxs)
    
    
    # Check that the same expressions are assigned to the same bins in each row
    def test_equal_expression(self):
        bins = quantize_cells(self.expression, n_bins=self.n_bins)
        
        for i in range(self.expression.shape[0]): # For each cell
            cell = self.expression.iloc[i]
            cell_bins = bins.iloc[i]
            unq = cell.unique()
            for expr in unq: # Iterate through each expression value in this cell
                expr_msk = cell == expr # Get locations of this expression in the cell
                expr_bins = cell_bins[expr_msk] # Get the expression bins of these locations in the quantized cell
                self.assertTrue(expr_bins.unique().shape[0] == 1) # Check that there is only one bin housing this expression level
    
    
    # Check behavior if all expression values are the same in a given cell
    def test_uniform_expression(self):
        expr = pd.DataFrame(
            [np.repeat(2, 8),
             np.repeat(5, 8),
             np.repeat(0, 8)],
            index=['Cell1', 'Cell2', 'Cell3'], 
            columns=['GeneA', 'GeneB', 'GeneC', 'GeneD', 'GeneE', 'GeneF', 'GeneG', 'GeneH']
        )
        bins = quantize_cells(expr, n_bins=self.n_bins)
        
        # Cell 0
        self.assertTrue(bins.iloc[0].unique()[0] == round(self.n_bins/2)) # Check set to median bin value
        self.assertTrue(bins.iloc[0].unique().shape[0] == 1) # Check that there is only one bin
        
        # Cell 1
        self.assertTrue(bins.iloc[1].unique()[0] == round(self.n_bins/2)) # Check set to median bin value
        self.assertTrue(bins.iloc[1].unique().shape[0] == 1) # Check that there is only one bin
        
        # Cell 2
        self.assertTrue(bins.iloc[2].unique()[0] == 0) # Check set to zero bin for zero expression
        self.assertTrue(bins.iloc[2].unique().shape[0] == 1) # Check that there is only one bin
    
    
    # Test various input expression layouts
    def test_various_expression(self):
        expr = pd.DataFrame(
            [[0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 7, 0],
            [0, 0, 4, 4, 0, 0, 4, 0],
            [0, 0, 1, 0, 0, 0, 3, 0]],
            index=['Cell1', 'Cell2', 'Cell3', 'Cell4'], 
            columns=['GeneA', 'GeneB', 'GeneC', 'GeneD', 'GeneE', 'GeneF', 'GeneG', 'GeneH']
        )
        
        n_bins = self.n_bins # Just to make this a little more concise
        expected = pd.DataFrame(
            [[0, 0, 0, 0, 0, 0, round(n_bins/2), 0], # With only one non-zero expression value, we expect the median bin value
            [0, 0, 0, 0, 0, 0, round(n_bins/2), 0],
            [0, 0, round(n_bins/2), round(n_bins/2), 0, 0, round(n_bins/2), 0],
            [0, 0, 1, 0, 0, 0, n_bins-1, 0]],
            index=['Cell1', 'Cell2', 'Cell3', 'Cell4'], 
            columns=['GeneA', 'GeneB', 'GeneC', 'GeneD', 'GeneE', 'GeneF', 'GeneG', 'GeneH']
        )
        bins = quantize_cells(expr, n_bins=self.n_bins)
        self.assertTrue(bins.equals(expected.astype(bins.dtypes)))
        

    # Test having more bins than genes
    def test_n_bins_greater_than_n_genes(self):
        original_n_bins = self.n_bins
        self.n_bins = self.expression.shape[1] + 1
        print(f"\nTesting n_bins={self.n_bins}")
        self.run_tests() # Run all the earlier tests with new n_bins value
        print(f"Finished testing n_bins={self.n_bins}")
        self.n_bins = original_n_bins
        
    
    # Test having many many more bins than genes | Make sure n_bins not divisible by num_genes
    def test_too_many_bins(self):
        original_n_bins = self.n_bins
        self.n_bins = self.expression.shape[1] * 71 + 5
        print(f"\nTesting n_bins={self.n_bins}")
        self.run_tests() # Run all the earlier tests with new n_bins value
        print(f"Finished testing n_bins={self.n_bins}")
        self.n_bins = original_n_bins
        
    
    # Test having as many bins as genes
    def test_n_bins_equals_n_genes(self):
        original_n_bins = self.n_bins
        self.n_bins = self.expression.shape[1]
        print(f"\nTesting n_bins={self.n_bins}")
        self.run_tests() # Run all the earlier tests with new n_bins value
        print(f"Finished testing n_bins={self.n_bins}")
        self.n_bins = original_n_bins
        
    
    # Test having only 2 bins (binary, 0 and 1)
    def test_2_bin(self):
        original_n_bins = self.n_bins
        self.n_bins = 2
        print(f"\nTesting n_bins={self.n_bins}")
        self.run_tests() # Run all the earlier tests with new n_bins value
        print(f"Finished testing n_bins={self.n_bins}")
        self.n_bins = original_n_bins
    
    def test_decimal_expression(self):
        original_expr = self.expression
        self.expression = pd.DataFrame(
            [[0.9, 10.2, 0, 0, 0, 0, 1.3, 0],
            [0, 0, 0, 0, 0, 0, 7.654, 0],
            [0, 0, 4.33, 4.33, 0, 0, 4.33, 0],
            [0, 0, 1.873, 0, 0, 0, 3.273, 0],
            [2.341, 2.341, 2.341, 2.341, 2.341, 2.341, 2.341, 2.341],
            [0.1, 0.2, 1.873, 0.4, 0, 8, 3.273, 0]],
            index=['Cell1', 'Cell2', 'Cell3', 'Cell4', 'Cell5', 'Cell6'], 
            columns=['GeneA', 'GeneB', 'GeneC', 'GeneD', 'GeneE', 'GeneF', 'GeneG', 'GeneH']
        )
        print(f"\nTesting decimal expression...")
        self.run_tests()
        self.run_bin_tests()
        print(f"\nFinished testing decimal expression")
        self.expression = original_expr


if __name__ == '__main__':
    unittest.main()
