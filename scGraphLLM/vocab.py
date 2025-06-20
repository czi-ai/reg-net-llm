import pandas as pd

from scGraphLLM._globals import CLS_GENE, PAD_GENE, MASK_GENE

class GeneVocab(object):
    """
    A class that maps gene names to node indices and vice versa.

    This is used to maintain a consistent vocabulary of genes for models
    that operate on gene graphs or expression data.

    Args:
        genes (list): List of genes and tokens in vocabulary
        nodes (list): List of integer ids corresponding to gene tokens
        require_special_tokens (bool): whether to require CLS, PAD and MASK tokens to 
            be in the vocabulary.

    Attributes:
        genes (list): List of gene names.
        nodes (list): Corresponding list of node indices.
        gene_to_node (dict): Mapping from gene name to node index.
        node_to_gene (dict): Mapping from node index to gene name.
    """
    cls_gene = CLS_GENE
    pad_gene = PAD_GENE
    mask_gene = MASK_GENE
    def __init__(self, genes, nodes, require_special_tokens=True):
        self.genes = genes
        self.nodes = nodes
        self.gene_to_node = dict(zip(genes, nodes))
        self.node_to_gene = dict(zip(nodes, genes))

        if len(genes) != len(nodes):
            raise ValueError("Genes and nodes must have the same length.")
        
        if len(self.gene_to_node) != len(genes) or len(self.node_to_gene) != len(nodes):
            raise ValueError("Relationship between genes and nodes is not one-to-one.")
        
        if require_special_tokens:
            special_tokens = [CLS_GENE, PAD_GENE, MASK_GENE]
            missing = [t for t in special_tokens if t not in self.gene_to_node]
            if missing:
                raise ValueError(f"Missing required special tokens: {missing}")
    
    @property
    def cls_node(self):
        return self.gene_to_node[self.cls_gene]
    
    @property
    def pad_node(self):
        return self.gene_to_node[self.pad_gene]
    
    @property
    def mask_node(self):
        return self.gene_to_node[self.mask_gene]

    @classmethod
    def from_csv(cls, path, gene_col="gene_name", node_col="idx", require_special_tokens=True, **kwargs):
        """
        Loads a GeneVocab from a CSV file.

        The CSV must contain at least two columns: one for gene names and one for node indices.

        Args:
            path (str): Path to the CSV file.
            gene_col (str): Column name for gene names. Defaults to "gene_name".
            node_col (str): Column name for node indices. Defaults to "idx".
            **kwargs: Additional keyword arguments passed to `pandas.read_csv`.

        Returns:
            GeneVocab: An instance of the GeneVocab class.

        Raises:
            ValueError: If the specified columns are not found in the CSV.
        """
        df = pd.read_csv(path, **kwargs)
        if gene_col not in df.columns or node_col not in df.columns:
            raise ValueError(f"Expected columns '{gene_col}' and '{node_col}' not found in CSV.")
        genes = df[gene_col].tolist()
        nodes = df[node_col].tolist()
        return cls(genes, nodes, require_special_tokens)