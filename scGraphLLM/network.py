import pandas as pd
from scGraphLLM._globals import *


class RegulatoryNetwork(object):
    """
    Represents a gene regulatory network as a dataframe of directed edges 
    between regulators and targets, with associated weights and likelihoods.

    Attributes:
        df (pd.DataFrame): Internal representation of the network.
        genes (set): Genes that appear both as regulators and targets.

    Column naming follows constants from `_globals.py`:
        REG_VALS, TAR_VALS, WT_VALS, LOGP_VALS
    """
    reg_name = REG_VALS
    tar_name = TAR_VALS
    wt_name = WT_VALS
    lik_name = LOGP_VALS

    def __init__(self, regulators, targets, weights, likelihoods):
        """
        Initializes the RegulatoryNetwork object.

        Args:
            regulators (list): List of regulator gene names.
            targets (list): List of target gene names.
            weights (list): List of edge weights (e.g. MI values).
            likelihoods (list): List of statistical confidences (e.g. log p-values).
        """
        self.df = pd.DataFrame({
            self.reg_name: regulators,
            self.tar_name: targets,
            self.wt_name: weights,
            self.lik_name: likelihoods
        }).astype({
            self.wt_name: float,
            self.lik_name: float
        })
        self.genes = set(self.regulators) | set(self.targets)

    def __len__(self):
        return len(self.df)

    @property
    def regulators(self):
        return self.df[self.reg_name]
    
    @property
    def targets(self):
        return self.df[self.tar_name]

    @property
    def weights(self):
        return self.df[self.wt_name]
    
    @property
    def likelihoods(self):
        return self.df[self.lik_name]

    @property
    def edges(self):
        return list(zip(self.regulators, self.targets))

    @classmethod
    def from_edge_ids(edge_ids, all_edges, weights, likelihoods):
        edges = list(zip(*all_edges[edge_ids]))
        regulators, targets = edges if len(edges) > 0 else [],[]
        return RegulatoryNetwork(regulators, targets, weights, likelihoods)

    @classmethod
    def from_csv(cls, path, reg_name="regulator.values", tar_name="target.values", wt_name="mi.values", lik_name="log.p.values", **kwargs):
        """
        Instantiates a RegulatoryNetwork from a CSV file.

        Args:
            path (str): Path to CSV file.
            reg_name (str): Column name for regulators.
            tar_name (str): Column name for targets.
            wt_name (str): Column name for weights.
            lik_name (str): Column name for likelihoods.
            **kwargs: Additional arguments passed to `pd.read_csv`.

        Returns:
            RegulatoryNetwork: A new instance based on the CSV content.
        """
        df = pd.read_csv(path, **kwargs)    
        return cls(df[reg_name], df[tar_name], df[wt_name], df[lik_name])

    def prune(
        self,
        limit_regulon=None,
        limit_graph=None,
        inplace=False
    ) -> "RegulatoryNetwork":
        """
        Prunes the network to reduce the number of edges.

        Args:
            limit_regulon (int, optional): Max number of targets per regulator.
            limit_graph (int, optional): Max number of total edges (by weight).
            inplace (bool): If True, modifies this object. Otherwise returns a new instance.

        Returns:
            RegulatoryNetwork: The pruned network (self or new).
        """
        df = self.df if inplace else self.df.copy()

        if limit_regulon is not None:
            df = df.groupby(self.reg_name, group_keys=False)\
                .apply(lambda grp: grp.nlargest(limit_regulon, self.wt_name))\
                .reset_index(drop=True)

        if limit_graph is not None:
            df = df.nlargest(limit_graph, self.wt_name)
        
        if inplace:
            self.df = df
            return self

        return RegulatoryNetwork(
            regulators=df[self.reg_name].tolist(),
            targets=df[self.tar_name].tolist(),
            weights=df[self.wt_name].tolist(),
            likelihoods=df[self.lik_name].tolist()
        )
    
    def make_undirected(self, drop_unpaired=False, inplace=False) -> "RegulatoryNetwork":
        """
        Convert the network to an undirected form:
        - Adds reverse edges if missing (default).
        - Drops unpaired edges if drop_unpaired=True.

        Parameters:
            drop_unpaired (bool): If True, only keep edges with reverse counterparts.
            inplace (bool): If True, modifies self. Else, returns a new RegulatoryNetwork.

        Returns:
            RegulatoryNetwork: Updated network object (self or new).
        """
        df = self.df if inplace else self.df.copy()
        edge_set = set(zip(df[self.reg_name], df[self.tar_name]))
        reverse_set = set((t, r) for r, t in edge_set)

        if drop_unpaired:
            # Keep only edges that have a reverse
            bidirectional_edges = edge_set & reverse_set
            mask = [(r, t) in bidirectional_edges for r, t in zip(df[self.reg_name], df[self.tar_name])]
            df = df[mask].reset_index(drop=True)
        else:
            existing_edges = set(zip(df[self.reg_name], df[self.tar_name]))
            reversed_edges = []
            for _, row in df.iterrows():
                src, tgt = row[self.reg_name], row[self.tar_name]
                if (tgt, src) not in existing_edges:
                    reversed_edges.append({
                        self.reg_name: tgt,
                        self.tar_name: src,
                        self.wt_name: row[self.wt_name],
                        self.lik_name: row[self.lik_name]
                    })
            if reversed_edges:
                reversed_df = pd.DataFrame(reversed_edges)
                df = pd.concat([df, reversed_df], ignore_index=True)

        if inplace:
            self.df = df
            return self

        return RegulatoryNetwork(
            regulators=df[self.reg_name].tolist(),
            targets=df[self.tar_name].tolist(),
            weights=df[self.wt_name].tolist(),
            likelihoods=df[self.lik_name].tolist()
        )

    def sort(self, by=None, ascending=True):
        """
        Sort the regulatory network edges by specified column(s).

        If no columns are provided, the default is to sort first by regulator, then by target.

        Args:
            by (Union[str, List[str]], optional): Column name or list of column names to sort by.
                If None, defaults to [self.reg_name, self.tar_name].
            ascending (Union[bool, List[bool]], optional): Sort order.
                True for ascending, False for descending.
                Can also be a list of booleans corresponding to the `by` list.
                Defaults to True.

        Returns:
            RegulatoryNetwork: The sorted network (self), allowing method chaining.
        """
        if by is None:
            by = [self.reg_name, self.tar_name]
        self.df = self.df.sort_values(by=by, ascending=ascending)
        self.df = self.df.reset_index()
        return self
    
    def __eq__(self, other):
        if not isinstance(other, RegulatoryNetwork):
            return NotImplemented
    
        # Sort edges by regulator and target in both networks
        self_edges = self.df[[self.reg_name, self.tar_name]].sort_values(
            by=[self.reg_name, self.tar_name]
        ).reset_index(drop=True)
        
        other_edges = other.df[[other.reg_name, other.tar_name]].sort_values(
            by=[other.reg_name, other.tar_name]
        ).reset_index(drop=True)
        
        # Compare the edge DataFrames for equality
        return self_edges.equals(other_edges)
    