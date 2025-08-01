import pandas as pd
import numpy as np
from typing import Union
from scGraphLLM._globals import *


class RegulatoryNetwork(object):
    """
    Represents a gene regulatory network as a dataframe of directed edges 
    between regulators and targets, with associated weights and likelihoods.

    Attributes:
        df (pd.DataFrame): Internal representation of the network.
        genes (set): Genes that appe
        ar both as regulators and targets.

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

    def __len__(self):
        return len(self.df)
    
    @property
    def df(self) -> pd.DataFrame:
        return self._df
    
    @df.setter
    def df(self, df: pd.DataFrame):
        self._df = df
        self.genes = set(self.regulators) | set(self.targets)

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

    def __str__(self):
        num_edges = len(self.df)
        num_genes = len(self.genes)
        targets_per_regulon = self.regulators.value_counts()
        num_regulons = len(targets_per_regulon)
        median_targets_per_regulon = int(targets_per_regulon.median()) if num_regulons > 0 else 0

        top_regulators = targets_per_regulon.head(3)
        top_reg_str = ', '.join(f"{reg} ({count})" for reg, count in top_regulators.items())

        return (
            f"RegulatoryNetwork with {num_edges:,} edges between {num_genes:,} genes.\n"
            f"Number of regulons: {num_regulons:,}\n"
            f"Median targets per regulon: {median_targets_per_regulon}\n"
            f"Top regulators (by out-degree): {top_reg_str if top_reg_str else 'N/A'}"
        )
    
    def __repr__(self):
        return self.__str__()

    def targets_of(self, regulator):
        return self.targets[self.regulators == regulator].tolist()

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
            df = df.sort_values(by=[self.reg_name, self.wt_name], ascending=[True, False])\
                .groupby(self.reg_name, group_keys=False)\
                .head(limit_regulon)

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

    def retain(self, which: Union[pd.Series, np.ndarray], inplace=True) -> "RegulatoryNetwork":
        """
        Retains only edges where the corresponding entry in `which` is True.

        Args:
            which (pd.Series or np.ndarray): Boolean mask indicating which edges to retain.
                                             Must be same length as self.df.
            inplace (bool): If True, modifies this object. Otherwise returns a new instance.

        Returns:
            RegulatoryNetwork: The retained network (self or new).
        """
        if len(which) != len(self.df):
            raise ValueError("Length of mask does not match number of edges in network.")
        
        new_df = self.df[which].reset_index(drop=True)

        if inplace:
            self.df = new_df
            return self

        return RegulatoryNetwork(
            regulators=new_df[self.reg_name].tolist(),
            targets=new_df[self.tar_name].tolist(),
            weights=new_df[self.wt_name].tolist(),
            likelihoods=new_df[self.lik_name].tolist()
        )
    
    def filter(self, which: Union[pd.Series, np.ndarray], inplace=True) -> "RegulatoryNetwork":
        """
        Filters out edges where the corresponding entry in `which` is True.

        Args:
            which (pd.Series or np.ndarray): Boolean mask indicating which edges to filter out.
                                             Must be same length as self.df.
            inplace (bool): If True, modifies this object. Otherwise returns a new instance.

        Returns:
            RegulatoryNetwork: The filtered network (self or new).
        """
        return self.retain(~which, inplace=inplace)

    
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