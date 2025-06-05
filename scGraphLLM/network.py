import pandas as pd
from scGraphLLM._globals import *


class RegulatoryNetwork(object):
    reg_name = REG_VALS
    tar_name = TAR_VALS
    wt_name = WT_VALS
    lik_name = LOGP_VALS
    def __init__(self, regulators, targets, weights, likelihoods):
        self.df = pd.DataFrame({
            self.reg_name: regulators,
            self.tar_name: targets,
            self.wt_name: weights,
            self.lik_name: likelihoods
        })
        self.genes = set(self.regulators) & set(self.targets)
    
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
    def from_csv(cls, path, reg_name="regulator.values", tar_name="target.values", wt_name="mi.values", lik_name="log.p.values", **kwargs):
        df = pd.read_csv(path, **kwargs)    
        return cls(df[reg_name], df[tar_name], df[wt_name], df[lik_name])

    def prune(
        self,
        limit_regulon=None,
        limit_graph=None,
        inplace=False
    ) -> "RegulatoryNetwork":
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
