# src/io/dataset.py
import pandas as pd

class Table:
    def __init__(self, csv_path, id_col, label_col, image_col='axial_path', emb_col=None):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.id_col = id_col
        self.label_col = label_col
        self.image_col = image_col
        self.emb_col = emb_col if (emb_col and emb_col in self.df.columns) else None

    @property
    def n(self): return len(self.df)
    def ids(self): return self.df[self.id_col].values
    def labels(self): return self.df[self.label_col].values
    def image_paths(self): return self.df[self.image_col].astype(str).values
    def embedding_paths(self):
        if self.emb_col is None: return None
        return self.df[self.emb_col].astype(str).values

# Backward-compatible alias for sim_runner
class EmbeddingTable(Table):
    pass

