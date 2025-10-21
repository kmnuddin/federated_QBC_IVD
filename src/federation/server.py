class FedServer:
    def __init__(self):
        self.global_trees = []

    def set_forest(self, trees):
        self.global_trees = list(trees)

    def ingest_replace(self, trees):
        self.set_forest(trees)

    def broadcast_forest(self, max_trees=None):
        if max_trees is None or max_trees >= len(self.global_trees):
            return list(self.global_trees)
        return list(self.global_trees[:max_trees])
