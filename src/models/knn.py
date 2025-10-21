from sklearn.neighbors import KNeighborsClassifier

class KNNModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._clf = None

    def fit(self, X, y):
        self._clf = KNeighborsClassifier(**self.kwargs).fit(X, y)
        return self

    def predict_proba(self, X): return self._clf.predict_proba(X)
    @property
    def classes_(self): return self._clf.classes_
