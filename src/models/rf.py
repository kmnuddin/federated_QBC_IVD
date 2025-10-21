from sklearn.ensemble import RandomForestClassifier

class RFModel:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self._clf = None

    def fit(self, X, y):
        self._clf = RandomForestClassifier(**self.kwargs).fit(X, y)
        return self

    def predict(self, X): return self._clf.predict(X)
    def predict_proba(self, X): return self._clf.predict_proba(X)
    @property
    def classes_(self): return self._clf.classes_
    @property
    def trees(self): return list(self._clf.estimators_)
