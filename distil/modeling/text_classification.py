import numpy as np

from sklearn.svm import LinearSVC
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import DistilBaseModel
from .metrics import metrics, classification_metrics, translate_d3m_metric

import logging

logger = logging.getLogger(__name__)

class TextClassifierCV(DistilBaseModel):
    

    def __init__(self, target_metric, param_grid=''):
        assert target_metric in classification_metrics

        self.param_grid = {
            "vect__ngram_range": [(1, 1), (1, 2)],
            "vect__max_features": [30000, ],
            "cls__C": [float(xx) for xx in np.logspace(-3, 1, 1000)],
            "cls__class_weight": ['balanced', None],
        }
        self.target_metric = target_metric
        self.scoring       = translate_d3m_metric(target_metric)
        self.n_jobs        = 32

        self.n_iter  = 256
        self.n_folds = 5
        self.n_runs  = 1
    
    def fit(self, X_train, y_train, U_train=None):
        #X_train = np.array([text[0] for text in X_train['filename']]) # TODO: move this up

        self.model = RandomizedSearchCV(
            Pipeline([
                ('vect', TfidfVectorizer()),
                ('cls', LinearSVC())
            ]),
            n_iter=self.n_iter,
            param_distributions=self.param_grid,
            cv= RepeatedStratifiedKFold(n_splits=self.n_folds, n_repeats=self.n_runs),
            scoring=self.scoring,
            iid=False,
            n_jobs=self.n_jobs,
            verbose=1
        )
        
        self.model = self.model.fit(X_train, y_train)
        
        self.best_params  = self.model.best_params_
        self.best_fitness = self.model.cv_results_['mean_test_score'].max()
        
        return self
    
    def predict(self, X):
        #X = np.array([text[0] for text in X['filename']])
        return self.model.predict(X)
    
    @property
    def details(self):
        return {
            "best_params"  : self.best_params,
            "best_fitness" : self.best_fitness,
        }
