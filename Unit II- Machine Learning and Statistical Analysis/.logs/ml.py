
simple_cols = ['BEDCERT', 'RESTOT', 'INHOSP', 'CCRC_FACIL', 'SFF', 'CHOW_LAST_12MOS', 'SPRINKLER_STATUS', 'EXP_TOTAL', 'ADJ_TOTAL']

class ColumnSelectTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.columns]
        
simple_features = Pipeline([
    ('cst', ColumnSelectTransformer(simple_cols)),
])
from sklearn.impute import SimpleImputer
simple_cols = ['BEDCERT', 'RESTOT', 'INHOSP', 'CCRC_FACIL', 'SFF', 'CHOW_LAST_12MOS', 'SPRINKLER_STATUS', 'EXP_TOTAL', 'ADJ_TOTAL']

class ColumnSelectTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.columns]
        
simple_features = Pipeline([
    ('cst', ColumnSelectTransformer(simple_cols)),
])
from sklearn.impute import SimpleImputer

simple_cols = ['BEDCERT', 'RESTOT', 'INHOSP', 'CCRC_FACIL', 'SFF', 'CHOW_LAST_12MOS', 'SPRINKLER_STATUS', 'EXP_TOTAL', 'ADJ_TOTAL']

class ColumnSelectTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.columns]
        
simple_features = Pipeline([
    ('cst', ColumnSelectTransformer(simple_cols)),
])
%logstop
%logstart -rtq ~/.logs/ml.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Wed, 24 Mar 2021 17:07:16
from static_grader import grader# Wed, 24 Mar 2021 17:07:16
%%bash
mkdir data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/providers-train.csv -nc -P ./ml-data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/providers-metadata.csv -nc -P ./ml-data# Wed, 24 Mar 2021 17:07:16
import numpy as np
import pandas as pd# Wed, 24 Mar 2021 17:07:16
metadata = pd.read_csv('./ml-data/providers-metadata.csv')
metadata.head()# Wed, 24 Mar 2021 17:07:16
data = pd.read_csv('./ml-data/providers-train.csv', encoding='latin1')

fine_counts = data.pop('FINE_CNT')
fine_totals = data.pop('FINE_TOT')
cycle_2_score = data.pop('CYCLE_2_TOTAL_SCORE')# Wed, 24 Mar 2021 17:07:16
data.head()# Wed, 24 Mar 2021 17:07:16
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

class GroupMeanEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, gb_col):
        self.gb_col = gb_col
        self.group_averages = {}
        self.global_avg = 0

    def fit(self, X, y):
        # Use self.group_averages to store the average penalty by group
        self.group_averages = y.groupby(X[self.gb_col]).mean().to_dict()
        self.global_avg = y.mean()
        return self

    def predict(self, X):
        if not isinstance (X, pd.DataFrame):
            X = pd.DataFrame(X)
        # Return a list of predicted penalties based on group of samples in X
        return [self.group_averages.get(row, self.global_avg) 
                for row in X[self.gb_col]]# Wed, 24 Mar 2021 17:07:17
from sklearn.pipeline import Pipeline

state_model = Pipeline([
    ('sme', GroupMeanEstimator(gb_col='STATE'))
    ])
state_model.fit(data, fine_totals)# Wed, 24 Mar 2021 17:07:17
state_model.predict(data.sample(5))# Wed, 24 Mar 2021 17:07:17
state_model.predict(pd.DataFrame([{'STATE': 'AS'}]))# Wed, 24 Mar 2021 17:07:17
grader.score.ml__state_model(state_model.predict)# Wed, 24 Mar 2021 17:07:25
from sklearn.impute import SimpleImputer

simple_cols = ['BEDCERT', 'RESTOT', 'INHOSP', 'CCRC_FACIL', 'SFF', 'CHOW_LAST_12MOS', 'SPRINKLER_STATUS', 'EXP_TOTAL', 'ADJ_TOTAL']

class ColumnSelectTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.columns]
        
simple_features = Pipeline([
    ('cst', ColumnSelectTransformer(simple_cols)),
])# Wed, 24 Mar 2021 17:08:04
assert data['RESTOT'].isnull().sum() > 0
assert not np.isnan(simple_features.fit_transform(data)).any()# Wed, 24 Mar 2021 17:09:40
pd.DataFrame(simple_features.fit_transform(data)).info()# Wed, 24 Mar 2021 17:12:34
assert data['RESTOT'].isnull().sum() > 0
assert not np.isnan(simple_features.fit_transform(data)).any()# Wed, 24 Mar 2021 17:15:41
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV# Wed, 24 Mar 2021 17:33:03
simple_features_model = Pipeline([
    ('simple', simple_features),
    # add your estimator here
    ('predictor', RandomForestClassifier(n_estimators = 50, max_depth = 5 ))
])

param_grid = {'predictor__n_estimators' : range(25, 126, 25),
              'predictor__max_depth' : range(6, 17, 2)}

sfm_gs = GridSearchCV(simple_features_model, 
                     param_grid = param_grid, n_jobs = -1, verbose = 1)

sfm_gs.fit(data, fine_counts > 0)# Wed, 24 Mar 2021 17:33:12
simple_features_model = Pipeline([
    ('simple', simple_features),
    # add your estimator here
    ('predictor', RandomForestClassifier(n_estimators = 50, max_depth = 5 ))
])

param_grid = {'predictor__n_estimators' : range(25, 126, 25),
              'predictor__max_depth' : range(6, 17, 2)}

sfm_gs = GridSearchCV(simple_features_model, 
                     param_grid = param_grid, n_jobs = -1, verbose = 1)

sfm_gs.fit(data, fine_counts > 0);# Wed, 24 Mar 2021 17:33:48
simple_features_model = Pipeline([
    ('simple', simple_features),
    # add your estimator here
    ('predictor', RandomForestClassifier(n_estimators=50, max_depth = 5))
])

param_grid = {'predictor__n_estimators': range(25, 126, 25),
              'predictor__max_depth': range(6, 17, 2)}

sfm_gs = GridSearchCV(simple_features_model, 
                      param_grid = param_grid, n_jobs = -1, verbose = 1)

sfm_gs.fit(data, fine_counts > 0);# Wed, 24 Mar 2021 17:34:40
assert data['RESTOT'].isnull().sum() > 0
assert not np.isnan(simple_features.fit_transform(data)).any()# Wed, 24 Mar 2021 17:35:24
from sklearn.impute import SimpleImputer

simple_cols = ['BEDCERT', 'RESTOT', 'INHOSP', 'CCRC_FACIL', 'SFF', 'CHOW_LAST_12MOS', 'SPRINKLER_STATUS', 'EXP_TOTAL', 'ADJ_TOTAL']

class ColumnSelectTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.columns]
        
simple_features = Pipeline([
    ('cst', ColumnSelectTransformer(simple_cols)),
    ('imputer', SimpleImputer())
])# Wed, 24 Mar 2021 17:35:25
pd.DataFrame(simple_features.fit_transform(data)).info()# Wed, 24 Mar 2021 17:35:33
assert data['RESTOT'].isnull().sum() > 0
assert not np.isnan(simple_features.fit_transform(data)).any()# Wed, 24 Mar 2021 17:35:36
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV# Wed, 24 Mar 2021 17:35:37
simple_features_model = Pipeline([
    ('simple', simple_features),
    # add your estimator here
    ('predictor', RandomForestClassifier(n_estimators=50, max_depth = 5))
])

param_grid = {'predictor__n_estimators': range(25, 126, 25),
              'predictor__max_depth': range(6, 17, 2)}

sfm_gs = GridSearchCV(simple_features_model, 
                      param_grid = param_grid, n_jobs = -1, verbose = 1)

sfm_gs.fit(data, fine_counts > 0);# Wed, 24 Mar 2021 17:36:59
simple_features_model.fit(data, fine_counts > 0)# Wed, 24 Mar 2021 17:37:00
def positive_probability(model):
    def predict_proba(X):
        return model.predict_proba(X)[:, 1]
    return predict_proba

grader.score.ml__simple_features(positive_probability(simple_features_model))# Wed, 24 Mar 2021 17:37:00
simple_features_model.fit(data, fine_counts > 0)# Wed, 24 Mar 2021 17:37:01
def positive_probability(model):
    def predict_proba(X):
        return model.predict_proba(X)[:, 1]
    return predict_proba

grader.score.ml__simple_features(positive_probability(simple_features_model))# Wed, 24 Mar 2021 17:38:59
simple_features_model.fit(data, fine_counts > 0)# Wed, 24 Mar 2021 17:38:59
def positive_probability(model):
    def predict_proba(X):
        return model.predict_proba(X)[:, 1]
    return predict_proba

grader.score.ml__simple_features(positive_probability(simple_features_model))# Wed, 24 Mar 2021 17:39:09
simple_features_model = Pipeline([
    ('simple', simple_features),
    # add your estimator here
    ('predictor', RandomForestClassifier(n_estimators=50, max_depth = 5))
])

param_grid = {'predictor__n_estimators': range(25, 126, 25),
              'predictor__max_depth': range(6, 17, 2)}

sfm_gs = GridSearchCV(simple_features_model, 
                      param_grid = param_grid, n_jobs = -1, verbose = 1)

sfm_gs.fit(data, fine_counts > 0);# Wed, 24 Mar 2021 17:40:07
simple_features_model = Pipeline([
    ('simple', simple_features),
    # add your estimator here
    ('predictor', RandomForestClassifier(n_estimators=50, max_depth = 5))
])

param_grid = {'predictor__n_estimators': range(25, 126, 25),
              'predictor__max_depth': range(6, 17, 2)}

sfm_gs = GridSearchCV(simple_features_model, 
                      param_grid = param_grid, n_jobs = -1, verbose = 1)

sfm_gs.fit(data, fine_counts > 0);# Wed, 24 Mar 2021 17:41:37
simple_features_model = Pipeline([
    ('simple', simple_features),
    # add your estimator here
    ('predictor', RandomForestClassifier(n_estimators=50, max_depth = 5))
])

param_grid = {'predictor__n_estimators': range(25, 126, 25),
              'predictor__max_depth': range(6, 17, 2)}

sfm_gs = GridSearchCV(simple_features_model, 
                      param_grid = param_grid, n_jobs = -1, verbose = 1)

sfm_gs.fit(data, fine_counts > 0);# Wed, 24 Mar 2021 17:43:00
simple_features_model.fit(data, fine_counts > 0)# Wed, 24 Mar 2021 17:43:01
def positive_probability(model):
    def predict_proba(X):
        return model.predict_proba(X)[:, 1]
    return predict_proba

grader.score.ml__simple_features(positive_probability(simple_features_model))# Wed, 24 Mar 2021 17:44:37
from sklearn.pipeline import FeatureUnion

owner_onehot = Pipeline([
    ('cst', ColumnSelectTransformer(['OWNERSHIP'])),
    ('ohe', OneHotEncoder (categories = 'auto', sparse = False))
])

cert_onehot = Pipeline([
    ('cst', ColumnSelectTransformer(['CERTIFICATION'])),
    ('ohe', OneHotEncoder (categories = 'auto', sparse = False))
])

categorical_features = FeatureUnion([
    ('owner', owner_onehot),
    ('cert', cert_onehot)
])# Wed, 24 Mar 2021 17:45:15
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder

owner_onehot = Pipeline([
    ('cst', ColumnSelectTransformer(['OWNERSHIP'])),
    ('ohe', OneHotEncoder (categories = 'auto', sparse = False))
])

cert_onehot = Pipeline([
    ('cst', ColumnSelectTransformer(['CERTIFICATION'])),
    ('ohe', OneHotEncoder (categories = 'auto', sparse = False))
])

categorical_features = FeatureUnion([
    ('owner', owner_onehot),
    ('cert', cert_onehot)
])# Wed, 24 Mar 2021 17:45:29
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder

owner_onehot = Pipeline([
    ('cst', ColumnSelectTransformer(['OWNERSHIP'])),
    ('ohe', OneHotEncoder (categories = 'auto', sparse = False))
])

cert_onehot = Pipeline([
    ('cst', ColumnSelectTransformer(['CERTIFICATION'])),
    ('ohe', OneHotEncoder (categories = 'auto', sparse = False))
])

categorical_features = FeatureUnion([
    ('owner', owner_onehot),
    ('cert', cert_onehot)
])# Wed, 24 Mar 2021 17:45:29
assert categorical_features.fit_transform(data).shape[0] == data.shape[0]
assert categorical_features.fit_transform(data).dtype == np.float64
assert not np.isnan(categorical_features.fit_transform(data)).any()# Wed, 24 Mar 2021 17:46:32
categorical_features_model = Pipeline([
    ('categorical', categorical_features),
    # add your estimator here
    ('classifier', RandomForestClassifier())
])# Wed, 24 Mar 2021 17:46:37
categorical_features_model.fit(data, fine_counts > 0)# Wed, 24 Mar 2021 17:47:32
grader.score.ml__categorical_features(positive_probability(categorical_features_model))# Wed, 24 Mar 2021 17:48:30
business_features = FeatureUnion([
    ('simple', simple_features),
    ('categorical', categorical_features)
])# Wed, 24 Mar 2021 17:49:03
from sklearn.preprocessing  import PolynomialFeatures
from sklearn.linear_model import LogisticRegression# Wed, 24 Mar 2021 17:49:04
business_features = FeatureUnion([
    ('simple', simple_features),
    ('categorical', categorical_features)
])# Wed, 24 Mar 2021 17:50:36

business_model = Pipeline([
    ('features', business_features),
    # add your estimator here
    ('ploy', PolynomialFeatures(2)),
    ('lr', LogisticRegression())
])# Wed, 24 Mar 2021 17:50:38

business_model = Pipeline([
    ('features', business_features),
    # add your estimator here
    ('ploy', PolynomialFeatures(2)),
    ('lr', LogisticRegression())
])# Wed, 24 Mar 2021 17:50:47
business_model.fit(data, fine_counts > 0)# Wed, 24 Mar 2021 17:50:48
grader.score.ml__business_model(positive_probability(business_model))# Wed, 24 Mar 2021 17:51:03

business_model = Pipeline([
    ('features', business_features),
    # add your estimator here
    ('ploy', PolynomialFeatures(5)),
    ('lr', LogisticRegression())
])# Wed, 24 Mar 2021 17:51:04
business_model.fit(data, fine_counts > 0)%logstop
%logstart -rtq ~/.logs/ml.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Wed, 24 Mar 2021 17:54:18
from static_grader import grader# Wed, 24 Mar 2021 17:54:18
%%bash
mkdir data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/providers-train.csv -nc -P ./ml-data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/providers-metadata.csv -nc -P ./ml-data# Wed, 24 Mar 2021 17:54:18
import numpy as np
import pandas as pd# Wed, 24 Mar 2021 17:54:18
metadata = pd.read_csv('./ml-data/providers-metadata.csv')
metadata.head()# Wed, 24 Mar 2021 17:54:18
data = pd.read_csv('./ml-data/providers-train.csv', encoding='latin1')

fine_counts = data.pop('FINE_CNT')
fine_totals = data.pop('FINE_TOT')
cycle_2_score = data.pop('CYCLE_2_TOTAL_SCORE')# Wed, 24 Mar 2021 17:54:18
data.head()# Wed, 24 Mar 2021 17:54:18
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

class GroupMeanEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, gb_col):
        self.gb_col = gb_col
        self.group_averages = {}
        self.global_avg = 0

    def fit(self, X, y):
        # Use self.group_averages to store the average penalty by group
        self.group_averages = y.groupby(X[self.gb_col]).mean().to_dict()
        self.global_avg = y.mean()
        return self

    def predict(self, X):
        if not isinstance (X, pd.DataFrame):
            X = pd.DataFrame(X)
        # Return a list of predicted penalties based on group of samples in X
        return [self.group_averages.get(row, self.global_avg) 
                for row in X[self.gb_col]]# Wed, 24 Mar 2021 17:54:18
from sklearn.pipeline import Pipeline

state_model = Pipeline([
    ('sme', GroupMeanEstimator(gb_col='STATE'))
    ])
state_model.fit(data, fine_totals)# Wed, 24 Mar 2021 17:54:18
state_model.predict(data.sample(5))# Wed, 24 Mar 2021 17:54:18
state_model.predict(pd.DataFrame([{'STATE': 'AS'}]))# Wed, 24 Mar 2021 17:54:18
grader.score.ml__state_model(state_model.predict)# Wed, 24 Mar 2021 17:54:18
from sklearn.impute import SimpleImputer

simple_cols = ['BEDCERT', 'RESTOT', 'INHOSP', 'CCRC_FACIL', 'SFF', 'CHOW_LAST_12MOS', 'SPRINKLER_STATUS', 'EXP_TOTAL', 'ADJ_TOTAL']

class ColumnSelectTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.columns]
        
simple_features = Pipeline([
    ('cst', ColumnSelectTransformer(simple_cols)),
    ('imputer', SimpleImputer())
])# Wed, 24 Mar 2021 17:54:18
pd.DataFrame(simple_features.fit_transform(data)).info()# Wed, 24 Mar 2021 17:54:18
assert data['RESTOT'].isnull().sum() > 0
assert not np.isnan(simple_features.fit_transform(data)).any()# Wed, 24 Mar 2021 17:54:18
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV# Wed, 24 Mar 2021 17:54:18
simple_features_model = Pipeline([
    ('simple', simple_features),
    # add your estimator here
    ('predictor', RandomForestClassifier(n_estimators=50, max_depth = 5))
])

param_grid = {'predictor__n_estimators': range(25, 126, 25),
              'predictor__max_depth': range(6, 17, 2)}

sfm_gs = GridSearchCV(simple_features_model, 
                      param_grid = param_grid, n_jobs = -1, verbose = 1)

sfm_gs.fit(data, fine_counts > 0);# Wed, 24 Mar 2021 17:55:42
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV# Wed, 24 Mar 2021 17:55:43
simple_features_model = Pipeline([
    ('simple', simple_features),
    # add your estimator here
    ('predictor', RandomForestClassifier(n_estimators=50, max_depth = 5))
])

param_grid = {'predictor__n_estimators': range(25, 126, 25),
              'predictor__max_depth': range(6, 17, 2)}

sfm_gs = GridSearchCV(simple_features_model, 
                      param_grid = param_grid, n_jobs = -1, verbose = 1)

sfm_gs.fit(data, fine_counts > 0);# Wed, 24 Mar 2021 17:56:26
simple_features_model = Pipeline([
    ('simple', simple_features),
    # add your estimator here
    ('predictor', RandomForestClassifier(n_estimators=50, max_depth = 5))
])

param_grid = {'predictor__n_estimators': range(25, 126, 25),
              'predictor__max_depth': range(6, 17, 2)}

sfm_gs = GridSearchCV(simple_features_model, 
                      param_grid = param_grid, n_jobs = -1, verbose = 1)

sfm_gs.fit(data, fine_counts > 0);# Wed, 24 Mar 2021 17:57:39
simple_features_model = Pipeline([
    ('simple', simple_features),
    # add your estimator here
    ('predictor', RandomForestClassifier(n_estimators=50, max_depth = 5))
])

param_grid = {'predictor__n_estimators': range(25, 126, 25),
              'predictor__max_depth': range(6, 17, 2)}

sfm_gs = GridSearchCV(simple_features_model, 
                      param_grid = param_grid, n_jobs = -1, verbose = 1)

sfm_gs.fit(data, fine_counts > 0);# Wed, 24 Mar 2021 17:59:02
simple_features_model.fit(data, fine_counts > 0)# Wed, 24 Mar 2021 17:59:03
def positive_probability(model):
    def predict_proba(X):
        return model.predict_proba(X)[:, 1]
    return predict_proba

grader.score.ml__simple_features(positive_probability(simple_features_model))# Wed, 24 Mar 2021 17:59:03
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder

owner_onehot = Pipeline([
    ('cst', ColumnSelectTransformer(['OWNERSHIP'])),
    ('ohe', OneHotEncoder (categories = 'auto', sparse = False))
])

cert_onehot = Pipeline([
    ('cst', ColumnSelectTransformer(['CERTIFICATION'])),
    ('ohe', OneHotEncoder (categories = 'auto', sparse = False))
])

categorical_features = FeatureUnion([
    ('owner', owner_onehot),
    ('cert', cert_onehot)
])# Wed, 24 Mar 2021 17:59:03
assert categorical_features.fit_transform(data).shape[0] == data.shape[0]
assert categorical_features.fit_transform(data).dtype == np.float64
assert not np.isnan(categorical_features.fit_transform(data)).any()# Wed, 24 Mar 2021 17:59:03
categorical_features_model = Pipeline([
    ('categorical', categorical_features),
    # add your estimator here
    ('classifier', RandomForestClassifier())
])# Wed, 24 Mar 2021 17:59:03
categorical_features_model.fit(data, fine_counts > 0)# Wed, 24 Mar 2021 17:59:03
grader.score.ml__categorical_features(positive_probability(categorical_features_model))# Wed, 24 Mar 2021 17:59:03

business_model = Pipeline([
    ('features', business_features),
    # add your estimator here
    ('ploy', PolynomialFeatures(2)),
    ('lr', LogisticRegression())
])# Wed, 24 Mar 2021 17:59:15
grader.score.ml__categorical_features(positive_probability(categorical_features_model))# Wed, 24 Mar 2021 17:59:17
from sklearn.preprocessing  import PolynomialFeatures
from sklearn.linear_model import LogisticRegression# Wed, 24 Mar 2021 17:59:18
business_features = FeatureUnion([
    ('simple', simple_features),
    ('categorical', categorical_features)
])# Wed, 24 Mar 2021 17:59:19

business_model = Pipeline([
    ('features', business_features),
    # add your estimator here
    ('ploy', PolynomialFeatures(2)),
    ('lr', LogisticRegression())
])# Wed, 24 Mar 2021 17:59:20
business_model.fit(data, fine_counts > 0)# Wed, 24 Mar 2021 17:59:21
grader.score.ml__business_model(positive_probability(business_model))# Wed, 24 Mar 2021 17:59:47
grader.score.ml__business_model(positive_probability(business_model))# Wed, 24 Mar 2021 18:00:21
grader.score.ml__business_model(positive_probability(business_model))# Wed, 24 Mar 2021 18:04:12
class TimedeltaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, t1_col, t2_col):
        self.t1_col = t1_col
        self.t2_col = t2_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance (X, pd.DataFrame):
            X = pd.DataFrame(X)
        results = (pd.to_datetime(X[self.t1_col]) - pd.to_datetime(X[self.t1_col]))
        
        results = results.apply(lambda X:X.days).values.reshape(-1, 1)
        
        return results# Wed, 24 Mar 2021 18:07:20
cycle_1_date = 'CYCLE_1_SURVEY_DATE'
cycle_2_date = 'CYCLE_2_SURVEY_DATE'
time_feature = TimedeltaTransformer(cycle_1_date, cycle_2_date)# Wed, 24 Mar 2021 18:08:12
cycle_1_cols = ['CYCLE_1_DEFS', 'CYCLE_1_NFROMDEFS', 'CYCLE_1_NFROMCOMP',
                'CYCLE_1_DEFS_SCORE', 'CYCLE_1_NUMREVIS',
                'CYCLE_1_REVISIT_SCORE', 'CYCLE_1_TOTAL_SCORE']
cycle_1_features = ColumnSelectTransformer(cycle_1_cols)# Wed, 24 Mar 2021 18:09:14
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Lasso


survey_model = Pipeline([
    ('features', FeatureUnion([
        ('business', business_features),
        ('survey', cycle_1_features),
        ('time', time_feature)
    ])),
    # add your estimator here
])# Wed, 24 Mar 2021 18:09:19
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Lasso

gs = GridSearchCV(Lasso(max_iter = 1000), param_grid = {'alpha': np.arange(0, 3.5, 0.5)}, cv=5, n_jobs=4, verbose=1)

survey_model = Pipeline([
    ('features', FeatureUnion([
        ('business', business_features),
        ('survey', cycle_1_features),
        ('time', time_feature)
    ])),
    # add your estimator here
])from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Lasso

gs = GridSearchCV(Lasso(max_iter = 1000), param_grid = {'alpha': np.arange(0, 3.5, 0.5)}, cv=5, n_jobs=4, verbose=1)

survey_model = Pipeline([
    ('features', FeatureUnion([
        ('business', business_features),
        ('survey', cycle_1_features),
        ('time', time_feature)
    ])),
    # add your estimator here
    ('poly', PolynomialFeatures(2)),
    ('decomp', TruncatedSVD(40)),
    ('gs', gs)
])
grader.score.ml__survey_model(survey_model.predict)
%logstop
%logstart -rtq ~/.logs/ml.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Wed, 24 Mar 2021 18:18:41
from static_grader import grader# Wed, 24 Mar 2021 18:18:42
%%bash
mkdir data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/providers-train.csv -nc -P ./ml-data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/providers-metadata.csv -nc -P ./ml-data# Wed, 24 Mar 2021 18:18:42
import numpy as np
import pandas as pd# Wed, 24 Mar 2021 18:18:42
metadata = pd.read_csv('./ml-data/providers-metadata.csv')
metadata.head()# Wed, 24 Mar 2021 18:18:42
data = pd.read_csv('./ml-data/providers-train.csv', encoding='latin1')

fine_counts = data.pop('FINE_CNT')
fine_totals = data.pop('FINE_TOT')
cycle_2_score = data.pop('CYCLE_2_TOTAL_SCORE')# Wed, 24 Mar 2021 18:18:42
data.head()# Wed, 24 Mar 2021 18:18:42
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

class GroupMeanEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, gb_col):
        self.gb_col = gb_col
        self.group_averages = {}
        self.global_avg = 0

    def fit(self, X, y):
        # Use self.group_averages to store the average penalty by group
        self.group_averages = y.groupby(X[self.gb_col]).mean().to_dict()
        self.global_avg = y.mean()
        return self

    def predict(self, X):
        if not isinstance (X, pd.DataFrame):
            X = pd.DataFrame(X)
        # Return a list of predicted penalties based on group of samples in X
        return [self.group_averages.get(row, self.global_avg) 
                for row in X[self.gb_col]]# Wed, 24 Mar 2021 18:18:42
from sklearn.pipeline import Pipeline

state_model = Pipeline([
    ('sme', GroupMeanEstimator(gb_col='STATE'))
    ])
state_model.fit(data, fine_totals)# Wed, 24 Mar 2021 18:18:42
state_model.predict(data.sample(5))# Wed, 24 Mar 2021 18:18:42
state_model.predict(pd.DataFrame([{'STATE': 'AS'}]))# Wed, 24 Mar 2021 18:18:42
grader.score.ml__state_model(state_model.predict)# Wed, 24 Mar 2021 18:18:42
from sklearn.impute import SimpleImputer

simple_cols = ['BEDCERT', 'RESTOT', 'INHOSP', 'CCRC_FACIL', 'SFF', 'CHOW_LAST_12MOS', 'SPRINKLER_STATUS', 'EXP_TOTAL', 'ADJ_TOTAL']

class ColumnSelectTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.columns]
        
simple_features = Pipeline([
    ('cst', ColumnSelectTransformer(simple_cols)),
    ('imputer', SimpleImputer())
])# Wed, 24 Mar 2021 18:18:42
pd.DataFrame(simple_features.fit_transform(data)).info()# Wed, 24 Mar 2021 18:18:42
assert data['RESTOT'].isnull().sum() > 0
assert not np.isnan(simple_features.fit_transform(data)).any()# Wed, 24 Mar 2021 18:18:42
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV# Wed, 24 Mar 2021 18:18:42
simple_features_model = Pipeline([
    ('simple', simple_features),
    # add your estimator here
    ('predictor', RandomForestClassifier(n_estimators=50, max_depth = 5))
])

param_grid = {'predictor__n_estimators': range(25, 126, 25),
              'predictor__max_depth': range(6, 17, 2)}

sfm_gs = GridSearchCV(simple_features_model, 
                      param_grid = param_grid, n_jobs = -1, verbose = 1)

sfm_gs.fit(data, fine_counts > 0);%logstop
%logstart -rtq ~/.logs/ml.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Wed, 24 Mar 2021 18:20:44
from static_grader import grader# Wed, 24 Mar 2021 18:20:44
%%bash
mkdir data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/providers-train.csv -nc -P ./ml-data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/providers-metadata.csv -nc -P ./ml-data# Wed, 24 Mar 2021 18:20:44
import numpy as np
import pandas as pd# Wed, 24 Mar 2021 18:20:44
metadata = pd.read_csv('./ml-data/providers-metadata.csv')
metadata.head()# Wed, 24 Mar 2021 18:20:44
data = pd.read_csv('./ml-data/providers-train.csv', encoding='latin1')

fine_counts = data.pop('FINE_CNT')
fine_totals = data.pop('FINE_TOT')
cycle_2_score = data.pop('CYCLE_2_TOTAL_SCORE')# Wed, 24 Mar 2021 18:20:44
data.head()# Wed, 24 Mar 2021 18:20:44
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

class GroupMeanEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, gb_col):
        self.gb_col = gb_col
        self.group_averages = {}
        self.global_avg = 0

    def fit(self, X, y):
        # Use self.group_averages to store the average penalty by group
        self.group_averages = y.groupby(X[self.gb_col]).mean().to_dict()
        self.global_avg = y.mean()
        return self

    def predict(self, X):
        if not isinstance (X, pd.DataFrame):
            X = pd.DataFrame(X)
        # Return a list of predicted penalties based on group of samples in X
        return [self.group_averages.get(row, self.global_avg) 
                for row in X[self.gb_col]]# Wed, 24 Mar 2021 18:20:44
from sklearn.pipeline import Pipeline

state_model = Pipeline([
    ('sme', GroupMeanEstimator(gb_col='STATE'))
    ])
state_model.fit(data, fine_totals)# Wed, 24 Mar 2021 18:20:45
state_model.predict(data.sample(5))# Wed, 24 Mar 2021 18:20:45
state_model.predict(pd.DataFrame([{'STATE': 'AS'}]))# Wed, 24 Mar 2021 18:20:45
grader.score.ml__state_model(state_model.predict)# Wed, 24 Mar 2021 18:20:53
from sklearn.impute import SimpleImputer

simple_cols = ['BEDCERT', 'RESTOT', 'INHOSP', 'CCRC_FACIL', 'SFF', 'CHOW_LAST_12MOS', 'SPRINKLER_STATUS', 'EXP_TOTAL', 'ADJ_TOTAL']

class ColumnSelectTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.columns]
        
simple_features = Pipeline([
    ('cst', ColumnSelectTransformer(simple_cols)),
    ('imputer', SimpleImputer())
])# Wed, 24 Mar 2021 18:20:53
pd.DataFrame(simple_features.fit_transform(data)).info()# Wed, 24 Mar 2021 18:20:53
assert data['RESTOT'].isnull().sum() > 0
assert not np.isnan(simple_features.fit_transform(data)).any()# Wed, 24 Mar 2021 18:21:02
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV# Wed, 24 Mar 2021 18:21:02
simple_features_model = Pipeline([
    ('simple', simple_features),
    # add your estimator here
    ('predictor', RandomForestClassifier(n_estimators=50, max_depth = 5))
])

param_grid = {'predictor__n_estimators': range(25, 126, 25),
              'predictor__max_depth': range(6, 17, 2)}

sfm_gs = GridSearchCV(simple_features_model, 
                      param_grid = param_grid, n_jobs = -1, verbose = 1)

sfm_gs.fit(data, fine_counts > 0);# Wed, 24 Mar 2021 18:22:26
simple_features_model.fit(data, fine_counts > 0)# Wed, 24 Mar 2021 18:22:26
def positive_probability(model):
    def predict_proba(X):
        return model.predict_proba(X)[:, 1]
    return predict_proba

grader.score.ml__simple_features(positive_probability(simple_features_model))# Wed, 24 Mar 2021 18:22:26
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder

owner_onehot = Pipeline([
    ('cst', ColumnSelectTransformer(['OWNERSHIP'])),
    ('ohe', OneHotEncoder (categories = 'auto', sparse = False))
])

cert_onehot = Pipeline([
    ('cst', ColumnSelectTransformer(['CERTIFICATION'])),
    ('ohe', OneHotEncoder (categories = 'auto', sparse = False))
])

categorical_features = FeatureUnion([
    ('owner', owner_onehot),
    ('cert', cert_onehot)
])# Wed, 24 Mar 2021 18:22:27
assert categorical_features.fit_transform(data).shape[0] == data.shape[0]
assert categorical_features.fit_transform(data).dtype == np.float64
assert not np.isnan(categorical_features.fit_transform(data)).any()# Wed, 24 Mar 2021 18:22:27
categorical_features_model = Pipeline([
    ('categorical', categorical_features),
    # add your estimator here
    ('classifier', RandomForestClassifier())
])# Wed, 24 Mar 2021 18:22:27
categorical_features_model.fit(data, fine_counts > 0)# Wed, 24 Mar 2021 18:22:27
grader.score.ml__categorical_features(positive_probability(categorical_features_model))# Wed, 24 Mar 2021 18:22:27
from sklearn.preprocessing  import PolynomialFeatures
from sklearn.linear_model import LogisticRegression# Wed, 24 Mar 2021 18:22:27
business_features = FeatureUnion([
    ('simple', simple_features),
    ('categorical', categorical_features)
])# Wed, 24 Mar 2021 18:22:27

business_model = Pipeline([
    ('features', business_features),
    # add your estimator here
    ('ploy', PolynomialFeatures(2)),
    ('lr', LogisticRegression())
])# Wed, 24 Mar 2021 18:22:27
business_model.fit(data, fine_counts > 0)# Wed, 24 Mar 2021 18:22:28
grader.score.ml__business_model(positive_probability(business_model))# Wed, 24 Mar 2021 18:22:28
class TimedeltaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, t1_col, t2_col):
        self.t1_col = t1_col
        self.t2_col = t2_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance (X, pd.DataFrame):
            X = pd.DataFrame(X)
        results = (pd.to_datetime(X[self.t1_col]) - pd.to_datetime(X[self.t1_col]))
        
        results = results.apply(lambda X:X.days).values.reshape(-1, 1)
        
        return results# Wed, 24 Mar 2021 18:22:28
cycle_1_date = 'CYCLE_1_SURVEY_DATE'
cycle_2_date = 'CYCLE_2_SURVEY_DATE'
time_feature = TimedeltaTransformer(cycle_1_date, cycle_2_date)# Wed, 24 Mar 2021 18:22:28
cycle_1_cols = ['CYCLE_1_DEFS', 'CYCLE_1_NFROMDEFS', 'CYCLE_1_NFROMCOMP',
                'CYCLE_1_DEFS_SCORE', 'CYCLE_1_NUMREVIS',
                'CYCLE_1_REVISIT_SCORE', 'CYCLE_1_TOTAL_SCORE']
cycle_1_features = ColumnSelectTransformer(cycle_1_cols)# Wed, 24 Mar 2021 18:22:28
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Lasso

gs = GridSearchCV(Lasso(max_iter = 1000), param_grid = {'alpha': np.arange(0, 3.5, 0.5)}, cv=5, n_jobs=4, verbose=1)

survey_model = Pipeline([
    ('features', FeatureUnion([
        ('business', business_features),
        ('survey', cycle_1_features),
        ('time', time_feature)
    ])),
    # add your estimator here
    ('poly', PolynomialFeatures(2)),
    ('decomp', TruncatedSVD(40)),
    ('gs', gs)
])# Wed, 24 Mar 2021 18:22:28
survey_model.fit(data, cycle_2_score.astype(int))# Wed, 24 Mar 2021 18:22:41
grader.score.ml__survey_model(survey_model.predict)# Wed, 24 Mar 2021 18:23:53
%logstop
%logstart -rtq ~/.logs/ml.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/ml.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
from static_grader import grader
%%bash
mkdir data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/providers-train.csv -nc -P ./ml-data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/providers-metadata.csv -nc -P ./ml-data
import numpy as np
import pandas as pd
metadata = pd.read_csv('./ml-data/providers-metadata.csv')
metadata.head()
data = pd.read_csv('./ml-data/providers-train.csv', encoding='latin1')

fine_counts = data.pop('FINE_CNT')
fine_totals = data.pop('FINE_TOT')
cycle_2_score = data.pop('CYCLE_2_TOTAL_SCORE')
data.head()
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

class GroupMeanEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, gb_col):
        self.gb_col = gb_col
        self.group_averages = {}
        self.global_avg = 0

    def fit(self, X, y):
        # Use self.group_averages to store the average penalty by group
        self.group_averages = y.groupby(X[self.gb_col]).mean().to_dict()
        self.global_avg = y.mean()
        return self

    def predict(self, X):
        if not isinstance (X, pd.DataFrame):
            X = pd.DataFrame(X)
        # Return a list of predicted penalties based on group of samples in X
        return [self.group_averages.get(row, self.global_avg) 
                for row in X[self.gb_col]]
from sklearn.pipeline import Pipeline

state_model = Pipeline([
    ('sme', GroupMeanEstimator(gb_col='STATE'))
    ])
state_model.fit(data, fine_totals)
state_model.predict(data.sample(5))
state_model.predict(pd.DataFrame([{'STATE': 'AS'}]))
grader.score.ml__state_model(state_model.predict)
from sklearn.impute import SimpleImputer

simple_cols = ['BEDCERT', 'RESTOT', 'INHOSP', 'CCRC_FACIL', 'SFF', 'CHOW_LAST_12MOS', 'SPRINKLER_STATUS', 'EXP_TOTAL', 'ADJ_TOTAL']

class ColumnSelectTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.columns]
        
simple_features = Pipeline([
    ('cst', ColumnSelectTransformer(simple_cols)),
    ('imputer', SimpleImputer())
])
pd.DataFrame(simple_features.fit_transform(data)).info()
assert data['RESTOT'].isnull().sum() > 0
assert not np.isnan(simple_features.fit_transform(data)).any()
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
simple_features_model = Pipeline([
    ('simple', simple_features),
    # add your estimator here
    ('predictor', RandomForestClassifier(n_estimators=50, max_depth = 5))
])

param_grid = {'predictor__n_estimators': range(25, 126, 25),
              'predictor__max_depth': range(6, 17, 2)}

sfm_gs = GridSearchCV(simple_features_model, 
                      param_grid = param_grid, n_jobs = -1, verbose = 1)

sfm_gs.fit(data, fine_counts > 0);
simple_features_model.fit(data, fine_counts > 0)
def positive_probability(model):
    def predict_proba(X):
        return model.predict_proba(X)[:, 1]
    return predict_proba

grader.score.ml__simple_features(positive_probability(simple_features_model))
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder

owner_onehot = Pipeline([
    ('cst', ColumnSelectTransformer(['OWNERSHIP'])),
    ('ohe', OneHotEncoder (categories = 'auto', sparse = False))
])

cert_onehot = Pipeline([
    ('cst', ColumnSelectTransformer(['CERTIFICATION'])),
    ('ohe', OneHotEncoder (categories = 'auto', sparse = False))
])

categorical_features = FeatureUnion([
    ('owner', owner_onehot),
    ('cert', cert_onehot)
])
assert categorical_features.fit_transform(data).shape[0] == data.shape[0]
assert categorical_features.fit_transform(data).dtype == np.float64
assert not np.isnan(categorical_features.fit_transform(data)).any()
categorical_features_model = Pipeline([
    ('categorical', categorical_features),
    # add your estimator here
    ('classifier', RandomForestClassifier())
])
categorical_features_model.fit(data, fine_counts > 0)
grader.score.ml__categorical_features(positive_probability(categorical_features_model))
from sklearn.preprocessing  import PolynomialFeatures
from sklearn.linear_model import LogisticRegression
business_features = FeatureUnion([
    ('simple', simple_features),
    ('categorical', categorical_features)
])

business_model = Pipeline([
    ('features', business_features),
    # add your estimator here
    ('ploy', PolynomialFeatures(2)),
    ('lr', LogisticRegression())
])
business_model.fit(data, fine_counts > 0)
grader.score.ml__business_model(positive_probability(business_model))
class TimedeltaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, t1_col, t2_col):
        self.t1_col = t1_col
        self.t2_col = t2_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance (X, pd.DataFrame):
            X = pd.DataFrame(X)
        results = (pd.to_datetime(X[self.t1_col]) - pd.to_datetime(X[self.t1_col]))
        
        results = results.apply(lambda X:X.days).values.reshape(-1, 1)
        
        return results
cycle_1_date = 'CYCLE_1_SURVEY_DATE'
cycle_2_date = 'CYCLE_2_SURVEY_DATE'
time_feature = TimedeltaTransformer(cycle_1_date, cycle_2_date)
cycle_1_cols = ['CYCLE_1_DEFS', 'CYCLE_1_NFROMDEFS', 'CYCLE_1_NFROMCOMP',
                'CYCLE_1_DEFS_SCORE', 'CYCLE_1_NUMREVIS',
                'CYCLE_1_REVISIT_SCORE', 'CYCLE_1_TOTAL_SCORE']
cycle_1_features = ColumnSelectTransformer(cycle_1_cols)
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Lasso

gs = GridSearchCV(Lasso(max_iter = 1000), param_grid = {'alpha': np.arange(0, 3.5, 0.5)}, cv=5, n_jobs=4, verbose=1)

survey_model = Pipeline([
    ('features', FeatureUnion([
        ('business', business_features),
        ('survey', cycle_1_features),
        ('time', time_feature)
    ])),
    # add your estimator here
    ('poly', PolynomialFeatures(2)),
    ('decomp', TruncatedSVD(40)),
    ('gs', gs)
])
survey_model.fit(data, cycle_2_score.astype(int))
grader.score.ml__survey_model(survey_model.predict)
%logstop
%logstart -rtq ~/.logs/ml.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Wed, 24 Mar 2021 18:23:53
from static_grader import grader# Wed, 24 Mar 2021 18:23:53
%%bash
mkdir data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/providers-train.csv -nc -P ./ml-data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/providers-metadata.csv -nc -P ./ml-data# Wed, 24 Mar 2021 18:23:53
import numpy as np
import pandas as pd# Wed, 24 Mar 2021 18:23:53
metadata = pd.read_csv('./ml-data/providers-metadata.csv')
metadata.head()# Wed, 24 Mar 2021 18:23:53
data = pd.read_csv('./ml-data/providers-train.csv', encoding='latin1')

fine_counts = data.pop('FINE_CNT')
fine_totals = data.pop('FINE_TOT')
cycle_2_score = data.pop('CYCLE_2_TOTAL_SCORE')# Wed, 24 Mar 2021 18:24:01
data.head()# Wed, 24 Mar 2021 18:24:01
from sklearn.base import BaseEstimator, RegressorMixin, TransformerMixin

class GroupMeanEstimator(BaseEstimator, RegressorMixin):
    def __init__(self, gb_col):
        self.gb_col = gb_col
        self.group_averages = {}
        self.global_avg = 0

    def fit(self, X, y):
        # Use self.group_averages to store the average penalty by group
        self.group_averages = y.groupby(X[self.gb_col]).mean().to_dict()
        self.global_avg = y.mean()
        return self

    def predict(self, X):
        if not isinstance (X, pd.DataFrame):
            X = pd.DataFrame(X)
        # Return a list of predicted penalties based on group of samples in X
        return [self.group_averages.get(row, self.global_avg) 
                for row in X[self.gb_col]]# Wed, 24 Mar 2021 18:24:01
from sklearn.pipeline import Pipeline

state_model = Pipeline([
    ('sme', GroupMeanEstimator(gb_col='STATE'))
    ])
state_model.fit(data, fine_totals)# Wed, 24 Mar 2021 18:24:01
state_model.predict(data.sample(5))# Wed, 24 Mar 2021 18:24:01
state_model.predict(pd.DataFrame([{'STATE': 'AS'}]))# Wed, 24 Mar 2021 18:24:01
grader.score.ml__state_model(state_model.predict)# Wed, 24 Mar 2021 18:24:01
from sklearn.impute import SimpleImputer

simple_cols = ['BEDCERT', 'RESTOT', 'INHOSP', 'CCRC_FACIL', 'SFF', 'CHOW_LAST_12MOS', 'SPRINKLER_STATUS', 'EXP_TOTAL', 'ADJ_TOTAL']

class ColumnSelectTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, columns):
        self.columns = columns

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance(X, pd.DataFrame):
            X = pd.DataFrame(X)
        return X[self.columns]
        
simple_features = Pipeline([
    ('cst', ColumnSelectTransformer(simple_cols)),
    ('imputer', SimpleImputer())
])# Wed, 24 Mar 2021 18:24:01
pd.DataFrame(simple_features.fit_transform(data)).info()# Wed, 24 Mar 2021 18:24:01
assert data['RESTOT'].isnull().sum() > 0
assert not np.isnan(simple_features.fit_transform(data)).any()# Wed, 24 Mar 2021 18:24:01
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV# Wed, 24 Mar 2021 18:24:01
simple_features_model = Pipeline([
    ('simple', simple_features),
    # add your estimator here
    ('predictor', RandomForestClassifier(n_estimators=50, max_depth = 5))
])

param_grid = {'predictor__n_estimators': range(25, 126, 25),
              'predictor__max_depth': range(6, 17, 2)}

sfm_gs = GridSearchCV(simple_features_model, 
                      param_grid = param_grid, n_jobs = -1, verbose = 1)

sfm_gs.fit(data, fine_counts > 0);# Wed, 24 Mar 2021 18:25:25
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV# Wed, 24 Mar 2021 18:25:25
simple_features_model = Pipeline([
    ('simple', simple_features),
    # add your estimator here
    ('predictor', RandomForestClassifier(n_estimators=50, max_depth = 5))
])

param_grid = {'predictor__n_estimators': range(25, 126, 25),
              'predictor__max_depth': range(6, 17, 2)}

sfm_gs = GridSearchCV(simple_features_model, 
                      param_grid = param_grid, n_jobs = -1, verbose = 1)

sfm_gs.fit(data, fine_counts > 0);# Wed, 24 Mar 2021 18:26:48
simple_features_model.fit(data, fine_counts > 0)# Wed, 24 Mar 2021 18:26:49
def positive_probability(model):
    def predict_proba(X):
        return model.predict_proba(X)[:, 1]
    return predict_proba

grader.score.ml__simple_features(positive_probability(simple_features_model))# Wed, 24 Mar 2021 18:26:49
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import OneHotEncoder

owner_onehot = Pipeline([
    ('cst', ColumnSelectTransformer(['OWNERSHIP'])),
    ('ohe', OneHotEncoder (categories = 'auto', sparse = False))
])

cert_onehot = Pipeline([
    ('cst', ColumnSelectTransformer(['CERTIFICATION'])),
    ('ohe', OneHotEncoder (categories = 'auto', sparse = False))
])

categorical_features = FeatureUnion([
    ('owner', owner_onehot),
    ('cert', cert_onehot)
])# Wed, 24 Mar 2021 18:26:49
assert categorical_features.fit_transform(data).shape[0] == data.shape[0]
assert categorical_features.fit_transform(data).dtype == np.float64
assert not np.isnan(categorical_features.fit_transform(data)).any()# Wed, 24 Mar 2021 18:26:49
categorical_features_model = Pipeline([
    ('categorical', categorical_features),
    # add your estimator here
    ('classifier', RandomForestClassifier())
])# Wed, 24 Mar 2021 18:26:49
categorical_features_model.fit(data, fine_counts > 0)# Wed, 24 Mar 2021 18:26:49
grader.score.ml__categorical_features(positive_probability(categorical_features_model))# Wed, 24 Mar 2021 18:26:49
from sklearn.preprocessing  import PolynomialFeatures
from sklearn.linear_model import LogisticRegression# Wed, 24 Mar 2021 18:26:49
business_features = FeatureUnion([
    ('simple', simple_features),
    ('categorical', categorical_features)
])# Wed, 24 Mar 2021 18:26:49

business_model = Pipeline([
    ('features', business_features),
    # add your estimator here
    ('ploy', PolynomialFeatures(2)),
    ('lr', LogisticRegression())
])# Wed, 24 Mar 2021 18:26:49
business_model.fit(data, fine_counts > 0)# Wed, 24 Mar 2021 18:26:50
grader.score.ml__business_model(positive_probability(business_model))# Wed, 24 Mar 2021 18:26:50
class TimedeltaTransformer(BaseEstimator, TransformerMixin):
    def __init__(self, t1_col, t2_col):
        self.t1_col = t1_col
        self.t2_col = t2_col

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if not isinstance (X, pd.DataFrame):
            X = pd.DataFrame(X)
        results = (pd.to_datetime(X[self.t1_col]) - pd.to_datetime(X[self.t1_col]))
        
        results = results.apply(lambda X:X.days).values.reshape(-1, 1)
        
        return results# Wed, 24 Mar 2021 18:26:50
cycle_1_date = 'CYCLE_1_SURVEY_DATE'
cycle_2_date = 'CYCLE_2_SURVEY_DATE'
time_feature = TimedeltaTransformer(cycle_1_date, cycle_2_date)# Wed, 24 Mar 2021 18:26:50
cycle_1_cols = ['CYCLE_1_DEFS', 'CYCLE_1_NFROMDEFS', 'CYCLE_1_NFROMCOMP',
                'CYCLE_1_DEFS_SCORE', 'CYCLE_1_NUMREVIS',
                'CYCLE_1_REVISIT_SCORE', 'CYCLE_1_TOTAL_SCORE']
cycle_1_features = ColumnSelectTransformer(cycle_1_cols)# Wed, 24 Mar 2021 18:26:50
from sklearn.ensemble import RandomForestRegressor
from sklearn.decomposition import TruncatedSVD
from sklearn.linear_model import Lasso

gs = GridSearchCV(Lasso(max_iter = 1000), param_grid = {'alpha': np.arange(0, 3.5, 0.5)}, cv=5, n_jobs=4, verbose=1)

survey_model = Pipeline([
    ('features', FeatureUnion([
        ('business', business_features),
        ('survey', cycle_1_features),
        ('time', time_feature)
    ])),
    # add your estimator here
    ('poly', PolynomialFeatures(2)),
    ('decomp', TruncatedSVD(40)),
    ('gs', gs)
])# Wed, 24 Mar 2021 18:26:50
survey_model.fit(data, cycle_2_score.astype(int))# Wed, 24 Mar 2021 18:27:03
grader.score.ml__survey_model(survey_model.predict)# Wed, 24 Mar 2021 18:28:50
grader.score.ml__survey_model(survey_model.predict)