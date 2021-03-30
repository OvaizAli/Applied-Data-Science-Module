# Wed, 24 Mar 2021 17:06:18

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
])# Wed, 24 Mar 2021 17:07:00
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
])# Wed, 24 Mar 2021 17:07:03
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
])# Wed, 24 Mar 2021 17:07:16
%logstop
%logstart -rtq ~/.logs/ml.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144# Wed, 24 Mar 2021 17:51:14

business_model = Pipeline([
    ('features', business_features),
    # add your estimator here
    ('ploy', PolynomialFeatures(5)),
    ('lr', LogisticRegression())
])# Wed, 24 Mar 2021 17:51:14
business_model.fit(data, fine_counts > 0)# Wed, 24 Mar 2021 17:54:00
business_features = FeatureUnion([
    ('simple', simple_features),
    ('categorical', categorical_features)
])# Wed, 24 Mar 2021 17:54:00
from sklearn.preprocessing  import PolynomialFeatures
from sklearn.linear_model import LogisticRegression# Wed, 24 Mar 2021 17:54:00
business_features = FeatureUnion([
    ('simple', simple_features),
    ('categorical', categorical_features)
])# Wed, 24 Mar 2021 17:54:17
%logstop
%logstart -rtq ~/.logs/ml.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144# Wed, 24 Mar 2021 18:18:32
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
])# Wed, 24 Mar 2021 18:18:33
grader.score.ml__survey_model(survey_model.predict)# Wed, 24 Mar 2021 18:18:41
%logstop
%logstart -rtq ~/.logs/ml.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144# Wed, 24 Mar 2021 18:20:44
%logstop
%logstart -rtq ~/.logs/ml.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144# Wed, 24 Mar 2021 18:39:14
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144# Wed, 24 Mar 2021 18:46:00
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144# Wed, 24 Mar 2021 18:46:32
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144# Wed, 24 Mar 2021 19:02:40
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144