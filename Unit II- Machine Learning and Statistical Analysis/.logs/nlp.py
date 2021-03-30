%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Wed, 24 Mar 2021 18:39:30
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Wed, 24 Mar 2021 18:39:31
from static_grader import grader# Wed, 24 Mar 2021 18:44:20
%%bash
mkdir data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/amazon_electronics_reviews_training.json.gz -nc -P ./data# Wed, 24 Mar 2021 18:44:22
%%bash
mkdir data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/amazon_electronics_reviews_training.json.gz -nc -P ./data# Wed, 24 Mar 2021 18:44:22
import gzip
import ujson as json

with gzip.open("data/amazon_electronics_reviews_training.json.gz", "r") as f:                                  
    data = [json.loads(line) for line in f]%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Wed, 24 Mar 2021 18:46:01
from static_grader import grader# Wed, 24 Mar 2021 18:46:01
%%bash
mkdir data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/amazon_electronics_reviews_training.json.gz -nc -P ./data# Wed, 24 Mar 2021 18:46:01
import gzip
import ujson as json

with gzip.open("data/amazon_electronics_reviews_training.json.gz", "r") as f:                                  
    data = [json.loads(line) for line in f]# Wed, 24 Mar 2021 18:46:07
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
from static_grader import grader
%%bash
mkdir data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/amazon_electronics_reviews_training.json.gz -nc -P ./data
import gzip
import ujson as json

with gzip.open("data/amazon_electronics_reviews_training.json.gz", "r") as f:                                  
    data = [json.loads(line) for line in f]
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Wed, 24 Mar 2021 18:46:07
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
from static_grader import grader
%%bash
mkdir data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/amazon_electronics_reviews_training.json.gz -nc -P ./data
import gzip
import ujson as json

with gzip.open("data/amazon_electronics_reviews_training.json.gz", "r") as f:                                  
    data = [json.loads(line) for line in f]
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Wed, 24 Mar 2021 18:47:47
ratings = [X['overall']  for X in data]# Wed, 24 Mar 2021 18:48:03
import gzip
import ujson as json

with gzip.open("data/amazon_electronics_reviews_training.json.gz", "r") as f:                                  
    data = [json.loads(line) for line in f]# Wed, 24 Mar 2021 18:48:09
ratings = [X['overall']  for X in data]# Wed, 24 Mar 2021 18:54:55
from sklearn.base import BaseEstimator, TransformerMixin

class KeySelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    
    def fit(self, X, Y=None):
        return self
    
    def transform(self, X):
        return[d[self.key] for d in X]# Wed, 24 Mar 2021 18:55:30
ks = KeySelector('reviewText')
x_trans = ks.fit_transform(data)
print(x_trans[0])# Wed, 24 Mar 2021 18:56:52
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge# Wed, 24 Mar 2021 18:56:52
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
ratings = [X['overall']  for X in data]
import gzip
import ujson as json

with gzip.open("data/amazon_electronics_reviews_training.json.gz", "r") as f:                                  
    data = [json.loads(line) for line in f]
ratings = [X['overall']  for X in data]
from sklearn.base import BaseEstimator, TransformerMixin

class KeySelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    
    def fit(self, X, Y=None):
        return self
    
    def transform(self, X):
        return[d[self.key] for d in X]
ks = KeySelector('reviewText')
x_trans = ks.fit_transform(data)
print(x_trans[0])
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Wed, 24 Mar 2021 18:57:48
bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge(alpha=0.6))
])

bag_of_words_model.fit(data, ratings);# Wed, 24 Mar 2021 18:58:49
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
ratings = [X['overall']  for X in data]
import gzip
import ujson as json

with gzip.open("data/amazon_electronics_reviews_training.json.gz", "r") as f:                                  
    data = [json.loads(line) for line in f]
ratings = [X['overall']  for X in data]
from sklearn.base import BaseEstimator, TransformerMixin

class KeySelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    
    def fit(self, X, Y=None):
        return self
    
    def transform(self, X):
        return[d[self.key] for d in X]
ks = KeySelector('reviewText')
x_trans = ks.fit_transform(data)
print(x_trans[0])
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge(alpha=0.6))
])

bag_of_words_model.fit(data, ratings);
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Wed, 24 Mar 2021 18:58:55
bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge(alpha=0.6))
])

bag_of_words_model.fit(data, ratings);# Wed, 24 Mar 2021 18:59:14
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
ratings = [X['overall']  for X in data]
import gzip
import ujson as json

with gzip.open("data/amazon_electronics_reviews_training.json.gz", "r") as f:                                  
    data = [json.loads(line) for line in f]
ratings = [X['overall']  for X in data]
from sklearn.base import BaseEstimator, TransformerMixin

class KeySelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    
    def fit(self, X, Y=None):
        return self
    
    def transform(self, X):
        return[d[self.key] for d in X]
ks = KeySelector('reviewText')
x_trans = ks.fit_transform(data)
print(x_trans[0])
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge(alpha=0.6))
])

bag_of_words_model.fit(data, ratings);
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge(alpha=0.6))
])

bag_of_words_model.fit(data, ratings);
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Wed, 24 Mar 2021 18:59:20
bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge(alpha=0.6))
])

bag_of_words_model.fit(data, ratings);# Wed, 24 Mar 2021 18:59:31
grader.score.nlp__bag_of_words_model(bag_of_words_model.predict)# Wed, 24 Mar 2021 18:59:51
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
ratings = [X['overall']  for X in data]
import gzip
import ujson as json

with gzip.open("data/amazon_electronics_reviews_training.json.gz", "r") as f:                                  
    data = [json.loads(line) for line in f]
ratings = [X['overall']  for X in data]
from sklearn.base import BaseEstimator, TransformerMixin

class KeySelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    
    def fit(self, X, Y=None):
        return self
    
    def transform(self, X):
        return[d[self.key] for d in X]
ks = KeySelector('reviewText')
x_trans = ks.fit_transform(data)
print(x_trans[0])
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge(alpha=0.6))
])

bag_of_words_model.fit(data, ratings);
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge(alpha=0.6))
])

bag_of_words_model.fit(data, ratings);
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge(alpha=0.6))
])

bag_of_words_model.fit(data, ratings);
grader.score.nlp__bag_of_words_model(bag_of_words_model.predict)
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Wed, 24 Mar 2021 19:02:40
from static_grader import grader# Wed, 24 Mar 2021 19:02:41
%%bash
mkdir data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/amazon_electronics_reviews_training.json.gz -nc -P ./data# Wed, 24 Mar 2021 19:02:41
import gzip
import ujson as json

with gzip.open("data/amazon_electronics_reviews_training.json.gz", "r") as f:                                  
    data = [json.loads(line) for line in f]# Wed, 24 Mar 2021 19:02:42
ratings = [X['overall']  for X in data]# Wed, 24 Mar 2021 19:02:42
from sklearn.base import BaseEstimator, TransformerMixin

class KeySelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    
    def fit(self, X, Y=None):
        return self
    
    def transform(self, X):
        return[d[self.key] for d in X]# Wed, 24 Mar 2021 19:02:42
ks = KeySelector('reviewText')
x_trans = ks.fit_transform(data)
print(x_trans[0])# Wed, 24 Mar 2021 19:02:43
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge# Wed, 24 Mar 2021 19:02:43
bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge(alpha=0.6))
])

bag_of_words_model.fit(data, ratings);# Wed, 24 Mar 2021 19:02:54
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
from static_grader import grader
%%bash
mkdir data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/amazon_electronics_reviews_training.json.gz -nc -P ./data
import gzip
import ujson as json

with gzip.open("data/amazon_electronics_reviews_training.json.gz", "r") as f:                                  
    data = [json.loads(line) for line in f]
ratings = [X['overall']  for X in data]
from sklearn.base import BaseEstimator, TransformerMixin

class KeySelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    
    def fit(self, X, Y=None):
        return self
    
    def transform(self, X):
        return[d[self.key] for d in X]
ks = KeySelector('reviewText')
x_trans = ks.fit_transform(data)
print(x_trans[0])
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge(alpha=0.6))
])

bag_of_words_model.fit(data, ratings);
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Wed, 24 Mar 2021 19:02:55
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
from static_grader import grader
%%bash
mkdir data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/amazon_electronics_reviews_training.json.gz -nc -P ./data
import gzip
import ujson as json

with gzip.open("data/amazon_electronics_reviews_training.json.gz", "r") as f:                                  
    data = [json.loads(line) for line in f]
ratings = [X['overall']  for X in data]
from sklearn.base import BaseEstimator, TransformerMixin

class KeySelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    
    def fit(self, X, Y=None):
        return self
    
    def transform(self, X):
        return[d[self.key] for d in X]
ks = KeySelector('reviewText')
x_trans = ks.fit_transform(data)
print(x_trans[0])
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge(alpha=0.6))
])

bag_of_words_model.fit(data, ratings);
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Wed, 24 Mar 2021 19:03:09
bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge(alpha=0.6))
])

bag_of_words_model.fit(data, ratings);# Wed, 24 Mar 2021 19:03:20
grader.score.nlp__bag_of_words_model(bag_of_words_model.predict)# Wed, 24 Mar 2021 19:04:00
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
from static_grader import grader
%%bash
mkdir data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/amazon_electronics_reviews_training.json.gz -nc -P ./data
import gzip
import ujson as json

with gzip.open("data/amazon_electronics_reviews_training.json.gz", "r") as f:                                  
    data = [json.loads(line) for line in f]
ratings = [X['overall']  for X in data]
from sklearn.base import BaseEstimator, TransformerMixin

class KeySelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    
    def fit(self, X, Y=None):
        return self
    
    def transform(self, X):
        return[d[self.key] for d in X]
ks = KeySelector('reviewText')
x_trans = ks.fit_transform(data)
print(x_trans[0])
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge(alpha=0.6))
])

bag_of_words_model.fit(data, ratings);
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge(alpha=0.6))
])

bag_of_words_model.fit(data, ratings);
grader.score.nlp__bag_of_words_model(bag_of_words_model.predict)
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Wed, 24 Mar 2021 19:04:11
grader.score.nlp__bag_of_words_model(bag_of_words_model.predict)# Wed, 24 Mar 2021 19:06:12
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
from static_grader import grader
%%bash
mkdir data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/amazon_electronics_reviews_training.json.gz -nc -P ./data
import gzip
import ujson as json

with gzip.open("data/amazon_electronics_reviews_training.json.gz", "r") as f:                                  
    data = [json.loads(line) for line in f]
ratings = [X['overall']  for X in data]
from sklearn.base import BaseEstimator, TransformerMixin

class KeySelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    
    def fit(self, X, Y=None):
        return self
    
    def transform(self, X):
        return[d[self.key] for d in X]
ks = KeySelector('reviewText')
x_trans = ks.fit_transform(data)
print(x_trans[0])
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge(alpha=0.6))
])

bag_of_words_model.fit(data, ratings);
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge(alpha=0.6))
])

bag_of_words_model.fit(data, ratings);
grader.score.nlp__bag_of_words_model(bag_of_words_model.predict)
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
grader.score.nlp__bag_of_words_model(bag_of_words_model.predict)
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Wed, 24 Mar 2021 19:06:49
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
from static_grader import grader
%%bash
mkdir data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/amazon_electronics_reviews_training.json.gz -nc -P ./data
import gzip
import ujson as json

with gzip.open("data/amazon_electronics_reviews_training.json.gz", "r") as f:                                  
    data = [json.loads(line) for line in f]
ratings = [X['overall']  for X in data]
from sklearn.base import BaseEstimator, TransformerMixin

class KeySelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    
    def fit(self, X, Y=None):
        return self
    
    def transform(self, X):
        return[d[self.key] for d in X]
ks = KeySelector('reviewText')
x_trans = ks.fit_transform(data)
print(x_trans[0])
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge(alpha=0.6))
])

bag_of_words_model.fit(data, ratings);
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge(alpha=0.6))
])

bag_of_words_model.fit(data, ratings);
grader.score.nlp__bag_of_words_model(bag_of_words_model.predict)
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
grader.score.nlp__bag_of_words_model(bag_of_words_model.predict)
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Wed, 24 Mar 2021 19:06:53
from sklearn.linear_model import Lasso# Wed, 24 Mar 2021 19:06:54
normalized_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', TfidfVectorizer()),
    ('predictor', Ridge())
])# Wed, 24 Mar 2021 19:06:55
grader.score.nlp__normalized_model(normalized_model.predict)# Wed, 24 Mar 2021 19:07:11
normalized_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', TfidfVectorizer()),
    ('predictor', Ridge())
])
normalized_model.fit(data, ratings);# Wed, 24 Mar 2021 19:07:20
grader.score.nlp__normalized_model(normalized_model.predict)# Wed, 24 Mar 2021 19:09:30
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
from static_grader import grader
%%bash
mkdir data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/amazon_electronics_reviews_training.json.gz -nc -P ./data
import gzip
import ujson as json

with gzip.open("data/amazon_electronics_reviews_training.json.gz", "r") as f:                                  
    data = [json.loads(line) for line in f]
ratings = [X['overall']  for X in data]
from sklearn.base import BaseEstimator, TransformerMixin

class KeySelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    
    def fit(self, X, Y=None):
        return self
    
    def transform(self, X):
        return[d[self.key] for d in X]
ks = KeySelector('reviewText')
x_trans = ks.fit_transform(data)
print(x_trans[0])
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge(alpha=0.6))
])

bag_of_words_model.fit(data, ratings);
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge(alpha=0.6))
])

bag_of_words_model.fit(data, ratings);
grader.score.nlp__bag_of_words_model(bag_of_words_model.predict)
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
grader.score.nlp__bag_of_words_model(bag_of_words_model.predict)
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
from sklearn.linear_model import Lasso
normalized_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', TfidfVectorizer()),
    ('predictor', Ridge())
])
grader.score.nlp__normalized_model(normalized_model.predict)
normalized_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', TfidfVectorizer()),
    ('predictor', Ridge())
])
normalized_model.fit(data, ratings);
grader.score.nlp__normalized_model(normalized_model.predict)
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Wed, 24 Mar 2021 19:09:58
bigrams_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer(ngram_range=(1,2))),
    ('predictor', Ridge(alpha=0.6))
])
bigrams_model.fit(data, ratings);# Wed, 24 Mar 2021 19:10:28
grader.score.nlp__bigrams_model(bigrams_model.predict)# Wed, 24 Mar 2021 19:11:39
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
from static_grader import grader
%%bash
mkdir data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/amazon_electronics_reviews_training.json.gz -nc -P ./data
import gzip
import ujson as json

with gzip.open("data/amazon_electronics_reviews_training.json.gz", "r") as f:                                  
    data = [json.loads(line) for line in f]
ratings = [X['overall']  for X in data]
from sklearn.base import BaseEstimator, TransformerMixin

class KeySelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    
    def fit(self, X, Y=None):
        return self
    
    def transform(self, X):
        return[d[self.key] for d in X]
ks = KeySelector('reviewText')
x_trans = ks.fit_transform(data)
print(x_trans[0])
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge(alpha=0.6))
])

bag_of_words_model.fit(data, ratings);
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge(alpha=0.6))
])

bag_of_words_model.fit(data, ratings);
grader.score.nlp__bag_of_words_model(bag_of_words_model.predict)
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
grader.score.nlp__bag_of_words_model(bag_of_words_model.predict)
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
from sklearn.linear_model import Lasso
normalized_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', TfidfVectorizer()),
    ('predictor', Ridge())
])
grader.score.nlp__normalized_model(normalized_model.predict)
normalized_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', TfidfVectorizer()),
    ('predictor', Ridge())
])
normalized_model.fit(data, ratings);
grader.score.nlp__normalized_model(normalized_model.predict)
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
bigrams_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer(ngram_range=(1,2))),
    ('predictor', Ridge(alpha=0.6))
])
bigrams_model.fit(data, ratings);
grader.score.nlp__bigrams_model(bigrams_model.predict)
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Wed, 24 Mar 2021 19:12:26
%%bash
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/amazon_one_and_five_star_reviews.json.gz -nc -P ./data# Wed, 24 Mar 2021 19:12:43
del data, ratings# Wed, 24 Mar 2021 19:13:11
import numpy as np
from sklearn.naive_bayes import MultinomialNB

with gzip.open("data/amazon_one_and_five_star_reviews.json.gz", "r") as f:
    data_polarity = [json.loads(line) for line in f]

ratings = [row['overall'] for row in data_polarity]# Wed, 24 Mar 2021 19:13:27
pipe = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', TfidfVectorizer(stop_words='english')),
    ('predictor', MultinomialNB())
])

pipe.fit(data_polarity, ratings);# Wed, 24 Mar 2021 19:15:02
pipe = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', TfidfVectorizer(stop_words='english')),
    ('predictor', MultinomialNB())
])

pipe.fit(data_polarity, ratings);# Wed, 24 Mar 2021 19:15:03
#get features (vocab) from model
feat_to_token = pipe['vectorizer'].get_feature_names()

#get the log probability from model
log_prob = pipe['predictor'].feature_log_prob_

#collapse log probability into one row
polarity = log_prob[0, :] - log_prob[1, :]

#combine polarity and feature names
most_polar = sorted(list(zip(polarity, feat_to_token)))# Wed, 24 Mar 2021 19:15:03
n=25
most_polar = most_polar[:n] + most_polar[-n:]# Wed, 24 Mar 2021 19:15:03
top_50 = [term for score, term in most_polar]# Wed, 24 Mar 2021 19:15:03
grader.score.nlp__most_polar(top_50)# Wed, 24 Mar 2021 19:15:03
pipe = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', TfidfVectorizer(stop_words='english')),
    ('predictor', MultinomialNB())
])

pipe.fit(data_polarity, ratings);# Wed, 24 Mar 2021 19:15:04
#get features (vocab) from model
feat_to_token = pipe['vectorizer'].get_feature_names()

#get the log probability from model
log_prob = pipe['predictor'].feature_log_prob_

#collapse log probability into one row
polarity = log_prob[0, :] - log_prob[1, :]

#combine polarity and feature names
most_polar = sorted(list(zip(polarity, feat_to_token)))# Wed, 24 Mar 2021 19:15:04
n=25
most_polar = most_polar[:n] + most_polar[-n:]# Wed, 24 Mar 2021 19:15:04
top_50 = [term for score, term in most_polar]# Wed, 24 Mar 2021 19:15:04
grader.score.nlp__most_polar(top_50)# Wed, 24 Mar 2021 19:15:05
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
from static_grader import grader
%%bash
mkdir data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/amazon_electronics_reviews_training.json.gz -nc -P ./data
import gzip
import ujson as json

with gzip.open("data/amazon_electronics_reviews_training.json.gz", "r") as f:                                  
    data = [json.loads(line) for line in f]
ratings = [X['overall']  for X in data]
from sklearn.base import BaseEstimator, TransformerMixin

class KeySelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    
    def fit(self, X, Y=None):
        return self
    
    def transform(self, X):
        return[d[self.key] for d in X]
ks = KeySelector('reviewText')
x_trans = ks.fit_transform(data)
print(x_trans[0])
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge(alpha=0.6))
])

bag_of_words_model.fit(data, ratings);
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge(alpha=0.6))
])

bag_of_words_model.fit(data, ratings);
grader.score.nlp__bag_of_words_model(bag_of_words_model.predict)
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
grader.score.nlp__bag_of_words_model(bag_of_words_model.predict)
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
from sklearn.linear_model import Lasso
normalized_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', TfidfVectorizer()),
    ('predictor', Ridge())
])
grader.score.nlp__normalized_model(normalized_model.predict)
normalized_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', TfidfVectorizer()),
    ('predictor', Ridge())
])
normalized_model.fit(data, ratings);
grader.score.nlp__normalized_model(normalized_model.predict)
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
bigrams_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer(ngram_range=(1,2))),
    ('predictor', Ridge(alpha=0.6))
])
bigrams_model.fit(data, ratings);
grader.score.nlp__bigrams_model(bigrams_model.predict)
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%%bash
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/amazon_one_and_five_star_reviews.json.gz -nc -P ./data
del data, ratings
import numpy as np
from sklearn.naive_bayes import MultinomialNB

with gzip.open("data/amazon_one_and_five_star_reviews.json.gz", "r") as f:
    data_polarity = [json.loads(line) for line in f]

ratings = [row['overall'] for row in data_polarity]
pipe = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', TfidfVectorizer(stop_words='english')),
    ('predictor', MultinomialNB())
])

pipe.fit(data_polarity, ratings);
pipe = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', TfidfVectorizer(stop_words='english')),
    ('predictor', MultinomialNB())
])

pipe.fit(data_polarity, ratings);
#get features (vocab) from model
feat_to_token = pipe['vectorizer'].get_feature_names()

#get the log probability from model
log_prob = pipe['predictor'].feature_log_prob_

#collapse log probability into one row
polarity = log_prob[0, :] - log_prob[1, :]

#combine polarity and feature names
most_polar = sorted(list(zip(polarity, feat_to_token)))
n=25
most_polar = most_polar[:n] + most_polar[-n:]
top_50 = [term for score, term in most_polar]
grader.score.nlp__most_polar(top_50)
pipe = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', TfidfVectorizer(stop_words='english')),
    ('predictor', MultinomialNB())
])

pipe.fit(data_polarity, ratings);
#get features (vocab) from model
feat_to_token = pipe['vectorizer'].get_feature_names()

#get the log probability from model
log_prob = pipe['predictor'].feature_log_prob_

#collapse log probability into one row
polarity = log_prob[0, :] - log_prob[1, :]

#combine polarity and feature names
most_polar = sorted(list(zip(polarity, feat_to_token)))
n=25
most_polar = most_polar[:n] + most_polar[-n:]
top_50 = [term for score, term in most_polar]
grader.score.nlp__most_polar(top_50)
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
# Wed, 24 Mar 2021 19:17:21
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
from static_grader import grader
%%bash
mkdir data
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/amazon_electronics_reviews_training.json.gz -nc -P ./data
import gzip
import ujson as json

with gzip.open("data/amazon_electronics_reviews_training.json.gz", "r") as f:                                  
    data = [json.loads(line) for line in f]
ratings = [X['overall']  for X in data]
from sklearn.base import BaseEstimator, TransformerMixin

class KeySelector(BaseEstimator, TransformerMixin):
    def __init__(self, key):
        self.key = key
    
    def fit(self, X, Y=None):
        return self
    
    def transform(self, X):
        return[d[self.key] for d in X]
ks = KeySelector('reviewText')
x_trans = ks.fit_transform(data)
print(x_trans[0])
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, HashingVectorizer, TfidfVectorizer
from sklearn.linear_model import Ridge
bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge(alpha=0.6))
])

bag_of_words_model.fit(data, ratings);
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
bag_of_words_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer()),
    ('regressor', Ridge(alpha=0.6))
])

bag_of_words_model.fit(data, ratings);
grader.score.nlp__bag_of_words_model(bag_of_words_model.predict)
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
grader.score.nlp__bag_of_words_model(bag_of_words_model.predict)
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
from sklearn.linear_model import Lasso
normalized_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', TfidfVectorizer()),
    ('predictor', Ridge())
])
grader.score.nlp__normalized_model(normalized_model.predict)
normalized_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', TfidfVectorizer()),
    ('predictor', Ridge())
])
normalized_model.fit(data, ratings);
grader.score.nlp__normalized_model(normalized_model.predict)
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
bigrams_model = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', HashingVectorizer(ngram_range=(1,2))),
    ('predictor', Ridge(alpha=0.6))
])
bigrams_model.fit(data, ratings);
grader.score.nlp__bigrams_model(bigrams_model.predict)
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%%bash
wget http://dataincubator-wqu.s3.amazonaws.com/mldata/amazon_one_and_five_star_reviews.json.gz -nc -P ./data
del data, ratings
import numpy as np
from sklearn.naive_bayes import MultinomialNB

with gzip.open("data/amazon_one_and_five_star_reviews.json.gz", "r") as f:
    data_polarity = [json.loads(line) for line in f]

ratings = [row['overall'] for row in data_polarity]
pipe = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', TfidfVectorizer(stop_words='english')),
    ('predictor', MultinomialNB())
])

pipe.fit(data_polarity, ratings);
pipe = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', TfidfVectorizer(stop_words='english')),
    ('predictor', MultinomialNB())
])

pipe.fit(data_polarity, ratings);
#get features (vocab) from model
feat_to_token = pipe['vectorizer'].get_feature_names()

#get the log probability from model
log_prob = pipe['predictor'].feature_log_prob_

#collapse log probability into one row
polarity = log_prob[0, :] - log_prob[1, :]

#combine polarity and feature names
most_polar = sorted(list(zip(polarity, feat_to_token)))
n=25
most_polar = most_polar[:n] + most_polar[-n:]
top_50 = [term for score, term in most_polar]
grader.score.nlp__most_polar(top_50)
pipe = Pipeline([
    ('selector', KeySelector('reviewText')),
    ('vectorizer', TfidfVectorizer(stop_words='english')),
    ('predictor', MultinomialNB())
])

pipe.fit(data_polarity, ratings);
#get features (vocab) from model
feat_to_token = pipe['vectorizer'].get_feature_names()

#get the log probability from model
log_prob = pipe['predictor'].feature_log_prob_

#collapse log probability into one row
polarity = log_prob[0, :] - log_prob[1, :]

#combine polarity and feature names
most_polar = sorted(list(zip(polarity, feat_to_token)))
n=25
most_polar = most_polar[:n] + most_polar[-n:]
top_50 = [term for score, term in most_polar]
grader.score.nlp__most_polar(top_50)
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
%logstop
%logstart -rtq ~/.logs/nlp.py append
%matplotlib inline
import matplotlib
import seaborn as sns
sns.set()
matplotlib.rcParams['figure.dpi'] = 144
