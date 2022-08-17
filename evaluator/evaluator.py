from sklearn.model_selection import KFold
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    mean_absolute_error,
)
from collections import defaultdict
import pandas as pd
import numpy as np
# from tqdm.notebook import tqdm

from xgboost import XGBClassifier, XGBRegressor
from lightgbm import LGBMClassifier, LGBMRegressor
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.ensemble import (
    RandomForestClassifier, 
    RandomForestRegressor,
    ExtraTreesClassifier,
    ExtraTreesRegressor,
)
from sklearn.impute import SimpleImputer
import optuna

class Model():
  def __init__(self, train_df, model_name="lgbm", model_type="clf", **params):
    self.model_name = model_name
    self.model_type = model_type
    self.train_df = train_df
    self.model = None
    if model_name == "lgbm":
      if model_type == "clf":
        self.model = LGBMClassifier(**params)
      elif model_type == "rgr":
        self.model = LGBMRegressor(**params)
    elif model_name == "xgb":
      if model_type == "clf":
        self.model = XGBClassifier(**params)
      elif model_type == "rgr":
        self.model = XGBRegressor(**params)
    elif model_name == "et":
      if model_type == "clf":
        self.model = ExtraTreesClassifier(**params)
      elif model_type == "rgr":
        self.model = ExtraTreesRegressor(**params)
    elif model_name == "rf":
      if model_type == "clf":
        self.model = RandomForestClassifier(**params)
      elif model_type == "rgr":
        self.model = RandomForestRegressor(**params)
    elif model_name == "cat":
      if model_type == "clf":
        self.model = CatBoostClassifier(**params)
      elif model_type == "rgr":
        self.model = CatBoostRegressor(**params)
    else:
      raise NameError("model_name must be in ('lgbm', 'xgb', 'rf', 'et', 'cat')")
    
    if self.model is None:
      raise NameError("model_type must be in ('clf', 'rgr')")

  def get_model(self):
    return {
        "train_df": self.train_df,
        "model": self.model,
        "model_name": self.model_name,
        "model_type": self.model_type,
    }

  def optimize(self, initial_params, **kwargs):
    self.optimizer = Optimizer(
        self.train_df, 
        initial_params, 
        self.model_name,
        self.model_type,
    )
    best_params = self.optimizer.run(**kwargs)
    self.__init__(self.train_df, self.model_name, self.model_type, **best_params)

class Optimizer():
  def __init__(self, train_df, initial_params, model_name, model_type, random_seed=42):
    self.params = {}
    self.train_df = train_df
    self.initial_params = initial_params
    self.model_name = model_name
    self.model_type = model_type
    self.random_seed = random_seed

  def objective(self, trial):
    ## Tuning Parmeters
    for param, dtype, value in self.initial_params:
      if dtype == "static":
        self.params[param] = value
      elif dtype == "int":
        self.params[param] = trial.suggest_int(param, *value)
      elif dtype == "float":
        self.params[param] = trial.suggest_uniform(param, *value)
      elif dtype == "log":
        self.params[param] = trial.suggest_loguniform(param, *value)
      elif dtype == "categorical":
        self.params[param] = trial.suggest_categorical(param, value)
      else:
        raise NameError("dtype must be one of ('static', 'int', 'float', 'log', 'categorical')")

    ## Objective Metric
    result_df = Evaluator(
        **Model(self.train_df, self.model_name, self.model_type, **self.params).get_model()
    ).run(train_acc=False)

    return result_df["roc_auc"]["mean"]

  def optimize(self, n_trials=100, n_runs=1, sampling="TPE"):
    if sampling == "random":
      sampler = optuna.samplers.RandomSampler(seed=self.random_seed)
    elif sampling == "TPE":
      sampler = optuna.samplers.TPESampler()

    self.n_runs = n_runs
    self.opt = optuna.create_study(
        direction='maximize',
        sampler=sampler,
    )
    self.opt.optimize(self.objective, n_trials=n_trials)

  def analyze(self):
    optuna.visualization.plot_optimization_history(self.opt).show()
    optuna.visualization.plot_param_importances(self.opt).show()
    optuna.visualization.plot_slice(self.opt).show()

  def best_params(self):
    print(self.opt.best_trial.value)
    print(self.opt.best_trial.params)
    return self.opt.best_trial.params

  def run(self, **kwargs):
    self.optimize(**kwargs)
    self.analyze()
    return self.best_params()

class Evaluator():
  def __init__(self, model, train_df, n_folds=4, random_state=42, model_name=None, model_type="clf"):
    self.kf = KFold(n_splits=n_folds, random_state=random_state, shuffle=True)
    self.model = model
    self.model_name = model_name
    self.model_type = model_type
    self.train_df = train_df
    if self.model_name == 'cat':
      self.fit_params = {'silent': True}
    else:
      self.fit_params = {}

  def drop_col(self, df, col_list=["index", "country"]):
    return df.drop(col_list, axis=1)

  def index_col(self, df, col_list=["country"]):
    def _indexer(col):
      col_list = list(set(df[col]))
      col_map = {
          c: i 
          for i, c in enumerate(col_list)
      }
      return df[col].apply(lambda x: col_map[x])

    for col in col_list:
      df[f"{col}_idx"] = _indexer(col)
    return self.drop_col(df, col_list)

  def preprocess(self, df=None, mode="index"):
    assert mode in ("index", "drop")

    df = self.train_df if df is None else df

    if mode == "drop":
      return self.drop_col(df)
    elif mode == "index":
      df = self.index_col(df)
      return self.drop_col(df, col_list=["index"])

  def evaluate(self, metrics="all", train_acc=True):
    train_x=self.train_df.drop(['nerdiness'], axis=1)
    train_y=self.train_df['nerdiness']

    if metrics == "all":
      metrics = [
          "accuracy", "precision", "recall", 
          "f1-score", "roc_auc", "mae"
      ]
    metrics_functions_map = {
        "accuracy": accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1-score": f1_score,
        "roc_auc": roc_auc_score,
        "mae": mean_absolute_error,
    }
    class_metrics = {
        "accuracy",
        "precision",
        "recall",
        "f1-score",
    }
    result_df = pd.DataFrame(
        columns = metrics + ["train_acc"]
    )

    for i, (train_index, val_index) in enumerate(self.kf.split(train_x)):
      X_train, X_test = train_x.loc[train_index], train_x.loc[val_index]
      y_train, y_test = train_y.loc[train_index], train_y.loc[val_index]

      self.model.fit(X_train, y_train, **self.fit_params)
      if self.model_type == 'rgr':
        predictions = self.model.predict(X_test)
      else:
        predictions = self.model.predict_proba(X_test)[:,1]

      row = {}
      for metric in metrics:
        if metric in class_metrics:
          score = metrics_functions_map[metric](
              y_test,
              np.round(predictions)
          )
        else:
          score = metrics_functions_map[metric](y_test, predictions)
        row[metric] = score
      result_df = result_df.append(
          row, ignore_index=True
      )

    result_df["fold"] = list(range(1, i+2))
    result_df = result_df.set_index("fold")

    ## add training accuracy
    mean = result_df.mean(axis=0)
    self.model.fit(train_x, train_y, **self.fit_params)
    if self.model_type == 'rgr':
      predictions = self.model.predict(train_x)
    else:
      predictions = self.model.predict_proba(train_x)[:,1]
    mean["train_acc"] = accuracy_score(np.round(predictions), train_y)
    result_df.loc["mean"] = mean

    return result_df

  def run(self, **kwargs):
    self.train_df = self.preprocess(self.train_df)
    self.train_df = self.train_df.dropna()
    self.train_df = self.train_df.reset_index()
    self.train_df = self.train_df.drop(["index"], axis=1)
    return self.evaluate(**kwargs)

  def train_best_model(self, n_runs=10, metric='roc_auc', threshold=None, **kwargs):
    best_model = None
    best_result = None

    self.train_df = self.preprocess(self.train_df)
    self.train_df = self.train_df.dropna()
    self.train_df = self.train_df.reset_index()
    self.train_df = self.train_df.drop(["index"], axis=1)

    for i in range(n_runs):
      result_df = self.evaluate(**kwargs)
      result_metric = result_df['roc_auc']['mean']

      if best_result is None or \
        best_result['roc_auc']['mean'] < result_metric:
          best_model = self.model
          best_result = result_df

      if threshold is not None and result_metric > threshold:
        break

    self.model = best_model
    return best_result

  def make_submission(self, test_df, submission_df):
    test_df = self.preprocess(test_df)

    # handle nan values
    imp = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    imp = imp.fit(test_df)
    test_df = imp.transform(test_df)
    
    if self.model_type == 'rgr':
      preds = self.model.predict(test_df)
    else:
      preds = self.model.predict_proba(test_df)[:,1]
    submission_df["nerdiness"] = preds
    return submission_df
