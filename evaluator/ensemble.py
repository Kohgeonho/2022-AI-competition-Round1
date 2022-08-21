import numpy as np
import yaml

from sklearn.ensemble import VotingRegressor
from evaluator.evaluator import Model

class VotingModel(VotingRegressor):
  def __init__(self, model_types, estimators):
    self.model_types = model_types
    super().__init__(estimators)

  def _validate_estimators(self):
      if self.estimators is None or len(self.estimators) == 0:
          raise ValueError(
              "Invalid 'estimators' attribute, 'estimators' should be a list"
              " of (string, estimator) tuples."
          )
      names, estimators = zip(*self.estimators)
      # defined by MetaEstimatorMixin
      self._validate_names(names)

      has_estimator = any(est != "drop" for est in estimators)
      if not has_estimator:
          raise ValueError(
              "All estimators are dropped. At least one is required "
              "to be an estimator."
          )

      return names, estimators

  def predict(self, X):
    return np.average(
      np.asarray([
        est.predict(X) if model_type=="rgr" else est.predict_proba(X)[:,1]
        for model_type, est in zip(self.model_types, self.estimators_)
      ]).T, 
      axis=1, 
      weights=self._weights_not_none
    )

class EnsembleModel():
  def __init__(
    self, 
    train_df,
    models=["lgbm", "xgb", "cat", "rf", "et"]
  ):
    with open('evaluator/best_models_config.yaml') as f:
      model_configs = yaml.load(f, Loader=yaml.FullLoader)

    self.train_df = train_df

    estimators = []
    model_types = []
    for model_name in models:
        configs = model_configs[model_name][0]
        model_type = configs["type"]
        params = configs["best_params"]
        
        estimators.append((
            model_name, 
            Model(train_df, model_name, model_type, **params).get_model()['model']
        ))
        model_types.append(model_type)

    self.ensemble_model = VotingModel(
        estimators=estimators,
        model_types=model_types,
    )

  def get_model(self):
    return {
        'train_df': self.train_df,
        'model': self.ensemble_model,
        'model_name': 'ensemble',
        'model_type': 'rgr',
    }