## Note
- evaluator를 수정한 뒤에는 test code 돌려보기
```python
!python evaluator/test.py
```
- evaluator를 변경할 때에는 branch 생성 -> PR 생성 -> Merge 순서로 작업 진행하기 ([링크](https://www.notion.so/simya/Github-Policy-82fc68ae4b38425486d4cb3334dbb77c))
- 변경 내용 Notion 및 현재 문서에 업데이트 하기

## Usage
### Mount Google Drive

```python
import os
import numpy as np
import pandas as pd
import warnings

from google.colab import drive

warnings.filterwarnings('ignore')
drive.mount("/content/drive")

os.chdir("drive/MyDrive/competition/2022-AI-competition-Round1") # Local Path
os.listdir()
```

### install requirements
```python
!pip install -r evaluator/requirements.txt
```

### Module Import (with Data)
```python
from evaluator.evaluator import Evaluator, Model

train_df = pd.read_csv('competition_data/train.csv')
test_df = pd.read_csv("competition_data/test.csv")
submission_df = pd.read_csv("competition_data/sample_submission.csv")
```

### Evaluate Model

> Model의 input parameter는 다음 중 하나를 선택해서 넣으면 된다  
> **model_name** : [”lgbm”, “xgb”, “rf”, “et”, “cat”]  
> **model_type** : [”clf”, “rgr”]  

#### Baseline
```python
evaluator = Evaluator(
    **Model(train_df, "lgbm", "rgr").get_model()
)
evaluator.run()
```

#### With Initial Parameters
```python
params = {
  "objective": "binary",
  "n_estimators": 500,
  'learning_rate': 0.026332779906149555,
  'num_leaves': 955,
  'reg_alpha': 6.90331310095056e-08,
  'reg_lambda': 2.30837413695962e-06
}
evaluator = Evaluator(
    **Model(train_df, "lgbm", "clf", **params).get_model(),
)
evaluator.run()
```

### Optimize Model

> initial_param은 각각 (name, dtype, value/range)로 이루어져 있다. valid dtype은 다음과 같다  
> **dtype** : [”static”, “int”, “float”, “log”, “categorical”]  

#### Initialize Model
```python
model = Model(train_df, "lgbm", "rgr")
```
#### Optimize
```python
initial_params = (
    ("n_estimators", "static", 100),
    ("objective", "static", "binary"),
    ("metric", "static", "auc"),
    ("learning_rate", "log", (1e-5, 1.0)),
    ("num_leaves", "int", (300, 2000)),
)
model.optimize(initial_params, n_trials=20)
```

#### Evaluate Optimized Model
```python
Evaluator(
    **model.get_model()
).run()
```

### Create Submission File
```python
submission_df = evaluator.make_submission(test_df, submission_df)
submission_df.to_csv("submission/2022-08-05_LGBM_optim_200.csv", index=False)
```

## Updates
### Version 0.3
- Version 0.3 Github에 업로드 (2022.08.10)
- [Hotfix] catboost 이외의 모델에 에러 발생하던 것 수정(2022.08.11) [[PR Link](https://github.com/Kohgeonho/2022-AI-competition-Round1/pull/1)]

## Appendix
### A
Evaluator Update할 때 사용하면 좋을 template
```python
from evaluator.evaulator import Model, Evaluator, Optimizer

class MyEvaluator(Evaluator):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

class MyModel(Model):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

  def optimize(self, initial_params, **kwargs):
    self.optimizer = MyOptimizer(
        self.train_df, 
        initial_params, 
        self.model_name,
        self.model_type,
    )
    best_params = self.optimizer.run(**kwargs)
    self.__init__(self.train_df, self.model_name, self.model_type, **best_params)

class MyOptimizer(Optimizer):
  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)

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
    result_df = MyEvaluator(
        **MyModel(self.train_df, self.model_name, self.model_type, **self.params).get_model()
    ).run(train_acc=False)

    return result_df["roc_auc"]["mean"]
```
```python
MyEvaluator(
    **MyModel(train_df, "lgbm", "clf").get_model()
).run()
```
