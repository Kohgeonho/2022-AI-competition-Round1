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
from evaluator.ensemble import EnsembleModel

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

### Ensemble Model

#### Update [best_models_config.yaml](best_models_config.yaml)
- 각 모델별로 best optimization을 update 해준다
- version, best_score 등도 같이 적어주면 관리하기 편함

#### Run Ensemble Model
```python
from evaluator.ensemble import EnsembleModel

evaluator = Evaluator(
    **EnsembleModel(train_df, models=['lgbm', 'xgb', 'et', 'rf']).get_model()
)
evaluator.run()
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

### Version 0.4
- CLF model이 0과 1의 label이 아닌 probability score를 반환하도록 변경(2022.08.15) [[PR Link](https://github.com/Kohgeonho/2022-AI-competition-Round1/pull/3)]

### Version 0.6
- best_models_config.yaml의 정보를 이용한 Ensemble 모델을 학습하는 EnsembleModel 모듈 추가(2022.08.20) [[PR Link](https://github.com/Kohgeonho/2022-AI-competition-Round1/pull/7)]
