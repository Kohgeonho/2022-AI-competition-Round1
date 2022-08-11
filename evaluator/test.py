import unittest

import numpy as np
import pandas as pd
from evaluator import Evaluator, Model

train_df = pd.read_csv('competition_data/train.csv')
test_df = pd.read_csv("competition_data/test.csv")
submission_df = pd.read_csv("competition_data/sample_submission.csv")


class EvaluatorTest(unittest.TestCase): 

    def run_baseline(self, model_name):
        print(f"running {model_name} regressor...", end=" ")
        params = {
            "n_estimators": 10
        }
        evaluator = Evaluator(
            **Model(train_df, model_name, "rgr", **params).get_model()
        )
        evaluator.run()
        print("OK")

    def test_runs(self):
        for model_name in ("lgbm", "xgb", "et", "rf", "cat"):
            self.run_baseline(model_name)


# unittest를 실행
if __name__ == '__main__':  
    unittest.main()