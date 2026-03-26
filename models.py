"""XGBoost models for baseball predictions."""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from config import MODELS_DIR, XGBOOST_PARAMS

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)


class BaseModel:
    """Base class for all XGBoost baseball models."""

    model_name: str = "base"

    def __init__(self) -> None:
        self.model: xgb.XGBModel | None = None
        self._model_path: str = os.path.join(MODELS_DIR, f"{self.model_name}.json")

    def _build_model(self) -> xgb.XGBModel:
        """Instantiate the XGBoost estimator from config params."""
        raise NotImplementedError

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit the model on ``X`` and ``y``."""
        self.model = self._build_model()
        self.model.fit(X, y)

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return raw predictions for ``X``."""
        if not self.is_trained():
            raise RuntimeError(f"Model '{self.model_name}' is not trained yet.")
        return self.model.predict(X)  # type: ignore[union-attr]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability predictions (models that support it)."""
        return self.predict(X)

    def save(self, path: str | None = None) -> None:
        """Save the trained model to a .json file."""
        if self.model is None:
            raise RuntimeError("Cannot save: model has not been trained.")
        dest = path or self._model_path
        self.model.save_model(dest)

    def load(self, path: str | None = None) -> None:
        """Load a previously saved model from disk."""
        src = path or self._model_path
        self.model = self._build_model()
        self.model.load_model(src)

    def is_trained(self) -> bool:
        """Return True if the model is ready for prediction."""
        return self.model is not None

    def is_saved(self, path: str | None = None) -> bool:
        """Return True if a saved model file exists on disk."""
        return os.path.exists(path or self._model_path)

    def load_if_saved(self) -> bool:
        """Load the model from disk if a saved file exists. Return True on success."""
        if self.is_saved():
            self.load()
            return True
        return False

    def _xgb_params(self) -> dict[str, Any]:
        params = XGBOOST_PARAMS[self.model_name].copy()
        # Pop objective so it can be passed separately to the XGB constructor
        params.pop("objective", None)
        return params


# ---------------------------------------------------------------------------
# Concrete model classes
# ---------------------------------------------------------------------------

class TotalRunsModel(BaseModel):
    """Predicts total runs scored in a game (regression)."""

    model_name = "total_runs"

    def _build_model(self) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            **self._xgb_params(),
        )

    def predict_with_confidence(
        self, X: pd.DataFrame
    ) -> list[dict[str, float]]:
        """Return predicted total runs and an approximate error margin."""
        preds = self.predict(X)
        results = []
        for pred in preds:
            results.append(
                {
                    "predicted_runs": float(pred),
                    "margin": float(pred) * 0.25,  # ±25% margin estimate
                }
            )
        return results


class GameWinnerModel(BaseModel):
    """Predicts the probability that the home team wins (binary classification)."""

    model_name = "game_winner"

    def _build_model(self) -> xgb.XGBClassifier:
        return xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            **self._xgb_params(),
        )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return class predictions (0 = away wins, 1 = home wins)."""
        if not self.is_trained():
            raise RuntimeError(f"Model '{self.model_name}' is not trained yet.")
        return self.model.predict(X)  # type: ignore[union-attr]

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return probability that home team wins."""
        if not self.is_trained():
            raise RuntimeError(f"Model '{self.model_name}' is not trained yet.")
        return self.model.predict_proba(X)[:, 1]  # type: ignore[union-attr]


class HomeRunsModel(BaseModel):
    """Predicts expected home runs for a batter (Poisson rate)."""

    model_name = "home_runs"

    def _build_model(self) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            objective="count:poisson",
            **self._xgb_params(),
        )

    def predict_with_confidence(
        self, X: pd.DataFrame
    ) -> list[dict[str, float]]:
        """Return lambda (expected HRs) and P(HR>=1)."""
        lambdas = self.predict(X)
        results = []
        for lam in lambdas:
            lam = max(float(lam), 0.0)
            prob_at_least_one = 1.0 - math.exp(-lam)
            results.append(
                {
                    "expected_hr": lam,
                    "prob_hr_ge_1": prob_at_least_one,
                }
            )
        return results


class StrikeoutsModel(BaseModel):
    """Predicts expected strikeouts for a starting pitcher (regression)."""

    model_name = "strikeouts"

    def _build_model(self) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            objective="reg:squarederror",
            **self._xgb_params(),
        )

    def predict_expected(self, X: pd.DataFrame) -> list[float]:
        """Return expected strikeout counts."""
        return [max(0.0, float(v)) for v in self.predict(X)]


class HitsModel(BaseModel):
    """Predicts expected hits for a batter (Poisson rate)."""

    model_name = "hits"

    def _build_model(self) -> xgb.XGBRegressor:
        return xgb.XGBRegressor(
            objective="count:poisson",
            **self._xgb_params(),
        )

    def predict_with_confidence(
        self, X: pd.DataFrame
    ) -> list[dict[str, float]]:
        """Return lambda (expected hits) and P(H>=1)."""
        lambdas = self.predict(X)
        results = []
        for lam in lambdas:
            lam = max(float(lam), 0.0)
            prob_at_least_one = 1.0 - math.exp(-lam)
            results.append(
                {
                    "expected_hits": lam,
                    "prob_hit_ge_1": prob_at_least_one,
                }
            )
        return results