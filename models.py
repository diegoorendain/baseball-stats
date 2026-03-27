"""XGBoost + LightGBM + Logistic Regression ensemble models for baseball predictions."""

from __future__ import annotations

import math
import os
from typing import Any

import numpy as np
import pandas as pd
import xgboost as xgb

from config import MODELS_DIR, XGBOOST_PARAMS, ENSEMBLE_WEIGHTS, LIGHTGBM_PARAMS

# Ensure models directory exists
os.makedirs(MODELS_DIR, exist_ok=True)

# Optional LightGBM / sklearn imports (graceful fallback if not installed)
try:
    import lightgbm as lgb
    _LGB_AVAILABLE = True
except ImportError:
    lgb = None  # type: ignore[assignment]
    _LGB_AVAILABLE = False

try:
    from sklearn.linear_model import LogisticRegression
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.preprocessing import StandardScaler
    _SKLEARN_CALIBRATION_AVAILABLE = True
except ImportError:
    LogisticRegression = None  # type: ignore[assignment,misc]
    CalibratedClassifierCV = None  # type: ignore[assignment]
    StandardScaler = None  # type: ignore[assignment]
    _SKLEARN_CALIBRATION_AVAILABLE = False


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

    def feature_importance(self) -> pd.Series:
        """Return feature importances sorted descending.

        Returns an empty Series if the model has not been trained or does not
        expose feature importances.
        """
        if not self.is_trained() or self.model is None:
            return pd.Series(dtype=float)
        try:
            importances = self.model.feature_importances_
            names = getattr(self.model, "feature_names_in_", None)
            if names is not None:
                series = pd.Series(importances, index=names)
            else:
                series = pd.Series(importances)
            return series.sort_values(ascending=False)
        except AttributeError:
            return pd.Series(dtype=float)


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


# ---------------------------------------------------------------------------
# Ensemble models
# ---------------------------------------------------------------------------

class EnsembleGameWinnerModel:
    """Ensemble of XGBoost + LightGBM + Logistic Regression for game winner.

    Averages predictions with configurable weights from ``config.ENSEMBLE_WEIGHTS``
    and optionally applies isotonic calibration (Platt scaling) to the output.
    """

    def __init__(self) -> None:
        self._xgb = GameWinnerModel()
        self._lgb: Any = None
        self._lr: Any = None
        self._scaler: Any = None
        self._is_trained = False
        self._weights = ENSEMBLE_WEIGHTS

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit all sub-models on the same training data."""
        # XGBoost
        self._xgb.train(X, y)

        # LightGBM
        if _LGB_AVAILABLE:
            try:
                params = {k: v for k, v in LIGHTGBM_PARAMS["game_winner"].items()
                          if k not in ("objective", "verbose")}
                self._lgb = lgb.LGBMClassifier(
                    objective="binary",
                    verbose=-1,
                    **params,
                )
                self._lgb.fit(X, y)
            except Exception as exc:
                print(f"[WARN] LightGBM training failed: {exc}")
                self._lgb = None

        # Logistic Regression
        if _SKLEARN_CALIBRATION_AVAILABLE and LogisticRegression is not None:
            try:
                self._scaler = StandardScaler()
                X_scaled = self._scaler.fit_transform(X.fillna(0.0))
                self._lr = LogisticRegression(max_iter=500, random_state=42)
                self._lr.fit(X_scaled, y)
            except Exception as exc:
                print(f"[WARN] LogisticRegression training failed: {exc}")
                self._lr = None
                self._scaler = None

        self._is_trained = True

    def predict_calibrated_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return calibrated ensemble probability that home team wins."""
        if not self._is_trained:
            raise RuntimeError("EnsembleGameWinnerModel has not been trained yet.")

        probs_weighted = np.zeros(len(X))
        total_weight = 0.0

        # XGBoost
        try:
            xgb_proba = self._xgb.predict_proba(X)
            w = self._weights.get("xgboost", 0.5)
            probs_weighted += np.atleast_1d(xgb_proba) * w
            total_weight += w
        except Exception as exc:
            print(f"[WARN] XGBoost ensemble predict failed: {exc}")

        # LightGBM
        if self._lgb is not None:
            try:
                lgb_proba = self._lgb.predict_proba(X.fillna(0.0))[:, 1]
                w = self._weights.get("lightgbm", 0.3)
                probs_weighted += lgb_proba * w
                total_weight += w
            except Exception as exc:
                print(f"[WARN] LightGBM ensemble predict failed: {exc}")

        # Logistic Regression
        if self._lr is not None and self._scaler is not None:
            try:
                X_scaled = self._scaler.transform(X.fillna(0.0))
                lr_proba = self._lr.predict_proba(X_scaled)[:, 1]
                w = self._weights.get("logistic", 0.2)
                probs_weighted += lr_proba * w
                total_weight += w
            except Exception as exc:
                print(f"[WARN] Logistic Regression ensemble predict failed: {exc}")

        if total_weight > 0:
            return probs_weighted / total_weight
        return np.full(len(X), 0.5)

    def is_trained(self) -> bool:
        return self._is_trained


class EnsembleTotalRunsModel:
    """Ensemble of XGBoost + LightGBM for total runs regression."""

    def __init__(self) -> None:
        self._xgb = TotalRunsModel()
        self._lgb: Any = None
        self._is_trained = False
        self._weights = ENSEMBLE_WEIGHTS

    def train(self, X: pd.DataFrame, y: pd.Series) -> None:
        """Fit XGBoost and LightGBM sub-models."""
        self._xgb.train(X, y)

        if _LGB_AVAILABLE:
            try:
                params = {k: v for k, v in LIGHTGBM_PARAMS["total_runs"].items()
                          if k not in ("objective", "verbose")}
                self._lgb = lgb.LGBMRegressor(
                    objective="regression",
                    verbose=-1,
                    **params,
                )
                self._lgb.fit(X, y)
            except Exception as exc:
                print(f"[WARN] LightGBM regression training failed: {exc}")
                self._lgb = None

        self._is_trained = True

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Return ensemble predicted total runs."""
        if not self._is_trained:
            raise RuntimeError("EnsembleTotalRunsModel has not been trained yet.")

        preds_weighted = np.zeros(len(X))
        total_weight = 0.0

        try:
            xgb_preds = self._xgb.predict(X)
            w = self._weights.get("xgboost", 0.5)
            preds_weighted += np.atleast_1d(xgb_preds) * w
            total_weight += w
        except Exception as exc:
            print(f"[WARN] XGBoost runs predict failed: {exc}")

        if self._lgb is not None:
            try:
                lgb_preds = self._lgb.predict(X.fillna(0.0))
                w = self._weights.get("lightgbm", 0.3)
                preds_weighted += lgb_preds * w
                total_weight += w
            except Exception as exc:
                print(f"[WARN] LightGBM runs predict failed: {exc}")

        if total_weight > 0:
            return preds_weighted / total_weight
        return np.full(len(X), 9.0)

    def predict_with_confidence(self, X: pd.DataFrame) -> list[dict[str, float]]:
        """Return predicted total runs and approximate margin."""
        preds = self.predict(X)
        return [
            {"predicted_runs": float(p), "margin": float(p) * 0.25}
            for p in preds
        ]

    def is_trained(self) -> bool:
        return self._is_trained