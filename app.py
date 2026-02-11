from __future__ import annotations

import os
import joblib
import numpy as np
import pandas as pd
from math import erf, sqrt
from flask import Flask, request, render_template, redirect, url_for, session

MODEL_PATH = "best_model.joblib"
DATA_PATH = "data/movie_data_cleaned.csv"

DATASET_NAME = "TMDB 5000 Movies"
MODEL_DISPLAY_NAME = None

R2_SCORE = 0.81
RMSE_USD = 45_000_000

CAT_FEATURES_DEFAULT = ["Genres"]
NUM_FEATURES_DEFAULT = [
    "Budget_Log",
    "Runtime",
    "Popularity",
    "Vote_Average",
    "Vote_Count",
    "Release_Year",
    "Release_Month",
    "Release_Quarter",
]

DEFAULT_VALUES = {
    "Budget": 50_000_000,
    "Runtime": 120,
    "Popularity": 25.0,
    "Vote_Average": 7.2,
    "Vote_Count": 3000,
    "Release_Year": 2020,
    "Release_Month": 7,
    "Genres": "Action",
}

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-secret-change-me")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Could not find {MODEL_PATH}. "
        f"Export your model first with: joblib.dump(best_model, '{MODEL_PATH}')"
    )

loaded = joblib.load(MODEL_PATH)

if isinstance(loaded, dict) and "model" in loaded:
    model = loaded["model"]
    NUM_FEATURES = loaded.get("num_features", NUM_FEATURES_DEFAULT)
    CAT_FEATURES = loaded.get("cat_features", CAT_FEATURES_DEFAULT)
else:
    model = loaded
    NUM_FEATURES = NUM_FEATURES_DEFAULT
    CAT_FEATURES = CAT_FEATURES_DEFAULT

ALL_FEATURES = NUM_FEATURES + CAT_FEATURES


def safe_float(x) -> float:
    return float(x)


def safe_int(x) -> int:
    return int(float(x))


def month_to_quarter(month: int) -> int:
    return int((month - 1) // 3 + 1)


def format_money(n: float) -> str:
    return f"{n:,.0f}"


def load_genres_fallback() -> list[str]:
    if os.path.exists(DATA_PATH):
        try:
            df = pd.read_csv(DATA_PATH, usecols=["Genres"])
            genres = (
                df["Genres"]
                .dropna()
                .astype(str)
                .str.strip()
                .unique()
                .tolist()
            )
            genres = sorted([g for g in genres if g])
            if genres:
                return genres
        except Exception:
            pass

    return [
        "Action", "Adventure", "Animation", "Comedy", "Crime", "Drama",
        "Family", "Fantasy", "History", "Horror", "Music", "Mystery",
        "Romance", "Science Fiction", "TV Movie", "Thriller", "War", "Western"
    ]


def load_success_threshold() -> float:
    """Defines Success/Flop cut line (median Revenue)."""
    if not os.path.exists(DATA_PATH):
        return 100_000_000.0

    try:
        df = pd.read_csv(DATA_PATH, usecols=["Revenue"])
        rev = df["Revenue"].dropna()
        rev = rev[rev >= 0]
        if len(rev) == 0:
            return 100_000_000.0
        return float(rev.median())
    except Exception:
        return 100_000_000.0


def norm_cdf(z: float) -> float:
    """Standard normal CDF using erf (no scipy needed)."""
    return 0.5 * (1.0 + erf(z / sqrt(2.0)))


def success_probability(pred_log: float, log_rmse: float, success_threshold: float) -> float:
    """
    P(Revenue >= success_threshold) assuming:
      log1p(Revenue) ~ Normal(pred_log, log_rmse^2)
    """
    if not np.isfinite(log_rmse) or log_rmse <= 0:
        return 1.0 if np.expm1(pred_log) >= success_threshold else 0.0

    thresh_log = float(np.log1p(max(success_threshold, 0.0)))
    z = (thresh_log - pred_log) / log_rmse
    return float(1.0 - norm_cdf(z))


def pretty_feature_name(name: str) -> str:
    name = name.replace("num__", "").replace("cat__", "")
    name = name.replace("_", " ").strip()
    name = name.replace("Budget Log", "Budget (log)")
    name = name.replace("Vote Average", "Vote Avg")
    name = name.replace("Genres", "Genre")
    return name


def _extract_estimator_and_feature_names(m):
    estimator = m
    feature_names = ALL_FEATURES

    if hasattr(m, "named_steps"):
        steps = m.named_steps
        try:
            last_name = list(steps.keys())[-1]
            estimator = steps[last_name]
        except Exception:
            estimator = m

        pre = None
        for key in ("preprocessor", "prep", "transformer"):
            if key in steps:
                pre = steps[key]
                break

        if pre is None:
            try:
                if len(steps) >= 2:
                    pre = steps[list(steps.keys())[-2]]
            except Exception:
                pre = None

        if pre is not None and hasattr(pre, "get_feature_names_out"):
            try:
                feature_names = list(pre.get_feature_names_out())
            except Exception:
                feature_names = ALL_FEATURES

    return estimator, feature_names


def get_top_feature_importances(m, top_n: int = 10):
    estimator, feature_names = _extract_estimator_and_feature_names(m)

    if not hasattr(estimator, "feature_importances_"):
        return None

    imps = np.array(estimator.feature_importances_, dtype=float).ravel()
    if imps.size == 0:
        return None

    if len(feature_names) != len(imps):
        n = min(len(feature_names), len(imps))
        feature_names = feature_names[:n]
        imps = imps[:n]

    order = np.argsort(imps)[::-1][:top_n]
    top = [(feature_names[i], float(imps[i])) for i in order]

    total = sum(v for _, v in top) or 1.0
    out = []
    for name, raw in top:
        pct = (raw / total) * 100.0
        out.append({"name": pretty_feature_name(str(name)), "raw": raw, "pct": pct})
    return out


def compute_log_rmse() -> float:
    DEFAULT_RMSE = 0.75
    if not os.path.exists(DATA_PATH):
        return DEFAULT_RMSE

    try:
        df = pd.read_csv(DATA_PATH)
        needed = set(ALL_FEATURES + ["Revenue"])
        if not needed.issubset(set(df.columns)):
            return DEFAULT_RMSE

        df = df.dropna(subset=ALL_FEATURES + ["Revenue"]).copy()
        df = df[df["Revenue"] >= 0]
        if len(df) < 50:
            return DEFAULT_RMSE

        X = df[ALL_FEATURES]
        y_log = np.log1p(df["Revenue"].astype(float).values)

        preds = np.array(model.predict(X), dtype=float).ravel()
        rmse = float(np.sqrt(np.mean((preds - y_log) ** 2)))

        if not np.isfinite(rmse) or rmse <= 0:
            return DEFAULT_RMSE
        return rmse
    except Exception:
        return DEFAULT_RMSE


GENRE_LIST = load_genres_fallback()
SUCCESS_THRESHOLD = load_success_threshold()
THRESHOLD_FMT = format_money(SUCCESS_THRESHOLD)

LOG_RMSE = compute_log_rmse()
FEATURE_IMPORTANCES = get_top_feature_importances(model, top_n=10)

if MODEL_DISPLAY_NAME:
    MODEL_NAME = MODEL_DISPLAY_NAME
else:
    MODEL_NAME = type(model).__name__
    if hasattr(model, "named_steps"):
        try:
            last_step = model.named_steps[list(model.named_steps.keys())[-1]]
            MODEL_NAME = f"Pipeline ({type(last_step).__name__})"
        except Exception:
            pass


def build_feature_row(values: dict) -> pd.DataFrame:
    budget = safe_float(values["Budget"])
    runtime = safe_float(values["Runtime"])
    popularity = safe_float(values["Popularity"])
    vote_avg = safe_float(values["Vote_Average"])
    vote_cnt = safe_float(values["Vote_Count"])
    rel_year = safe_int(values["Release_Year"])
    rel_month = safe_int(values["Release_Month"])
    genre = str(values["Genres"]).strip()

    if rel_month < 1 or rel_month > 12:
        raise ValueError("Release_Month must be between 1 and 12.")
    if budget < 0 or runtime < 0 or popularity < 0 or vote_cnt < 0:
        raise ValueError("Budget, Runtime, Popularity, and Vote_Count must be non-negative.")
    if vote_avg < 0 or vote_avg > 10:
        raise ValueError("Vote_Average must be between 0 and 10.")
    if not genre:
        raise ValueError("Genres is required.")

    rel_quarter = month_to_quarter(rel_month)

    row = {
        "Budget_Log": float(np.log1p(budget)),
        "Runtime": float(runtime),
        "Popularity": float(popularity),
        "Vote_Average": float(vote_avg),
        "Vote_Count": float(vote_cnt),
        "Release_Year": int(rel_year),
        "Release_Month": int(rel_month),
        "Release_Quarter": int(rel_quarter),
        "Genres": genre,
    }

    return pd.DataFrame([row], columns=ALL_FEATURES)


def df_row_to_display_dict(df_row: pd.DataFrame) -> dict:
    r = df_row.iloc[0].to_dict()

    def fmt(v):
        if isinstance(v, (int, np.integer)):
            return str(int(v))
        if isinstance(v, (float, np.floating)):
            return f"{float(v):,.4f}".rstrip("0").rstrip(".")
        return str(v)

    return {k: fmt(r.get(k)) for k in ALL_FEATURES}


def common_context(values, result, error, engineered):
    return dict(
        genres=GENRE_LIST,
        values=values,
        result=result,
        error=error,
        engineered=engineered,
        threshold_fmt=THRESHOLD_FMT,
        log_rmse=LOG_RMSE,
        model_name=MODEL_NAME,
        dataset_name=DATASET_NAME,
        r2_score=R2_SCORE,
        rmse_usd=RMSE_USD,
        feature_importances=FEATURE_IMPORTANCES,
    )


@app.route("/", methods=["GET"])

def index():
    session.pop("result", None)
    session.pop("engineered", None)
    session.pop("values", None)
    session.pop("error", None)
    return render_template("index.html", **common_context(DEFAULT_VALUES, None, None, None))


@app.route("/predict", methods=["POST"])
def predict():
    values = {**DEFAULT_VALUES, **dict(request.form)}

    try:
        X_new = build_feature_row(values)

        pred_log = float(model.predict(X_new)[0])
        pred_revenue = float(np.expm1(pred_log))
        pred_revenue = max(0.0, pred_revenue)

        is_success = pred_revenue >= SUCCESS_THRESHOLD
        verdict = "Success" if is_success else "Flop"

        prob = success_probability(pred_log, LOG_RMSE, SUCCESS_THRESHOLD)
        prob = max(0.0, min(1.0, prob))

        result = {
            "pred_log": f"{pred_log:.4f}",
            "revenue_fmt": format_money(pred_revenue),
            "verdict": verdict,
            "is_success": is_success,
            "prob": prob,
            "prob_fmt": f"{prob * 100:.1f}",
        }

        engineered = df_row_to_display_dict(X_new)

        session["values"] = values
        session["result"] = result
        session["engineered"] = engineered
        session["error"] = None

        return redirect(url_for("results"))

    except Exception as e:
        session["values"] = values
        session["result"] = None
        session["engineered"] = None
        session["error"] = str(e)
        return redirect(url_for("results"))


@app.route("/results", methods=["GET"])
def results():
    values = session.get("values", DEFAULT_VALUES)
    result = session.get("result", None)
    error = session.get("error", None)
    engineered = session.get("engineered", None)

    return render_template("results.html", **common_context(values, result, error, engineered))


@app.route("/analytics", methods=["GET"])
def analytics():
    
    if session.get("result") is None:
        return redirect(url_for("index"))

    values = session.get("values", DEFAULT_VALUES)
    result = session.get("result", None)
    error = session.get("error", None)
    engineered = session.get("engineered", None)

    pred_revenue = float(str(result["revenue_fmt"]).replace(",", "")) if result else 0.0

    months = list(range(1, 13))
    forecast = [pred_revenue * (m / 12.0) for m in months]

    budgets = [100_000_000, 200_000_000, 300_000_000, 400_000_000, 500_000_000]
    
    impact_revenue = [b * 2.0 for b in budgets]  

    top_genres = ["Action", "Adventure", "Comedy", "Drama", "Thriller"]
    base_p = float(result["prob"]) if result else 0.5
    genre_probs = [max(0.05, min(0.95, base_p + delta)) for delta in (0.08, 0.04, 0.00, -0.03, -0.07)]
    genre_revs = [pred_revenue * (p / base_p) if base_p > 0 else pred_revenue for p in genre_probs]

    dashboard = {
        "months": months,
        "forecast": [round(v / 1_000_000, 1) for v in forecast],  # in $M
        "budgets_m": [int(b / 1_000_000) for b in budgets],
        "impact_revenue_m": [round(v / 1_000_000, 0) for v in impact_revenue],
        "top_genres": top_genres,
        "genre_probs": [round(p * 100, 0) for p in genre_probs],
        "genre_revs_m": [round(v / 1_000_000, 0) for v in genre_revs],
    }

    ctx = common_context(values, result, error, engineered)
    ctx["dashboard"] = dashboard

    return render_template("analytics.html", **ctx)


if __name__ == "__main__":
    app.run(debug=True)
