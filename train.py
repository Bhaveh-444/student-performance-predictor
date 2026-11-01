"""Train a model on the Indian student lifestyle dataset (any CSV in data/)."""
import glob

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

path = sorted(glob.glob("data/*.csv"))[0]
df = pd.read_csv(path, sep=None, engine="python")  # auto-detects , or ;
df.columns = [c.strip() for c in df.columns]
print(f"Loaded {path}: {df.shape}\nColumns: {list(df.columns)}")

# Find columns by name so small naming differences don't break anything
target = next(c for c in df.columns if "gpa" in c.lower())
drop = [c for c in df.columns if "id" == c.lower().replace("_", "").replace(" ", "")[-2:]
        or c.lower().replace("_", "").replace(" ", "") in ("studentid", "id")]
stress = [c for c in df.columns if "stress" in c.lower()]  # an outcome, not a habit -> excluded
features = [c for c in df.columns if c not in [target, *drop, *stress]]
hour_feats = [c for c in features if "hour" in c.lower() and pd.api.types.is_numeric_dtype(df[c])]
print(f"Target: {target} | Features: {features} | Hour features: {hour_feats}")

df = df.dropna(subset=[target] + features)
X, y = df[features], df[target]
cat = X.select_dtypes(exclude="number").columns.tolist()
num = [c for c in features if c not in cat]


def make_pipeline(model):
    pre = ColumnTransformer([
        ("cat", OneHotEncoder(handle_unknown="ignore"), cat),
        ("num", StandardScaler(), num),
    ])
    return Pipeline([("pre", pre), ("model", model)])


MODELS = {
    "Ridge": Ridge(alpha=1.0),
    "RandomForest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42),
    "GradientBoosting": GradientBoostingRegressor(n_estimators=200, max_depth=2,
                                                  learning_rate=0.05, random_state=42),
}
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

rows, fitted = [], {}
for name, model in MODELS.items():
    pipe = make_pipeline(model)
    cv = cross_val_score(pipe, X_train, y_train, cv=5, scoring="neg_mean_absolute_error")
    pipe.fit(X_train, y_train)
    pred = pipe.predict(X_test)
    rows.append({"model": name, "cv_MAE": round(-cv.mean(), 3),
                 "test_MAE": round(mean_absolute_error(y_test, pred), 3),
                 "test_R2": round(r2_score(y_test, pred), 3)})
    fitted[name] = pipe

results = pd.DataFrame(rows)
print("\n", results.to_string(index=False))
results.to_csv("models/results.csv", index=False)

best = results.sort_values("test_MAE").iloc[0]
pipe = fitted[best.model]
print(f"\nBest model: {best.model}")

imp = permutation_importance(pipe, X_test, y_test, n_repeats=20, random_state=42,
                             scoring="neg_mean_absolute_error")
(pd.DataFrame({"feature": X_test.columns, "importance": imp.importances_mean})
 .sort_values("importance", ascending=False).to_csv("models/importance.csv", index=False))

pipe.fit(X, y)  # refit on all data for the app
joblib.dump({
    "model": pipe, "features": features, "hour_features": hour_feats, "target": target,
    "defaults": X.median(numeric_only=True).to_dict() | X[cat].mode().iloc[0].to_dict() if cat
    else X.median(numeric_only=True).to_dict(),
    "ranges": {c: (float(df[c].min()), float(df[c].max())) for c in hour_feats},
    "y_range": (float(y.min()), float(y.max())),
    "best_model": best.model, "test_MAE": float(best.test_MAE),
}, "models/advice_model.pkl")
print("Saved models/advice_model.pkl")