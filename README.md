# Student Performance Coach

Predicts a student's final grade (0-20) from habits and background, then tells the student
which habit changes are associated with the biggest improvement.

## Approach
- Dataset: UCI Student Performance (`student-mat.csv`, 395 students).
- Two model families compared: a **benchmark** using earlier grades (G1, G2) and an **advice**
  model using only habits/background, so its suggestions are actionable.
- Models: Ridge, Random Forest, Gradient Boosting. Metrics: MAE, R².
- Explainability: permutation importance. Advice: re-predict after one realistic habit change
  and rank by predicted gain.

## Results
(paste the table from `models/results.csv` here)

## Run it
```bash
pip install -r requirements.txt
# place student-mat.csv in data/
python train.py
streamlit run app.py
```

## Limitations
- Associations, not causation.
- Data is from two Portuguese schools; may not generalise to other colleges.
- Small dataset (395 rows).

## Future work
SHAP explanations, a classification (pass/fail) mode, more subjects (student-por.csv).
