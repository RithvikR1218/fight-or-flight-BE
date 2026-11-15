# SafeRoute risk graphs (res-10 model, regenerated 2026-11-15)

Made from the res-10 scoring-v2 code (H3_RES = 10, COVERAGE_RES = 9, log risk scale + danger boost,
linear blue-light score drop) with crime.db sha256 777ae135...cffd03c. The heatmaps replay
**2023-12-15** (crime history to 2023-12-14), because LAPD data stops in 2024 and the danger boost
only shows on dates with recent crimes. USC campus / USC Village tiles (43 at res 10) are shown at 0.

## calculation/  how one tile's risk is computed
1_severity.png, 2_recency.png, 3_time_of_day.png, 4_risk_scale.png, 5_danger_boost.png,
6_bluelight.png, 7_risk_distribution.png, 8_hourly_profiles.png

## heatmap/  every H3 tile
- risk_map.html: interactive map (hover a tile, drag the hour slider, click to pin). Open in a browser.
- risk_map_10pm.png: snapshot of the interactive map at 22:00.
- 9_cell_heatmap.png: static heatmap at 02:00, 10:00, 18:00 and 22:00.

## model/  explore_train_infer notebook, re-executed end to end (retrains the model)
nb_*.png are the notebook's figures (ROC/PR/calibration, SHAP, feature correlation, sanity checks,
partial dependence, hour profiles, live heatmap for 2026-11-14). calibration.png / shap_*.png are
from models/train_xgb.py. 2023 test: ROC-AUC 0.849, PR-AUC 11.6x base rate, top 10% of
cell-hours hold 54% of incidents; all 6 sanity checks pass.
The notebook's two blue-light cells were updated from the old bluelight_factor(dist) call to the
current score-drop rule so it runs.

## Re-run
From a checkout that has the res-10 code as an `algorithm` package:
    python -m algorithm.scoring.risk_graphs --out <dir>
    python -m algorithm.scoring.risk_map --out <dir>/risk_map.html
