# Post-Clinical Mobility Assessment
### Human Activity Recognition from IMU Sensor Data

A machine learning pipeline that classifies human physical activities using accelerometer and gyroscope data from smartphones. Directly applicable to clinical gait analysis, fall detection, and movement disorder monitoring in conditions such as Parkinson's disease.

---

## Results

| Model | Accuracy | Macro-F1 |
|---|---|---|
| Random Forest | 92.57% | — |
| XGBoost | **95.51%** | **0.9548** |
| SVM (RBF) | 95.50% | 0.9540 |
<img width="3859" height="1763" alt="fig4_model_comparison_multi" src="https://github.com/user-attachments/assets/a24c96ea-c8a7-46f3-a320-fcb74dc0ccb8" />


Evaluated using **Leave-One-Subject-Out (LOSO)** cross-validation — 30-fold, one subject held out per fold. McNemar test confirms XGBoost vs baseline significance at **p = 0.014**.

---

## Subject-Independent Evaluation

| Protocol | Detail |
|---|---|
| Method | LOSO Cross-Validation |
| Subjects | 30 (ages 19–48) |
| Folds | 30 (one subject per fold) |
| Sensor | Samsung Galaxy S2 @ 50Hz |

No subject overlap between folds — model generalises to individuals it has never seen before, simulating real clinical deployment.

---

## Per-Activity Performance (XGBoost — LOSO)

| Activity | F1-Score |
|---|---|
| LAYING | 1.00 |
| WALKING | 0.95 |
| WALKING_DOWNSTAIRS | 0.94 |
| WALKING_UPSTAIRS | 0.93 |
| STANDING | 0.91 |
| SITTING | 0.89 |
<img width="2959" height="1462" alt="fig9_per_activity_f1" src="https://github.com/user-attachments/assets/f04c2a34-7f8b-4122-bc71-38f919c9ec69" />

---

## Key Findings

**1. SITTING vs STANDING hardest to distinguish (F1: 0.89 and 0.91)**
Both activities produce similar low-magnitude acceleration patterns — consistent with known challenges in static activity recognition.

**2. Raw accelerometer signals reveal distinct activity signatures:**
- WALKING: high amplitude oscillating signals from rhythmic foot strikes
- SITTING: signals settle from high to near-zero as body stabilises
- LAYING: very low amplitude flat signals indicating minimal movement

**3. Gravity components dominate feature importance**
tGravityAcc-min()-X was the single strongest predictor (TreeSHAP) — body orientation relative to gravity is the primary discriminator between activities.
<img width="3558" height="5303" alt="fig1_raw_signals_all6" src="https://github.com/user-attachments/assets/714bc050-5a5a-4178-a6db-87c3fca2ee00" />
<img width="2658" height="2063" alt="fig3_tsne_embedding" src="https://github.com/user-attachments/assets/bfb74264-7476-48ec-8e7c-b5a7dc09ace2" />

---

## Clinical Relevance

- Fall detection systems for elderly patients
- Gait analysis for Parkinson's disease monitoring
- Rehabilitation progress tracking
- Remote patient monitoring via wearable sensors
<img width="2961" height="2064" alt="fig7_shap_global" src="https://github.com/user-attachments/assets/58ae06b1-3293-4a7e-801c-c3315559efd9" />

---

## Dataset

- Source: UCI Machine Learning Repository
- 30 subjects, ages 19–48
- Sampled at 50Hz using Samsung Galaxy S2
- 561 features extracted from accelerometer and gyroscope signals
- [UCI HAR Dataset](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones)

---

## Pipeline

1. Load raw accelerometer and gyroscope features
2. Visualise raw signals and t-SNE feature embedding
3. Train and compare RF, XGBoost, LightGBM, SVM, MLP classifiers
4. Evaluate using 30-fold LOSO protocol
5. Analyse per-activity F1 and TreeSHAP feature importance
6. McNemar statistical significance testing

---

## Technologies

- Python 3.x · pandas · numpy
- scikit-learn · XGBoost · LightGBM
- SHAP · matplotlib · seaborn
- Jupyter Notebook

---

## How to Reproduce

1. Clone this repository
2. `pip install -r requirements.txt`
3. Download UCI HAR Dataset from the link above
4. Place `UCI HAR Dataset` folder in project root
5. Run `01_har_classifier.ipynb` sequentially

---

## Author

**Kritika Patidar**
B.Tech Biomedical Engineering 
BS Data Science & Applications — IIT Madras
kritika.bme@gmail.com
