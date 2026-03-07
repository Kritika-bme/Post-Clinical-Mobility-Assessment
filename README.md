# Human Activity Recognition from IMU Sensor Data

## Overview
A machine learning pipeline that classifies human physical activities 
using accelerometer and gyroscope data from smartphones. Directly applicable 
to clinical gait analysis, fall detection, and movement disorder monitoring 
in conditions such as Parkinson's disease.

## Results
| Model | Accuracy |
|-------|----------|
| Random Forest | 92.57% |
| XGBoost | 93.79% |

Evaluated using subject independent protocol — training and testing on 
completely different individuals, simulating real clinical deployment.

## Subject Independent Evaluation
| Split | Subjects | Samples |
|-------|----------|---------|
| Training | 21 subjects | 7,352 |
| Testing | 9 unseen subjects | 2,947 |

No subject overlap between train and test sets confirming the model 
generalises to individuals it has never seen before.

## Per-Activity Performance (XGBoost)
| Activity | F1-Score |
|----------|----------|
| LAYING | 1.00 |
| WALKING | 0.95 |
| WALKING_DOWNSTAIRS | 0.94 |
| WALKING_UPSTAIRS | 0.93 |
| STANDING | 0.91 |
| SITTING | 0.89 |

## Key Findings
1. SITTING and STANDING were hardest to distinguish (F1: 0.89 and 0.91)
   Both activities produce similar low-magnitude acceleration patterns
   since the person is relatively still — consistent with known challenges
   in static activity recognition.

2. Raw accelerometer analysis reveals distinct signal patterns:
   - WALKING: high amplitude oscillating signals from rhythmic foot strikes
   - SITTING: signals settle from high to near-zero as body stabilises
   - LAYING: very low amplitude flat signals indicating minimal movement

3. Gravity acceleration components (tGravityAcc-mean()-X, tGravityAcc-max()-X)
   were the strongest predictors — body orientation relative to gravity
   is the primary discriminator between activities.

## Clinical Relevance
This project demonstrates the core methodology behind:
- Fall detection systems for elderly patients
- Gait analysis for Parkinson's disease monitoring
- Rehabilitation progress tracking
- Remote patient monitoring via wearable sensors

## Dataset
- Source: UCI Machine Learning Repository
- 30 subjects, ages 19-48
- Sampled at 50Hz using Samsung Galaxy S2
- 7,352 training samples, 2,947 test samples
- 561 features extracted from accelerometer and gyroscope signals
- Link: https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones

## Pipeline
1. Load raw accelerometer and gyroscope features from UCI HAR Dataset
2. Visualize raw accelerometer signals per activity category
3. Train and compare Random Forest and XGBoost classifiers
4. Evaluate using subject independent protocol
5. Analyze per-activity F1 scores and feature importance

## Technologies
- Python 3.13
- pandas, numpy
- scikit-learn, XGBoost
- matplotlib, seaborn
- Jupyter Notebook

## How to Reproduce
1. Clone this repository
2. pip install -r requirements.txt
3. Download UCI HAR Dataset from the link above
4. Place UCI HAR Dataset folder in project root directory
5. Run 01_har_classifier.ipynb sequentially

## Author
Kritika
B.Tech Biomedical Engineering
