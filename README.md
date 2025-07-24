# 🧤 Premier League Goalkeeper Prediction

## Overview

This project aimed to predict how many goals Premier League goalkeepers would concede over selected matches using stats from the 2021/22 season. We analysed key goalkeeping metrics and applied machine learning models to evaluate predictive accuracy.

## Dataset

- Sourced from [FBRef](https://fbref.com/)
- Stats for 11 top PL goalkeepers
- Key features: Save %, xG Against, Clean Sheets, Post-Shot xG, Shots on Target, Goals Against, etc.
- Data prepared manually in a combined spreadsheet:  
  [View Data](https://docs.google.com/spreadsheets/d/1hjOioRM990E50bjjhuT73_lgtum50DR4/edit?usp=sharing)

## Models Used

- 🔹 **Linear Regression** (low accuracy)
- 🔹 **Gaussian Naive Bayes** – final model
  - ~93% accuracy
  - Best performance predicting high-concession matches
  - Confusion matrix used to evaluate prediction breakdown

## Key Insights

- High-performing keepers conceded fewer goals than expected overall  
- Save % didn't always correlate strongly with goals conceded  
- Prediction is more consistent over time periods than individual matches

## Tools

- Python (Pandas, scikit-learn, matplotlib)
- Excel & OpenRefine (data prep)
- GitLab for collaboration
