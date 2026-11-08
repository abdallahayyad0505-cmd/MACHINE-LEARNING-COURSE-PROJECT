# ML 402331 — IoT Machine Learning Project (Phase 1)

**Student:** Abdallah Ayyad (`2310216`)  
**Course:** ML 402331 (Fall 2026/2027) — American University of Madaba (AUM)  
**Instructor:** Dr. Mohd Taha  
**Project Idea:** Idea #1 — Smart Room Occupancy  
**GitHub Repository:** https://github.com/abdallahayyad0505-cmd/MACHINE-LEARNING-COURSE-PROJECT

## Overview
This project uses four simple room sensors (`Temperature`, `Humidity`, `Light`, and `CO2`) to predict room occupancy (`People_Count` in Phase 1 using Linear Regression, and Occupied vs. Empty in Phase 2 using Logistic Regression).

## Files in this Repository
* `ML402331_Phase1_Proposal.pdf` — Plain-text Phase 1 Project Proposal.
* `Phase1_Starter_Linear_Regression.ipynb` — Google Colab notebook running our first Linear Regression model.
* `starter_linear_regression.py` — Plain Python script version of the notebook.
* `room_occupancy_data.csv` — Starter dataset with 550 rows of sensor readings.

## Phase 1 Linear Regression Results
* **Formula Learned:** `h(x) = -4.3371 + 0.1549(Temperature) + 0.0204(Humidity) + 0.0032(Light) + 0.0052(CO2)`
* **Mean Squared Error (MSE):** `0.0586`
* **R² Score:** `0.964` (`96.4%` accuracy in explaining occupancy variation)

## AI Disclosure
AI was used to help document and organize the original source code and proposal text.
