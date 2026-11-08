## overview
this project uses four simple room sensors (`Temperature`, `Humidity`, `Light`, and `CO2`) to predict room occupancy (`People_Count` in Phase 1 using Linear Regression, and Occupied vs. Empty in Phase 2 using Logistic Regression).

## files in this repository
* `ML402331_Phase1_Proposal.pdf` — 2-page Phase 1 Project Proposal.
* `Phase1_Starter_Linear_Regression.ipynb` — Google Colab notebook running our first Linear Regression model.
* `starter_linear_regression.py` — Plain Python script version of the notebook.
* `room_occupancy_data.csv` — Starter dataset with 550 rows of sensor readings.

## Phase 1 linear regression results
* **formula learned:** $h(x) = -4.3371 + 0.1549(\text{Temperature}) + 0.0204(\text{Humidity}) + 0.0032(\text{Light}) + 0.0052(\text{CO2})$
* **mean squared error (MSE):** `0.0586`
* **$R^2$ Score:** `0.964` (`96.4%` accuracy in explaining occupancy variation)
