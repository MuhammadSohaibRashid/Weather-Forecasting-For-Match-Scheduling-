# Weather Forecasting for Match Scheduling

This project uses historical weather data from three Pakistani cities to estimate future weather conditions and support the decision of whether outdoor sports activities, such as a cricket match, should be scheduled.

The work is implemented as a machine-learning study in the Jupyter notebook [`Weather Prediction Using Machine Learning.ipynb`](Weather%20Prediction%20Using%20Machine%20Learning.ipynb). It covers the complete data-science workflow: collecting and exploring weather data, preparing features, selecting useful variables, training prediction models, comparing results, and interpreting the output for match planning.

## Project objective

Weather can directly affect the safety, quality, and feasibility of an outdoor match. The project investigates whether measurable weather variables can be used to:

- Predict temperature for a selected city and date.
- Classify the likely weather condition, such as clear sky, clouds, rain, mist, or thunderstorm.
- Identify the weather variables that contribute most to the prediction.
- Provide evidence that can support a match-scheduling decision.

The model output is intended as a decision-support aid, not as a replacement for an official weather service or professional safety assessment.

## Dataset

The repository includes `KLM Weather Prediction Dataset.csv`, containing approximately five years of daily observations for:

- Karachi
- Lahore
- Multan

The main columns are:

| Column | Description |
| --- | --- |
| `city` | City where the observation was recorded |
| `date` | Observation date |
| `temperature` | Recorded temperature and the main regression target |
| `humidity` | Relative humidity measurement |
| `pressure` | Atmospheric pressure |
| `weather` | Categorical weather description |
| `windspeed` | Wind-speed measurement |
| `visibility` | Visibility measurement |

The notebook also contains an OpenWeather API data-collection example for retrieving city weather observations. API keys should be kept outside the notebook and supplied through environment variables when this workflow is used.

## Machine-learning workflow

### 1. Data exploration and cleaning

The notebook examines the structure of the dataset, summary statistics, unique values, missing values, distributions, and possible outliers. It uses histograms, box plots, scatter plots, correlation matrices, and covariance heatmaps to understand the relationships among temperature, humidity, pressure, windspeed, and visibility.

### 2. Feature preparation

- The date field is removed from the modelling matrix in the training examples.
- Missing rows are removed before training.
- Categorical values such as `city` and `weather` are converted into numerical representations using one-hot encoding or label encoding, depending on the experiment.
- The dataset is divided into training and testing subsets with an 80/20 split and a fixed random state for repeatability.

### 3. Feature selection

Several approaches are explored to determine which variables are most useful:

- Mutual information and chi-squared scoring.
- `SelectKBest` and recursive feature elimination.
- Forward and backward sequential feature selection.
- Exhaustive feature selection.
- Random Forest feature importance.

The notebook identifies humidity, pressure, windspeed, visibility, and city as important predictors for the temperature experiments.

### 4. Temperature prediction

Temperature is treated as a continuous regression target. The notebook compares:

- Linear Regression
- Random Forest Regression
- AdaBoost Regression
- Gradient Boosting Regression
- Stacking Regression

The models are evaluated using Mean Squared Error (MSE) and the R² score. These metrics show both the size of prediction errors and how much of the target variation is explained by the model.

### 5. Weather-condition classification

The weather description is also treated as a classification target. A Random Forest Classifier is trained after encoding the weather labels. The notebook evaluates the classifier with:

- Confusion matrix
- Classification report
- Precision, recall, and F1-score

This classification output can be used as one input to a scheduling rule. For example, predicted rain or severe weather may indicate that a match needs to be postponed or reviewed, while clear conditions may support proceeding with the planned schedule.

## Match-scheduling use case

The practical idea is to combine the predicted weather condition and numerical forecasts with match requirements. A scheduling system could review:

1. The selected city and match date.
2. Predicted temperature and weather condition.
3. Rain, wind, visibility, and humidity risk indicators.
4. The sport's acceptable operating conditions.
5. A final recommendation such as **schedule**, **review**, or **postpone**.

The current repository provides the predictive analysis and the foundation for this decision. A production scheduling application would still need a clearly defined threshold policy, future-date validation, live forecast integration, and domain-specific safety rules.

## Technologies and skills demonstrated

- Python and Jupyter Notebook
- Pandas and NumPy for data preparation
- Matplotlib and Seaborn for exploratory visualization
- Scikit-learn for preprocessing, feature selection, regression, classification, and evaluation
- Mlxtend for sequential and exhaustive feature selection
- Exploratory data analysis and statistical interpretation
- Model comparison using MSE and R²
- Classification evaluation using confusion matrices and precision/recall metrics
- API-based weather data collection concepts
- Translating machine-learning results into a real-world scheduling use case

## Repository structure

```text
.
├── KLM Weather Prediction Dataset.csv
├── README.md
└── Weather Prediction Using Machine Learning.ipynb
```

## Running the notebook

1. Clone the repository.
2. Install the notebook dependencies:

   ```bash
   pip install pandas numpy matplotlib seaborn scikit-learn mlxtend jupyter
   ```

3. Start Jupyter:

   ```bash
   jupyter notebook
   ```

4. Open `Weather Prediction Using Machine Learning.ipynb` and run the cells in order.

Some exploratory cells reference city-specific CSV paths or Google Colab paths that were used during development. Update those paths to the included dataset or split the dataset by city before running those cells locally.

## Limitations and future improvements

- The notebook is an experimental analysis rather than a deployed forecasting API.
- Future-date forecasting requires a time-aware validation strategy and a reliable source of future weather features.
- The match-scheduling policy should be formalized with sport-specific thresholds.
- Hyperparameter tuning and cross-validation can improve model reliability.
- A future version could expose the predictions through a web dashboard or REST API.
- Weather data and model performance should be monitored as new observations become available.

## Author

**Muhammad Sohaib Rashid**

Machine-learning project for weather-aware sports match scheduling in Pakistani cities.