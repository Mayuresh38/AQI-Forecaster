# 🌍 Real-Time Air Quality (AQI) AI Forecaster

A production-ready, end-to-end Machine Learning web application that fetches real-time atmospheric telemetry across major Indian cities and forecasts the next day's Air Quality Index (AQI) using a highly optimized XGBoost pipeline.

🚀 **[Live Web Application Link]([(https://aqi-forecaster.streamlit.app/))**

---

## 📊 Project Overview

This project bridges the gap between pure data science training and robust MLOps engineering. Instead of using static historical datasets, this application integrates directly with the **OpenAQ API** to stream live ground-sensor readings from the Central Pollution Control Board (CPCB) network across 10 major Indian cities, processes the data on the fly, and delivers instant predictive insights via a Streamlit interface.

**Supported Cities:** Delhi · Mumbai · Chennai · Kolkata · Bangalore · Hyderabad · Pune · Ahmedabad · Jaipur · Lucknow

### Key Features
* **Live Telemetry Stream:** Parallelized data ingestion tracking critical pollutants ($PM_{2.5}$, $PM_{10}$, $NO_2$, $CO$, $SO_2$, $O_3$) from real CPCB ground sensors via OpenAQ v3.
* **Next-Day AQI Forecasting:** Inference engine outputting numerical AQI predictions and categorical health risk assessments (Good → Severe).
* **Official CPCB AQI Formula:** AQI is computed using the official Indian CPCB sub-index breakpoints for all pollutants — not estimated or approximated.
* **Production-Grade Resilience:** Built-in safeguards against API rate limits, sensor outages, and data gaps.

---

## 🤖 Model Performance

| Metric | Value |
| :--- | :--- |
| **Algorithm** | XGBoost Regressor |
| **$R^2$ (Test Set)** | 0.80 |
| **Tuning Engine** | Optuna Hyperparameter Optimization |
| **Training Period** | 2020 – 2023 |
| **Test Period** | 2024 (Strict chronological split) |
| **Target Variable** | Next-day AQI (`AQI_tomorrow`) |

*The model was trained on ~5 years of OpenAQ ground sensor data. A strict chronological train/test split was enforced to prevent temporal data leakage, and Optuna was used for hyperparameter tuning.*

---

## 🛠️ MLOps & Data Engineering Highlights

* **Eliminating Temporal Data Leakage:** Explicit temporal identifiers like `Year` were stripped from features. A chronological splitter forces the model to learn real atmospheric momentum (lag features, rolling means) rather than memorizing dates.
* **CPCB-Compliant AQI Recalculation:** Raw sensor readings are converted to AQI using official Indian CPCB breakpoints per pollutant. The final AQI is the maximum sub-index across all available pollutants — consistent with how official AQI is reported in India.
* **Lag Feature Engineering:** Features include AQI lag values at 1, 3, and 7 days, plus a 7-day rolling mean — capturing atmospheric momentum critical for next-day prediction.
* **Resilient API Throttling:** Micro-pacing throttles across threaded requests prevent HTTP 429 errors. Data windows are kept to lean 8-day rolling payloads to minimize API load.
* **Graceful Degradation:** If sensors flatline or the API fails, a flatline detector triggers an automatic fallback to a statistical simulator, keeping the UI fully operational at all times.

---

## 💻 Tech Stack

| Layer | Tools |
| :--- | :--- |
| **Frontend / UI** | Streamlit (Dark Mode Dashboard) |
| **ML Model** | XGBoost Regressor |
| **ML Pipeline** | Scikit-Learn (`Pipeline`, `ColumnTransformer`, `StandardScaler`, `OneHotEncoder`) |
| **Hyperparameter Tuning** | Optuna |
| **Data Manipulation** | Pandas, NumPy |
| **Async Execution** | Python `concurrent.futures` (`ThreadPoolExecutor`) |
| **Live Data Source** | OpenAQ v3 API (CPCB ground sensors) |
| **Security** | Python-Dotenv, Streamlit Secrets Management |

---

## 📁 Repository Structure

```text
├── .gitignore                                     # Excludes API keys and local checkpoints
├── requirements.txt                               # Pinned production dependencies
├── app.py                                         # Streamlit app and inference backend
├── aqi_pipeline_v2.pkl                            # Serialized champion XGBoost pipeline
├── Data_Collection_and_Feature_Engineering.ipynb  # Data pipeline and feature engineering
├── Final_model.ipynb                              # Model training, tuning, and evaluation
├── Model_Testing.ipynb                            # Test set evaluation and diagnostics
├── api.env                                        # Local API config (excluded via .gitignore)
└── README.md                                      # Project documentation
