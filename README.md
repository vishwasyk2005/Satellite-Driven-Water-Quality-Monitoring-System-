# 🌍 Satellite-Driven Water Quality Monitoring & Forecasting

This project is an **interactive web-based dashboard** for **satellite-driven water quality monitoring and forecasting**. It uses **remote sensing–derived parameters** and **time-series forecasting** to analyze and predict the health of water bodies.

The application is built using **Streamlit** and leverages **Facebook Prophet** for forecasting, **Plotly** for visualization, and **Folium** for spatial heatmaps.

---

## 📌 Project Overview

* **Domain**: Environmental Monitoring / Remote Sensing
* **Focus**: Water quality analysis & forecasting
* **Input**: Satellite-derived CSV data (e.g., from Google Earth Engine / Roboflow-style exports)
* **Output**:

  * Time-series forecasts
  * Model evaluation metrics
  * Interactive spatial heatmaps

---

## 🚀 Key Features

* 📂 CSV upload with automated preprocessing
* 🧹 Data cleaning, interpolation & outlier capping
* 📈 Time-series forecasting using Prophet
* 📊 Model evaluation (R², RMSE, MAE)
* 🗓 Adjustable forecast horizon (30–730 days)
* 🗺 Interactive geospatial heatmaps
* 📥 Downloadable processed dataset
* 🌐 Fully interactive Streamlit dashboard

---

## 🛠️ Tech Stack

* **Python**
* **Streamlit** – Web dashboard
* **Pandas / NumPy** – Data processing
* **Prophet** – Time-series forecasting
* **Plotly** – Interactive charts
* **Folium + Branca** – Geospatial heatmaps
* **scikit-learn** – Model evaluation metrics

---

## 📂 Project Structure

```
├── app.py                     # Main Streamlit application
├── requirements.txt           # Python dependencies
├── README.md                  # Project documentation
└── sample_data.csv            # Example input dataset
```

---

## 📊 Water Quality Parameters

The system currently supports forecasting of:

* **Turbidity** – Indicates water clarity
* **NDWI (Normalized Difference Water Index)** – Reflects surface water presence
* **Chlorophyll** – Proxy for algal concentration

---

## 📥 Input Data Format

The uploaded CSV file must contain the following columns:

| Column Name    | Description                                             |
| -------------- | ------------------------------------------------------- |
| `system:index` | Date encoded as `YYYY_MM_DD_*`                          |
| `.geo`         | GeoJSON string with coordinates `[longitude, latitude]` |
| `Turbidity`    | Numeric value                                           |
| `NDWI`         | Numeric value                                           |
| `Chlorophyll`  | Numeric value                                           |

---

## ▶️ How to Run the Application

### 1️⃣ Clone the Repository

```bash
git clone https://github.com/your-username/satellite-water-quality-monitoring.git
cd satellite-water-quality-monitoring
```

### 2️⃣ Create a Virtual Environment (Recommended)

```bash
python -m venv venv
source venv/bin/activate      # Linux / macOS
venv\Scripts\activate         # Windows
```

### 3️⃣ Install Dependencies

```bash
pip install -r requirements.txt
```

### 4️⃣ Run the Streamlit App

```bash
streamlit run app.py
```

The dashboard will open automatically in your browser.

---

## 📈 Forecasting Methodology

* Uses **Facebook Prophet**, suitable for environmental time-series data
* Handles missing values and seasonality
* Multiplicative seasonality mode
* Forecast horizon selectable via UI
* Confidence intervals included for uncertainty estimation

---

## 🗺 Spatial Visualization

* Latest available satellite measurements are plotted on an interactive map
* Color-coded heatmaps for each parameter
* Clickable markers with value and date information

---
