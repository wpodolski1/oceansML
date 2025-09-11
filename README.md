# oceansML

![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)  
![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)  
![Status](https://img.shields.io/badge/status-active-success.svg)  
![Data](https://img.shields.io/badge/datasets-NOAA%20%7C%20UNEP--WCMC-blueviolet)  

**oceansML** is a Python project for exploring, visualizing, and modelling ocean datasets. It provides tools for preprocessing, model evaluation, and visualization of oceanographic data, focusing on coral reefs and coral bleaching events.

---

## Table of Contents

1. [Project Overview](#project-overview)  
2. [Motivation & Problem Statement](#motivation--problem-statement)  
3. [Data Sources](#data-sources)  
4. [Installation](#installation)  
5. [Code Structure](#code-structure)  
6. [Usage](#usage)  
7. [Results & Evaluation](#results--evaluation)  
8. [Roadmap](#roadmap)  
9. [Future Work](#future-work)  
10. [Contributing](#contributing)  
11. [Acknowledgments](#acknowledgments)  
12. [License](#license)

---

## Project Overview

oceansML is designed to support the analysis of coral reef datasets by providing:

- Loading and preprocessing sea surface temperature (SST) and coral reef location data  
- Visualizing coral bleaching hotspots following NOAA guidelines  
- Building and evaluating machine learning models (e.g., LSTM) for forecasting reef stress  

The long-term goal is to estimate yearly reef damage and predict bleaching risk under warming trends.

---

## Motivation & Problem Statement

Coral reefs are among the most fragile ecosystems on Earth, yet they are poorly described in global datasets. Monitoring bleaching events is challenging due to:

- Lack of high-quality datasets  
- Limited temporal and spatial coverage  
- Quality issues in existing observations  

This project addresses these issues by:

- Combining SST and reef polygon data to identify bleaching hotspots  
- Estimating the percentage of reefs at risk under NOAA bleaching thresholds  
- Building predictive models to forecast reef bleaching given ongoing climate trends  

---

## Data Sources

### Sea Surface Temperature (SST)  
- **NOAA PSL Climate Data Repository**  
  Used for both climatology baseline and model training.  

### Coral Reef Locations  
- **UNEP-WCMC (2025)** — Ocean+ Habitats [Online]. *Coral Reefs (2018 dataset)*  
  Available at: [habitats.oceanplus.org](https://habitats.oceanplus.org)  

### Datasets & Preprocessing

**Climatology baseline (1985–1990 & 1993):**
- `sst.day.mean.1985.nc`  
- `sst.day.mean.1986.nc`  
- `sst.day.mean.1987.nc`  
- `sst.day.mean.1988.nc`  
- `sst.day.mean.1989.nc`  
- `sst.day.mean.1990.nc`  
- `sst.day.mean.1993.nc`  

Steps:  
1. Calculate monthly means.  
2. Extract maximum monthly mean (MMM) across baseline years.  

**Model training (2016 SST data):**  
- `sst.day.mean.2016.nc`  
- `WCMC008_CoralReefs2018_v4_1.zip`  

Steps:  
1. Apply coral reef mask from polygons onto SST dataset.  
2. Extract SST values only on reef locations → (points × time).  
3. Convert to Pandas DataFrame.  
4. Add day-of-year feature for seasonality.  
5. Transform into sequences for LSTM input.  
6. Normalize data using `StandardScaler()`.  
7. Train/test split (80/20, no shuffle).  

**Visualization of bleaching impact:**  
- `sst.day.mean.2022.nc`  
- `sst.day.mean.2023.nc`  

---

## Installation

```bash
# Clone the repo
git clone https://github.com/Aspanu1/oceansML.git
cd oceansML

# (Optional) create and activate virtual environment
python3 -m venv .env
source .env/bin/activate    # on Windows: .env\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

---

## Results & Evaluation

- **Baseline climatology**: MMM derived from 1985–1990 & 1993 SST data.  
- **Hotspot visualization**: bleaching thresholds applied to 2022–2023 SST data showed global reef stress patterns.  
- **LSTM performance**: achieved *[add metrics here]* on SST time series forecasting for reef locations.  

> Add plots (time series predictions, hotspot maps) and tables with metrics here.

[Model eval][https://i.imgur.com/E6BNK9x.png]

RMSE:0.557272°C
MAE:0.38°C
R2:0.90
Bias:-0.05°C

---

## Roadmap

- [ ] **Fix climatology baseline calculation**  
  Implement linear regression models for each unique lat × lon point to account for warming trend.  
  *(Current baseline underestimates bleaching events.)*
  
- [ ] **Add error handling and QOL features (config file etc.)**  
  Build an interactive dashboard to visualize bleaching events.  
  
- [ ] **Predict future reef stress**  
  Extend pipeline to generate forecasts for future years and assess how rising sea surface temperature affects coral reefs and their societal impact.  

- [ ] **Bleaching event pipeline**  
  Deliver an automated end-to-end pipeline to predict possible bleaching events in the nearest future using NOAA daily SST data and neural networks.  

- [ ] **Model improvements**  
  Experiment with alternative architectures.  

- [ ] **Visualization**  
  Build an interactive dashboard to visualize bleaching events.  
  

---

## Acknowledgments

- **NOAA PSL** — for SST datasets  
- **UNEP-WCMC Ocean+ Habitats** — for coral reef polygons  
- Python scientific ecosystem: `numpy`, `pandas`, `matplotlib`, `scikit-learn`, `tensorflow/keras`  

---

## License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.  
Note: Dataset licenses (NOAA, UNEP-WCMC) must be respected separately.
