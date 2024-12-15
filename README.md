# RSI Divergence Detector

The **RSI Divergence Detector** is a Python-based tool designed to detect **bullish** and **bearish RSI divergences** in cryptocurrency price data. By leveraging machine learning models such as LSTMs and Transformers, the tool aims to identify divergence patterns, predict market movements, and evaluate short-term trading profitability.

---

## **Features**

- **Divergence Detection**:
  - Identifies RSI-based bullish and bearish divergences in multiple timeframes.
- **Profitability Estimation**:
  - Calculates profitability of trades within divergence regions (e.g., 5–50 candles).
- **Flexible Data Handling**:
  - Supports various data resolutions (`1m`, `5m`, `15m`, `1h`, etc.) for customizable analysis.
- **Machine Learning Models**:
  - Utilizes LSTM and Transformer models for divergence prediction.
- **Visualization**:
  - Includes Jupyter notebooks for in-depth data exploration and visualization.

---

## **Project Structure**

```plaintext
rsi_divergence_detector/
├── README.md                      # Project documentation
├── data/                          # Market data storage
│   ├── processed_data/            # Preprocessed data for model training
│   ├── raw_data/                  # Raw cryptocurrency data
│   └── training_data/             # Pickled datasets for training and validation
├── etc/                           # Miscellaneous configurations or logs
├── model_data/                    # Pre-trained models and training outputs
│   └── mixed_lstm/                # LSTM-based pre-trained models
├── notebooks/                     # Jupyter notebooks for analysis and visualization
├── poetry.lock                    # Poetry dependency lock file
├── pyproject.toml                 # Poetry project configuration
└── src/                           # Core project source code
    ├── data_process/              # Scripts for data fetching, preprocessing, and labeling
    ├── dataset/                   # Dataset classes for LSTM models
    ├── model/                     # LSTM and Transformer model definitions
    └── training/                  # Training scripts for LSTM and Transformer models
```

---

## **Installation**
1. Clone the repository
```bash
git clone https://github.com/Shawn-Kim96/rsi_divergence_detector.git
cd rsi_divergence_detector
```

2. Set up Python Environment
This project uses [Poetry](https://python-poetry.org/) for dependency management.

```bash
pip install poetry
poetry install
poetry shell
```

## **Usage**

### Run Jupyter Notebook (Recommended)
- Execute `notebooks/4.0-train_model.ipynb` will bring and train the model
- You can choose the parameter inside the notebooks.


### Run Main Application 
The main entry point for the application is located in src/main.py. This script trains the model using prepared data
**Make sure** you have your data ready
- `data/processed_data/training_data.pickle`
- `data/processed_data/divergence_data.pickle`

It is much faster if you have dataset in `data/training_data`
- `data/training_data/train_dataset.pickle`
- `data/training_data/valid_dataset.pickle`
- `data/training_data/test_dataset.pickle`


```bash
python src/main.py
```

This code will start training the model.
- TODO: should update main script for various parameter tuning.


### Notebook Analysis
Explore the notebooks/ directory for step-by-step data analysis, divergence testing, and visualization. The following notebooks are included:

- Data Analysis: 1.0-data_analysis.ipynb
- Divergence Testing: 2.0-divergence_detector_test.ipynb
- Data Visualization: 3.0-data_visualization.ipynb
- Training Data Generation: 4.0-generate_training_data.ipynb

## Data Requirements
Executing `src/data_process/main.py` will generate data needed. However, it takes long time to generate data, and you have to change the project path written in `main.py`
The code should be executed like

```bash
poetry shell
python src/data_process/main.py
```

This code acts like
1. Fetch BTC data from external APIs and generate time series data of BTC in data/raw_data by timeframe
2. It will process the data and generate `divergence_data.pickle` and  `training_data.pickle`
    - `divergence_data.pickle` is a dictionary that have divergence occurance data for every timeframe
    - `training_data.pickle` is a pandas dataframe that have all timeframes price data


## Models
1. LSTM Mixed
- Path: model_data/mixed_lstm/
- Designed for sequential analysis of RSI patterns.
- Training script: src/training/train_lstm.py

2. Transformer
- Path: src/model/transformer.py
- Utilized for multi-head attention-based divergence detection.
- Training script: src/training/train_transformer.py