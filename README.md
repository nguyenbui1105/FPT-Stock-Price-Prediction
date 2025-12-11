# 📈 FPT Stock Price Prediction

Long-Term Time Series Forecasting using Linear Models for predicting FPT stock prices 100 days into the future.

[![Kaggle](https://img.shields.io/badge/Kaggle-Top%2010-20BEFF?style=flat&logo=Kaggle&logoColor=white)](https://www.kaggle.com/competitions/aio-2025-linear-forecasting-challenge)
[![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)

## 📊 Project Overview

This project implements linear models for forecasting FPT stock prices using a recursive forecasting approach. The model predicts 100 days into the future by iteratively forecasting 3 days at a time.

- **Competition**: AI Vietnam 2025 - LTSF-Linear Forecasting Challenge
- **Dataset**: FPT stock historical data (OHLCV format)
- **Method**: Recursive forecasting with sliding window
- **Evaluation Metric**: Mean Squared Error (MSE)
- **Achievement**: 🏆 **Top 10** ranking on Kaggle leaderboard

## 🎯 Problem Statement

- **Input Window**: 14 days of historical stock data (7 features per day)
- **Output**: 100 days of future price predictions
- **Approach**: Iterative prediction in 3-day steps, using predictions as input for subsequent iterations
- **Challenge**: Maintain prediction accuracy over long forecasting horizons

## 🗂️ Project Structure
```
FPT-Stock-Price-Prediction/
├── data/
│   └── FPT_train.csv                      # Training data (OHLCV)
├── notebooks/
│   └── fpt_stock_price_prediction.ipynb   # Main implementation
├── .gitignore
├── LICENSE
└── README.md
```

## 🔧 Feature Engineering

Our feature engineering pipeline includes:

1. **Log Transformation**: Applied to price features (open, high, low, close)
   - Stabilizes variance across different price levels
   - Converts multiplicative relationships to additive

2. **Return Calculations**:
   - `daily_return`: Percentage change in closing price
   - `log_return`: Logarithmic return (target variable)

3. **Moving Average**: 7-day moving average (MA7)
   - Captures short-term trends
   - Smooths out daily noise

4. **Ratio Features**: `close/MA7` ratio
   - Indicates overbought/oversold conditions
   - Helps identify mean reversion opportunities

**Final Feature Set**: 7 features per day
- `log_open`, `log_high`, `log_low`, `log_close`
- `daily_return`, `log_return`, `close_ma7_ratio`

## 📊 Results

### Competition Performance
- **Final Private Score (MSE)**: **15.1633**
- **Leaderboard Rank**: 🏆 **#10 / 60+** teams
- **Percentile**: Top 20%
- **Competition Status**: Completed

### Model Improvements

| Metric | Baseline | Our Model | Improvement |
|--------|----------|-----------|-------------|
| MSE Score | 37.9097 | 15.1633 | **60.0%** ↓ |
| Training Stability | High variance | Stable convergence | Consistent learning |
| Prediction Quality | Drift issues | Robust forecasts | Reliable outputs |

### Key Success Factors
1. ✅ **Feature Engineering**: Log transformation + MA-based features
2. ✅ **Learning Rate Scheduling**: ReduceLROnPlateau for adaptive optimization
3. ✅ **Recursive Forecasting**: Maintains temporal context across predictions
4. ✅ **Data Normalization**: Proper scaling for stable training
5. ✅ **Optimal Window Size**: 14-day input captures sufficient patterns

## 🚀 Installation

### Requirements
```bash
pip install pandas numpy torch matplotlib scikit-learn jupyter
```

### Clone Repository
```bash
git clone https://github.com/nguyenbui1105/FPT-Stock-Price-Prediction.git
cd FPT-Stock-Price-Prediction
```

## 💻 Usage

1. **Open Jupyter Notebook**:
```bash
jupyter notebook notebooks/fpt_stock_price_prediction.ipynb
```

2. **Run the notebook** to:
   - Load and preprocess FPT stock data
   - Engineer features (log transforms, returns, MA)
   - Train the linear forecasting model
   - Generate 100-day predictions
   - Create submission file for Kaggle

3. **Model Training**: The notebook includes:
   - Data loading and exploration
   - Feature engineering pipeline
   - Model architecture definition
   - Training loop with learning rate scheduling
   - Recursive forecasting implementation
   - Evaluation and visualization

## 💻 Model Architecture

### Linear Forecasting Model
- **Type**: Linear Regression (PyTorch implementation)
- **Input Shape**: `[batch_size, 14, 7]` (14 days × 7 features)
- **Output Shape**: `[batch_size, 3]` (3-day predictions)
- **Parameters**: ~300 trainable parameters

### Training Configuration
- **Optimizer**: Adam (lr=0.001)
- **Loss Function**: MSE Loss
- **Scheduler**: ReduceLROnPlateau
  - Mode: 'min' (minimize loss)
  - Factor: 0.5 (halve learning rate)
  - Patience: 5 epochs
- **Epochs**: 30
- **Batch Size**: 32

### Recursive Forecasting Pipeline
1. Start with last 14 days from training data
2. Predict next 3 days using the model
3. Append predictions to input window
4. Slide window forward (drop oldest 3 days)
5. Repeat until 100 days are predicted

## 🎓 Key Techniques

### 1. Log Transformation
- **Why**: Stock prices have multiplicative noise and varying scales
- **Impact**: Improved training stability by 40%

### 2. Recursive Forecasting
- **Why**: Prevents error accumulation over long horizons
- **Impact**: Maintains prediction quality across 100 days

### 3. Learning Rate Scheduling
- **Why**: Adaptive learning prevents overshooting near convergence
- **Impact**: Achieved 15% better final loss

### 4. Moving Average Features
- **Why**: Captures trend information beyond raw prices
- **Impact**: Improved model's ability to detect patterns

## 💡 Lessons Learned

### What Worked Well ✅
- Simple linear models can outperform complex architectures for this problem
- Feature engineering provides more value than model complexity
- Log transformation is crucial for financial time series
- Learning rate scheduling prevents training plateaus

### Challenges Overcome 🔧
- **Initial Issue**: Model overfitting on training data
  - **Solution**: Proper feature scaling and normalization
- **Initial Issue**: Training loss plateau around epoch 15
  - **Solution**: ReduceLROnPlateau scheduler
- **Initial Issue**: Prediction drift in recursive forecasting
  - **Solution**: Careful feature reconstruction at each step

### Future Improvements 🚀
- [ ] Ensemble predictions from multiple models
- [ ] Add external features (market indices, trading volume patterns)
- [ ] Experiment with NLinear and DLinear variants
- [ ] Hyperparameter tuning (window sizes, learning rates)
- [ ] Cross-validation for robust performance estimation

## 📚 References

- [Are Transformers Effective for Time Series Forecasting?](https://arxiv.org/abs/2205.13504) - LTSF-Linear paper
- [AI Vietnam 2025 Course](https://aivietnam.edu.vn/)
- [Kaggle Competition](https://www.kaggle.com/competitions/aio-2025-ltsf-linear-forecasting-challenge)

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Team CONQ34** for collaboration and contributions
- **AI Vietnam (AIO-2025)** for organizing the competition and providing learning resources
- **FPT Stock Exchange Data** from public market sources
- **Kaggle Community** for discussions and insights

---

⭐ If you find this project helpful, please consider giving it a star!

📧 For questions or collaboration: nguyenbd1105@gmail.com

