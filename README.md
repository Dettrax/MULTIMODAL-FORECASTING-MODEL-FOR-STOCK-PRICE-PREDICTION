# Project: Advanced Time-Series Forecasting of Oil Prices: An Integration of ARIMA, GARCH, LSTM, Mamba Models, and Attention Mechanisms with Sentiment Analysis

This project aims to predict oil prices using a combination of historical prices and sentiment analysis of news articles. The project employs several models and techniques to achieve this, including ARIMA, GARCH, LSTM, bidirectional LSTM, CNN-Attention model, Scaled Dot-Product Attention, Multi-Head Attention, and a Linear-Time Sequence Modeling with Selective State Spaces called Mamba. Additionally, an XGBoost model is used to fine-tune and improve the performance of the LSTM, bidirectional LSTM, and CNN-Attention models.


## Key Mathematical and Algorithmic Components

- **ARIMA (AutoRegressive Integrated Moving Average)**: A statistical analysis model that uses time series data to either better understand the data set or to predict future trends. It uses autoregressions, differences, and moving averages to model the data.

- **GARCH (Generalized Autoregressive Conditional Heteroskedasticity)**: A statistical model used for time-series data that describes the variance of the current error term or innovation as a function of the actual sizes of the previous time periods' error terms.

- **LSTM (Long Short-Term Memory)**: A type of recurrent neural network (RNN) that can learn and remember over long sequences and is not sensitive to the length of the input sequence. It uses gates to control the memorizing process.

- **CNN-Attention Model**: This model combines Convolutional Neural Networks (CNNs) and attention mechanisms. CNNs are primarily used for image processing, video processing, and natural language processing. The attention mechanism allows the model to focus on specific parts of the input when producing the output.

- **Soft Attention, Scaled Dot-Product Attention, and Multi-Head Attention**: These are attention mechanisms used in the context of RNNs. They allow the model to focus on different parts of the input sequence when producing an output sequence, giving more weight to more important parts.

- **Mamba Model**: A Linear-Time Sequence Modeling with Selective State Spaces. It's a novel approach to sequence modeling that is designed to be efficient and effective, especially for large sequences.

- **Sentiment Analysis**: The use of natural language processing to identify, extract, quantify, and study affective states and subjective information. In this project, it's used to analyze news articles related to oil prices.

- **XGBoost**: A decision-tree-based ensemble Machine Learning algorithm that uses a gradient boosting framework. In this project, it's used to fine-tune and improve the performance of the LSTM, bidirectional LSTM, and CNN-Attention models.


## Methodology

1. **ARIMA and GARCH Models**: These models are used to predict oil prices based on historical data. The corresponding scripts are `Arima.py` and `Garch.py`.

2. **LSTM with ARIMA and GARCH**: This approach combines the LSTM model with ARIMA and GARCH models to improve the prediction accuracy. The script `call_lstm.py` is used for this purpose. You can choose between 1, 2, 3 for Single, Multi, and bidirectional models. The performance of these models is further enhanced using an XGBoost model for fine-tuning.

3. **CNN-Attention Model with ARIMA and GARCH**: This approach uses 1d Conv with attention algorithm ARIMA and GARCH models. The script `call_attention.py` is used for this purpose. The performance of these model is further enhanced using an XGBoost model for fine-tuning.

4. **Soft Attention, Scaled Dot-Product Attention and Multi-Head Attention**: These attention mechanisms are used to improve the performance of the LSTM and bidirectional LSTM models. The script `call_attention.py` is used for this purpose, which switches between Soft Attention, Scaled Dot-Product Attention, and Multi-Head Attention models based on the configuration.

5. **Mamba Model**: This is a Linear-Time Sequence Modeling with Selective State Spaces. It is incorporated with news sentiment analysis to improve prediction metrics significantly. The script `Mamba.py` is used for this purpose. It has three versions:
   - V1: Prediction with historic prices
   - V2: Prediction with historic prices and sentiment scores
   - V3: Prediction with historic prices, sentiment scores, and news vectors
   

## Data Fetching and Preprocessing

1. **Data Fetching**: Run `scrapper.py` with a change in page number. To make things faster, you can run this in multiple kernels with parts like kernel 1: 1-100 page, kernel 2: 101-200, and so on.

2. **Data Preprocessing**: The scripts `Data_processing.py`, `final_process.py`, and `finbert_vec.py` are used for data preprocessing.

## Running the Project

To run this project, follow these steps:

1. Clone the repository and navigate to the project directory.

2. Install the necessary dependencies. You can do this by running `pip install -r requirements.txt` in your terminal.

3. Run the data fetching script with the command `python scrapper.py`.

4. Run the data preprocessing scripts in the following order:
   - `python Data_processing.py`
   - `python final_process.py`
   - `python finbert_vec.py`

5. Run the model scripts in the following order:
   - `python Arima.py`
   - `python Garch.py`
   - `python call_lstm.py` (Choose between 1, 2, 3 for Seq, bidirectional, and CNN-Attention models.)
   - `python call_attention.py` (This script switches between Soft Attention, Scaled Dot-Product Attention, and Multi-Head Attention models based on the configuration.)
   - `python Mamba.py` (Choose between V1, V2, V3 for different versions.)

Please note that the `call_attention.py` script switches between Soft Attention, Scaled Dot-Product Attention, and Multi-Head Attention models based on the configuration. You do not need to run separate scripts for each attention mechanism.

## Contributing

Contributions to this project are welcome. Please fork the repository and create a pull request with your changes.

## License

This project is open source and available under the [MIT License](LICENSE).

## Dependencies

This project is implemented in Python and uses several libraries. The complete list of dependencies can be found in the `requirements.txt` file. Some of the key libraries used include:

- numpy
- pandas
- torch
- nltk
- optuna
- sklearn
- matplotlib
- xgboost
- transformers
- newspaper3k
- selenium

Please ensure these are installed before running the project.

## Requirements

To run this project, you will need the following Python packages:

- numpy
- pandas
- torch
- nltk
- optuna
- sklearn
- matplotlib

You can install these packages using pip:

```bash
pip install numpy pandas torch nltk optuna sklearn matplotlib