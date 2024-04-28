# Project: Predicting Oil Prices with ARIMA, GARCH, LSTM, and Mamba Models

This project aims to predict oil prices using a combination of historical prices and sentiment analysis of news articles. The project employs several models and techniques to achieve this, including ARIMA, GARCH, LSTM, bidirectional LSTM, CNN-Attention model, and a Linear-Time Sequence Modeling with Selective State Spaces called Mamba. Additionally, an XGBoost model is used to fine-tune and improve the performance of the LSTM, bidirectional LSTM, and CNN-Attention models.

## Methodology

1. **ARIMA and GARCH Models**: These models are used to predict oil prices based on historical data. The corresponding scripts are `Arima.py` and `Garch.py`.

2. **LSTM with ARIMA and GARCH**: This approach combines the LSTM model with ARIMA and GARCH models to improve the prediction accuracy. The script `call_lstm` is used for this purpose. You can choose between 1, 2, 3 for Seq, bidirectional, and CNN-Attention models. The performance of these models is further enhanced using an XGBoost model for fine-tuning.

3. **Mamba Model**: This is a Linear-Time Sequence Modeling with Selective State Spaces. It is incorporated with news sentiment analysis to improve prediction metrics significantly. The script `Mamba` is used for this purpose. It has three versions:
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
   - `python Mamba.py` (Choose between V1, V2, V3 for different versions.)

## Dependencies

This project is implemented in Python and uses several libraries including NumPy, pandas, PyTorch, NLTK, Optuna, and more. Please ensure these are installed before running the project.

## Contributing

Contributions to this project are welcome. Please fork the repository and create a pull request with your changes.

## License

This project is open source and available under the [MIT License](LICENSE).

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
```

You will also need to have the FinBERT transformer model available for the sentiment analysis. You can find more information about this model and how to use it [here](https://github.com/ProsusAI/finBERT).

Finally, you will need access to the oilprices.com website to fetch the news articles, and the `brent_with_forecasted_volatility_prime.csv` and `brent_vec.xlsx` data files.