# Project: Predicting Oil Prices with ARIMA, GARCH, LSTM, and Mamba Models

This project aims to predict oil prices using a combination of historical prices and sentiment analysis of news articles. The project employs several models and techniques to achieve this, including ARIMA, GARCH, LSTM, bidirectional LSTM, CNN-Attention model, and a Linear-Time Sequence Modeling with Selective State Spaces called Mamba.

## Methodology

1. **ARIMA and GARCH Models**: These models are used to predict oil prices based on historical data.

2. **LSTM with ARIMA and GARCH**: This approach combines the LSTM model with ARIMA and GARCH models to improve the prediction accuracy.

3. **Bidirectional LSTM and CNN-Attention Model**: This method uses a bidirectional LSTM and a CNN-Attention model to predict oil prices.

4. **Mamba Model**: This is a Linear-Time Sequence Modeling with Selective State Spaces. It is incorporated with news sentiment analysis to improve prediction metrics significantly. The Mamba model uses a selective state space algorithm that optimizes feature selection and reduces computational complexity. The algorithm works by selecting a subset of the state space that is most relevant to the prediction task, thereby reducing the dimensionality of the problem and improving computational efficiency.

## News Processing

1. **News Fetching**: News articles are fetched from oilprices.com.

2. **Sentiment Analysis and Vectorization**: The FinBERT transformer model is used for sentiment analysis and vectorization of the news articles to 512 dimensions.

## Extras

An XGBoost model is also included to check and reduce the dimension vs accuracy tradeoff.

## Dependencies

This project is implemented in Python and uses several libraries including NumPy, pandas, PyTorch, NLTK, Optuna, and more. Please ensure these are installed before running the project.

## Usage

To use this project, clone the repository, install the necessary dependencies, and run the Python scripts in the order they are presented in the project structure.

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