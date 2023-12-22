import pandas as pd
from pandas.errors import SettingWithCopyWarning
from sklearn.ensemble import GradientBoostingRegressor  # Import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
from main_funcs.tensor import combined_data
from main_funcs.num_seasons import playerw10seasons
import numpy as np
import warnings
import matplotlib.pyplot as plt

# Filter out the specific warnings
warnings.filterwarnings("ignore", category=SettingWithCopyWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Create an empty DataFrame to store prediction_evaluations
all_predictions = pd.DataFrame()

# Create empty lists to store evaluation metrics
mse_list, mae_list, correlation_list = [], [], []

# Iterate over the list of player names
for player_name in playerw10seasons:
    # Filter the data to get the current player's attributes
    player_data = combined_data[combined_data['Player'] == player_name]

    # Ensure G+A/90 is a numeric column (remove any non-numeric entries)
    player_data['G+A/90'] = pd.to_numeric(player_data['G+A/90'], errors='coerce')

    # Define the number of previous seasons to consider
    num_previous_seasons = 3

    # Create columns for G+A/90 from the previous three seasons
    for i in range(1, num_previous_seasons + 1):
        player_data[f'G+A/90_{i}'] = player_data['G+A/90'].shift(i)

    # Create an empty list to store the prediction_evaluations for the current player
    predictions = []

    # Define the starting and ending seasons
    start_season = '2016-17'
    end_season = '2022-23'

    # Initialize the model for the current player
    model = GradientBoostingRegressor(n_estimators=100, random_state=42)  # Use GradientBoostingRegressor

    # Iterate through seasons, training and predicting
    for season in player_data['Season'].unique():
        if season >= start_season and season <= end_season:
            # Filter data for the current season
            current_data = player_data[player_data['Season'] == season]

            # Extract features for the current season
            features = ['G+A/90_' + str(i) for i in range(1, num_previous_seasons + 1)] + ['Age', '90s']

            # Extract actual G+A/90 for the current season
            actual_ga90 = current_data['G+A/90'].values[0]

            if season > start_season:
                # Use the trained model to predict G+A/90 for the current season
                prediction = model.predict(current_data[features].values.reshape(1, -1))[0]

                # Clip the prediction to be non-negative
                prediction = np.clip(prediction, 0, None)

                # Append the results to the prediction_evaluations list
                predictions.append({'Player': player_name, 'Season': season, 'Actual': actual_ga90, 'Predicted': prediction})

            # Train the model using data up to the current season
            training_data = player_data[player_data['Season'].between(start_season, season)]
            X_train = training_data[features]
            y_train = training_data['G+A/90']
            model.fit(X_train, y_train)

    # Create a DataFrame from the prediction_evaluations list for the current player
    player_predictions_df = pd.DataFrame(predictions)

    # Concatenate the prediction_evaluations for the current player to the overall prediction_evaluations DataFrame
    all_predictions = pd.concat([all_predictions, player_predictions_df])

    # Evaluate performance for the current player
    mse = mean_squared_error(player_predictions_df['Actual'], player_predictions_df['Predicted'])
    mae = mean_absolute_error(player_predictions_df['Actual'], player_predictions_df['Predicted'])
    correlation = player_predictions_df['Actual'].corr(player_predictions_df['Predicted'])

    # Append evaluation metrics to lists
    mse_list.append(mse)
    mae_list.append(mae)
    correlation_list.append(correlation)

    # Print evaluation metrics for the current player
    print(f"\nEvaluation metrics for {player_name}:")
    print(f"MSE: {mse}")
    print(f"MAE: {mae}")
    print(f"Correlation: {correlation}")

    # Plot actual vs predicted values for the last player
    if player_name == playerw10seasons[-1]:
        plt.figure(figsize=(10, 6))
        plt.scatter(player_predictions_df['Season'], player_predictions_df['Actual'], label='Actual', color='blue')
        plt.plot(player_predictions_df['Season'], player_predictions_df['Predicted'], label='Predicted', color='red')
        plt.title(f"Actual vs Predicted G+A/90 for {player_name} (Gradient Boosting)")
        plt.xlabel("Season")
        plt.ylabel("G+A/90")
        plt.ylim(0,1) # x axis range from 0 to 1
        plt.legend()
        plt.show()

# Display the prediction_evaluations DataFrame with Actual and Predicted values for all players
print(all_predictions)

# Print average evaluation metrics
average_mse = np.mean(mse_list)
average_mae = np.mean(mae_list)
average_correlation = np.mean(correlation_list)

print(f"\nAverage MSE: {average_mse}")
print(f"Average MAE: {average_mae}")
print(f"Average Correlation: {average_correlation}")

warnings.resetwarnings()
