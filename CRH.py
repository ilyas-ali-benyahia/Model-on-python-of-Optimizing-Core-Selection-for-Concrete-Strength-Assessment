import pandas as pd
import numpy as np
from random import sample
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Data
data = {
    'F': [
        5.3, 8.8, 7.3, 10.4, 8.0, 10.4, 8.5, 13.4, 9.4, 8.3, 12.6, 13.3, 7.7, 11.3, 
        8.1, 11.6, 7.6, 8.4, 7.5, 10.2, 14.1, 9.7, 12.9, 12.2, 10.0, 14.1, 10.9, 
        17.2, 13.7, 15.2, 14.5, 11.4, 12.9, 14.5, 9.2, 16.0, 14.5, 10.4, 13.1, 
        10.9, 16.4, 14.2, 14.7, 11.6, 16.0, 16.2, 18.3, 20.8, 20.3, 15.3, 15.9, 
        13.6, 13.1, 14.5, 13.3, 10.9, 15.7, 17.8, 15.6, 14.3, 13.2, 16.9, 15.0, 
        9.9, 13.4, 20.5, 14.6, 18.2, 15.0, 15.5, 12.2, 16.0, 17.1, 14.9, 14.5, 
        19.5, 15.0, 13.0, 22.1, 17.7, 14.3, 13.3, 18.4, 14.2, 15.1, 26.1, 17.1, 
        17.5, 15.3, 22.1, 19.0, 21.2, 18.2, 12.4, 15.2, 16.5, 20.6, 15.2, 11.8, 
        17.8, 21.8, 18.5, 19.3, 18.3, 19.3, 26.4, 20.8, 16.1, 26.6, 22.0, 19.2, 
        20.7, 16.5, 13.4, 22.6, 20.6, 17.1, 17.9, 29.2, 12.7, 15.8, 20.2, 16.5, 
        15.8, 17.9, 15.4, 16.1, 18.7, 17.9, 13.1, 14.2, 24.7, 31.3, 23.5, 19.6, 
        25.5, 17.8, 21.1, 23.8, 25.9, 18.7, 24.9, 30.6, 13.8, 20.4, 17.9, 19.4, 
        20.5, 17.1, 20.4, 24.5, 26.7, 25.0, 24.2, 28.4, 21.1, 21.4, 24.8, 25.9, 
        23.8, 26.9, 23.7, 21.8, 23.5, 29.4, 33.3, 23.7, 29.1, 21.1, 21.2, 27.7, 
        29.5, 20.1, 20.1, 32.4, 25.6, 25.0, 23.2, 22.7, 26.6, 20.8, 24.2, 24.4, 
        26.4, 32.5, 27.6, 20.7, 30.9, 33.2, 28.1, 21.6, 30.6, 30.0, 25.3, 29.0, 
        27.8, 28.2, 23.9, 27.5, 29.4, 28.9, 26.8, 37.0, 28.9, 32.7
    ],
    'RH': [
        22, 26.5, 26, 25, 23.5, 29, 26, 29.5, 26.5, 24, 26, 29, 23.5, 30.5, 27, 27, 
        25, 22.5, 22.5, 28.5, 34, 27.5, 26.5, 30, 30, 32, 26.5, 35, 28.5, 32, 32, 
        32, 25.5, 26, 25, 29, 31, 24, 34.5, 27, 32, 33, 33, 29, 35, 30, 32.5, 38.5, 
        35.5, 33.5, 32.5, 32, 31, 29, 31, 24, 29, 32, 33, 25, 29.5, 33.5, 31, 23.5, 
        30.5, 40.5, 33.5, 28.5, 30, 28, 29, 33, 32, 34.5, 30, 33, 31.5, 32, 35, 34, 
        28, 28, 32, 35, 35, 41, 28.5, 29.5, 34, 35, 37.5, 33, 35, 33, 33, 37, 32, 
        33, 25, 34, 37, 32, 35, 39.5, 34, 43, 35, 35.5, 43, 33, 31, 31, 31, 31, 
        34.5, 40, 34, 39.5, 43, 31, 32, 32, 34, 36.5, 35, 28, 30, 40, 38, 31, 28, 
        43, 40.5, 43, 31.5, 38, 30, 39, 41.5, 37, 36.5, 36, 44, 27, 37, 33, 37, 
        40.5, 34.5, 38.5, 40, 38, 36.5, 38, 45, 34.5, 34, 38, 37.5, 37.5, 45, 37, 
        40.5, 44, 40, 45, 36, 44, 39.5, 41, 43, 39.5, 29, 39, 41, 38, 36.5, 40, 
        40, 38, 40, 41, 42, 40, 45, 37.5, 40, 42, 46, 40, 38, 45, 42, 42.5, 42, 
        41, 37.5, 38.5, 44, 38, 38.5, 43.5, 45, 44, 44
    ]
}

# Convert data to a DataFrame
df = pd.DataFrame(data)
# Constants
Ni = 300
N=205  # Number of iterations
Nc_values = range(3, 20)  # Range of Nc values from 3 to max available rows


mean_rmse_dict = {}
# Dictionary to store mean RMSE for each Nc
for Nc in Nc_values:
    rmse_list = []
    for i in range(Ni):
        # Randomly select Nc rows
      
# Step 1: Randomly select 3 points from the data
        selected_indices = sample(range(len(df)), Nc)
        selected_points = df.iloc[selected_indices]

        # Step 2: Fit a line to the selected points
        X_selected = selected_points[['RH']]
        y_selected = selected_points['F']
        model = LinearRegression().fit(X_selected, y_selected)

        

        # Get the slope (m) and intercept (b) of the line
        m = model.coef_[0]
        b = model.intercept_

        # Print the equation of the line

        # Step 3: Drop the selected points from the dataframe
        df_remaining = df.drop(selected_indices)

        # Step 4: Calculate Fe values for remaining RH values
        df_remaining['Fe'] = m * df_remaining['RH'] + b

        # Step 5: Calculate the error E = Fe - F
        df_remaining['E'] = df_remaining['Fe'] - df_remaining['F']

        rmse_iteration = np.sqrt(np.sum(df_remaining['E'] ** 2) / (N - Nc))
        rmse_list.append(rmse_iteration)
    
    if rmse_list:
        mean_rmse = sum(rmse_list) / len(rmse_list)
        mean_rmse_dict[Nc] = mean_rmse


print(mean_rmse_dict, Nc_values,sep='\n')


plt.figure(figsize=(10, 6))
plt.plot(list(mean_rmse_dict.keys()), list(mean_rmse_dict.values()), marker='o', color='b', label="Mean RMSE")
plt.xlabel('Nc')
plt.ylabel('Mean RMSE')
plt.title('Mean RMSE vs. Number of Selected Points (Nc)')
plt.legend()
plt.grid(True)
plt.show()
