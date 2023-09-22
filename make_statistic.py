import pandas as pd
import numpy as np

# Create synthetic data
np.random.seed(0)  # for reproducibility

# Generate random data for the number of hours spent on the phone (target)
hours_spent = np.random.randint(1, 10, size=100)  # Generate 100 random values between 1 and 10

# Generate random data for the person's eyesight (feature)
eyesight = np.random.normal(20, 5, size=100)  # Generate 100 random values from a normal distribution with mean 20 and standard deviation 5

# Create a DataFrame to store the data
data = pd.DataFrame({'Hours_Spent_on_Phone': hours_spent, 'Eyesight': eyesight})

# Save the dataset to a CSV file
data.to_csv('dataset.csv', index=False)
