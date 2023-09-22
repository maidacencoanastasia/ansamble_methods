import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load your dataset (replace 'your_dataset.csv' with your actual dataset)
data = pd.read_csv("test_dataset.csv")
'''
# Pairplot to visualize relationships between numeric features
sns.pairplot(data, vars=['age', 'height(cm)', 'weight(kg)', 'waist(cm)', 'systolic', 'relaxation', 'hemoglobin'])
plt.title('Pairplot of Numeric Features')
plt.show()

# Correlation matrix heatmap
corr_matrix = data.corr()
plt.figure(figsize=(12, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix Heatmap')
plt.show()

# Histogram of hemoglobin levels
plt.figure(figsize=(8, 6))
sns.histplot(data['hemoglobin'], kde=True, color='blue')
plt.title('Distribution of Hemoglobin Levels')
plt.xlabel('Hemoglobin Level')
plt.ylabel('Frequency')
plt.show()

'''
# Create subplots for histograms
plt.figure(figsize=(8, 6))

# Histogram: Age vs. Dental Caries
plt.hist(data[data['dental caries'] > 0]['age'], bins=20, color='blue', alpha=0.7, label='With Caries')
plt.hist(data[data['dental caries'] == 0]['age'], bins=20, color='gray', alpha=0.7, label='No Caries')
plt.title('Histogram: Age vs. Dental Caries')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()

# Categorize fasting blood sugar as 'Normal' or 'High'
data.loc[data['fasting blood sugar'] > 100, 'blood_sugar_category'] = 'High'
# Create a histogram: Age vs. Fasting Blood Sugar Category
plt.figure(figsize=(8, 6))
plt.hist(data[data['blood_sugar_category'] == 'High']['age'], bins=20, color='red', alpha=0.7, label='High Sugar')
plt.title('Histogram: Age vs. Fasting Blood Sugar Category')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.legend()

plt.show()