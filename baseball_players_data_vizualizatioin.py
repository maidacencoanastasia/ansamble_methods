import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Specify the CSV file path (replace with your actual file path)
csv_file_path = "datasets/baseball_players.csv"

# Read data from CSV using pandas
try:
    df = pd.read_csv(csv_file_path)
except FileNotFoundError:
    print(f"Error: CSV file not found at {csv_file_path}")
    exit(1)

# Select numeric columns for heatmap
numeric_columns = ["Height(inches)", "Weight(pounds)", "Age"]

# # Create the heatmap
# plt.figure(figsize=(8, 6))
# sns.heatmap(df[numeric_columns], cmap="coolwarm", annot=True)
# plt.title("Heatmap of Numeric Features")
# plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
# plt.tight_layout()
# plt.show()
#
# # Create the correlation map
# plt.figure(figsize=(8, 6))
# correlation_matrix = df[numeric_columns].corr()  # Calculate correlation coefficients
# sns.heatmap(correlation_matrix, cmap="coolwarm", annot=True)
# plt.title("Correlation Heatmap")
# plt.xticks(rotation=45)
# plt.yticks(rotation=0)  # Rotate y-axis labels to avoid overlapping
# plt.tight_layout()
# plt.show()
# Descriptive Statistics
# print("\nDescriptive Statistics:")
# print(df.describe(include='all'))  # Provides summary statistics for all columns
#
# # Grouped Statistics by Position
# print("\nAverage Height and Weight by Position:")
# position_averages = df.groupby('Position')[['Height(inches)', 'Weight(pounds)']].mean()
# print(position_averages)
#
# # Finding Players with Highest and Lowest Values
# print("\nPlayers with Highest and Lowest Values:")
# print(f"Player with Highest Weight: {df[df['Weight(pounds)'] == df['Weight(pounds)'].max()]}")
# print(f"Player with Lowest Height: {df[df['Height(inches)'] == df['Height(inches)'].min()]}")
#
# # Filtering by Age Range
# filtered_df = df[(df['Age'] >= 30) & (df['Age'] <= 35)]  # Adjust age range as needed
# print("\nPlayers Between 30 and 35 Years Old:")
# print(filtered_df)
plt.figure(figsize=(8, 6))
plt.hist(df["Height(inches)"], bins=10, edgecolor="black")  # Adjust bins as needed
plt.xlabel("Height (inches)")
plt.ylabel("Number of Players")
plt.title("Distribution of Player Heights")
plt.grid(True)
plt.show()


plt.figure(figsize=(10, 6))
sns.boxplot(
    x="Position",
    y="Height(inches)",
    showmeans=True,  # Display means as markers
    data=df
)
plt.title("Height Distribution by Position")
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.boxplot(
    x="Position",
    y="Weight(pounds)",
    showmeans=True,  # Display means as markers
    data=df
)
plt.title("Weight Distribution by Position")
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(df["Height(inches)"], df["Weight(pounds)"])
plt.xlabel("Height (inches)")
plt.ylabel("Weight (pounds)")
plt.title("Height vs. Weight")
plt.grid(True)
plt.show()