# Future 50 Restaurants Analysis

import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Future50.csv")

# Clean and convert percentage columns
df['YOY_Sales'] = df['YOY_Sales'].str.replace('%', '').astype(float)
df['YOY_Units'] = df['YOY_Units'].str.replace('%', '').astype(float)

# Question 1: Average Unit Volume by Franchising
avg_unit_volume = df.groupby('Franchising')['Unit_Volume'].mean()
print("Rata-rata Unit Volume berdasarkan Franchising:")
print(avg_unit_volume)

# Question 2: Top 5 restaurants by YOY_Units
top_yoy_units = df[['Restaurant', 'YOY_Units']].sort_values(by='YOY_Units', ascending=False).head(5)
print("\nTop 5 restoran berdasarkan pertumbuhan unit tahunan:")
print(top_yoy_units)

# Visualization: Top 10 by YOY_Sales
top_yoy_sales = df[['Restaurant', 'YOY_Sales']].sort_values(by='YOY_Sales', ascending=False).head(10)
plt.figure(figsize=(12, 6))
plt.barh(top_yoy_sales['Restaurant'][::-1], top_yoy_sales['YOY_Sales'][::-1], color='skyblue')
plt.xlabel('YOY Sales Growth (%)')
plt.title('Top 10 Restaurants by YOY Sales Growth')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()
