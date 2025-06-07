import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
df = pd.read_csv("Future50.csv")

# Clean and convert percentage columns
df['YOY_Sales'] = df['YOY_Sales'].str.replace('%', '').astype(float)
df['YOY_Units'] = df['YOY_Units'].str.replace('%', '').astype(float)

st.title("📊 Analisis Future 50 Restaurants")

# Question 1: Average Unit Volume by franchise status
avg_unit_volume = df.groupby('Franchising')['Unit_Volume'].mean()
st.subheader("1. Average Unit Sales by Franchise")
st.dataframe(avg_unit_volume)

# Question 2: Top 5 restaurants by annual unit growth (YOY_Units)
top_yoy_units = df[['Restaurant', 'YOY_Units']].sort_values(by='YOY_Units', ascending=False).head(5)
st.subheader("2. Top 5 Restaurants with the Highest Annual Unit Growth")
st.dataframe(top_yoy_units)

# Chart: Top 10 restaurants by YOY_Sales
top_yoy_sales = df[['Restaurant', 'YOY_Sales']].sort_values(by='YOY_Sales', ascending=False).head(10)
fig, ax = plt.subplots(figsize=(10, 6))
ax.barh(top_yoy_sales['Restaurant'][::-1], top_yoy_sales['YOY_Sales'][::-1], color='skyblue')
ax.set_xlabel('YOY Sales Growth (%)')
ax.set_title('Top 10 Restaurants by YOY Sales Growth')
st.subheader("📈 Sales Growth Chart (YOY Sales)")
st.pyplot(fig)
