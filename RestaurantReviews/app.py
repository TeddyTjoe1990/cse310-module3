import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Function to load and clean the dataset
def load_and_clean_data(filepath):
    """
    Loads the CSV file and cleans percentage columns.
    Converts YOY_Sales and YOY_Units from strings with '%' to float values.
    """
    df = pd.read_csv(filepath)
    df['YOY_Sales'] = df['YOY_Sales'].str.replace('%', '', regex=False).astype(float)
    df['YOY_Units'] = df['YOY_Units'].str.replace('%', '', regex=False).astype(float)
    return df

# Function to compute average unit sales based on franchising status
def average_unit_sales(df):
    """
    Returns the average Unit Volume grouped by Franchising status.
    """
    return df.groupby('Franchising')['Unit_Volume'].mean()

# Function to get the top 5 restaurants by unit growth
def top_unit_growth(df):
    """
    Returns the top 5 restaurants with the highest YOY Unit Growth.
    """
    return df[['Restaurant', 'YOY_Units']].sort_values(by='YOY_Units', ascending=False).head(5)

# Function to create a bar chart of top 10 restaurants by YOY sales growth
def plot_top_sales_growth(df):
    """
    Generates a horizontal bar chart showing the top 10 restaurants 
    with the highest YOY Sales Growth.
    """
    top_yoy_sales = df[['Restaurant', 'YOY_Sales']].sort_values(by='YOY_Sales', ascending=False).head(10)
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(top_yoy_sales['Restaurant'][::-1], top_yoy_sales['YOY_Sales'][::-1], color='skyblue')
    ax.set_xlabel('YOY Sales Growth (%)')
    ax.set_title('Top 10 Restaurants by YOY Sales Growth')
    return fig

# Function to calculate correlation between YOY_Units and YOY_Sales
def calculate_correlation(df, method):
    """
    Calculates correlation between YOY_Units and YOY_Sales using the chosen method.
    Supported methods: 'pearson', 'spearman', 'kendall'.
    """
    return df['YOY_Units'].corr(df['YOY_Sales'], method=method)

# Function to generate scatter plot for correlation
def plot_correlation(df, method):
    """
    Generates a scatter plot showing the relationship between 
    YOY Unit Growth and YOY Sales Growth using Seaborn.
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.scatterplot(x='YOY_Units', y='YOY_Sales', data=df, ax=ax, color='green')
    ax.set_title(f'YOY Unit Growth vs YOY Sales Growth ({method.title()} Correlation)')
    ax.set_xlabel('YOY Unit Growth (%)')
    ax.set_ylabel('YOY Sales Growth (%)')
    return fig

# Function to display dataset overview
def show_dataset_overview(df):
    """
    Displays the number of rows and columns, and shows the first few rows of the dataset.
    """
    st.write(f"Dataset contains **{df.shape[0]} rows** and **{df.shape[1]} columns**.")
    st.write("Here are the first few rows:")
    st.dataframe(df.head())

# ================= MAIN APP ==================

# Set the title of the Streamlit app
st.title("📊 Future 50 Restaurants Analysis")

# Load and clean the data
df = load_and_clean_data("Future50.csv")

# Show overview of the dataset
st.subheader("Dataset Overview")
show_dataset_overview(df)

# Question 1: Average unit sales by franchise status
st.subheader("1. Average Unit Sales by Franchise")
avg_unit_volume = average_unit_sales(df)
st.dataframe(avg_unit_volume)

# Question 2: Top 5 restaurants by annual unit growth
st.subheader("2. Top 5 Restaurants with the Highest Annual Unit Growth")
top_yoy_units = top_unit_growth(df)
st.dataframe(top_yoy_units)

# Question 3: Bar chart of top 10 restaurants by YOY sales growth
st.subheader("📈 Sales Growth Chart (YOY Sales)")
fig = plot_top_sales_growth(df)
st.pyplot(fig)

# Question 4: Correlation analysis between unit and sales growth
st.subheader("3. Correlation Between Unit Growth and Sales Growth")
method = st.selectbox("Select Correlation Method", options=["pearson", "spearman", "kendall"])
corr_value = calculate_correlation(df, method)
st.write(f"**{method.title()} Correlation:** {corr_value:.2f}")

# Show correlation scatter plot
fig2 = plot_correlation(df, method)
st.pyplot(fig2)

# Additional Insight: Bar chart showing distribution of franchised vs non-franchised restaurants
st.subheader("Additional Insight: Franchise Distribution")
franchise_counts = df['Franchising'].value_counts()
st.bar_chart(franchise_counts)

# Bonus Feature: Filter restaurants with YOY sales above a threshold
st.subheader("Bonus Filter: Restaurants with YOY Sales Above Threshold")
sales_threshold = st.slider("Minimum YOY Sales %", min_value=0, max_value=100, value=20)
filtered_df = df[df['YOY_Sales'] > sales_threshold][['Restaurant', 'YOY_Sales']]
st.dataframe(filtered_df)
