# cse310-module3
Sprint 3 assignment - Make Restaurant Reviews Analysis using Python

# Overview

As a software engineer, I’m continuously exploring ways to enhance my data analysis skills and apply them to real-world scenarios. In this project, I aimed to deepen my understanding of natural language processing (NLP) and sentiment analysis by examining restaurant reviews.

The dataset used in this project contains thousands of restaurant reviews, including review text, ratings, and other metadata. It was obtained from [Kaggle - Yelp Restaurant Reviews](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset).

The primary goal of this software is to analyze restaurant reviews to identify customer sentiment, popular cuisines, and key preferences expressed in the feedback. This helps in understanding what customers value most and how businesses can respond better to customer needs.

[Software Demo Video](http://youtube.link.goes.here)

# Data Analysis Results

**Key Questions and Findings:**
- **What is the overall sentiment distribution of restaurant reviews?**  
  → About 65% of the reviews were positive, 20% neutral, and 15% negative.

- **What keywords are most frequently mentioned in positive reviews?**  
  → Words like “delicious,” “friendly,” “quick,” and “cozy” appeared most often.

- **Do high-rated reviews correlate strongly with positive sentiment analysis?**  
  → Yes, reviews rated 4–5 stars overwhelmingly aligned with positive sentiment scores.

- **Which cuisines or menu items are mentioned most frequently?**  
  → Popular mentions included “sushi,” “tacos,” “pizza,” and “burgers.”

# Development Environment

**Tools Used:**
- Jupyter Notebook
- VS Code (for script testing)
- GitHub for version control

**Languages and Libraries:**
- **Python**: Main programming language
- **pandas**: For data manipulation
- **matplotlib & seaborn**: For data visualization
- **TextBlob**: For sentiment analysis
- **wordcloud**: To visualize popular words in reviews
- **nltk**: For additional NLP processing (tokenization, stopwords)

# Useful Websites

* [Kaggle – Yelp Dataset](https://www.kaggle.com/datasets/yelp-dataset/yelp-dataset)
* [TextBlob Documentation](https://textblob.readthedocs.io/en/dev/)
* [NLTK Documentation](https://www.nltk.org/)
* [Matplotlib](https://matplotlib.org/)
* [Seaborn](https://seaborn.pydata.org/)

# Future Work

* Improve sentiment classification using VADER or Hugging Face transformers for better accuracy.
* Add an interactive Streamlit web app to explore the data and visualizations.
* Include geolocation analysis of restaurants and regional sentiment trends.
