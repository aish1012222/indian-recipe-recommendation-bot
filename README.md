# 🍛 Indian Recipe Recommendation Bot

This **Streamlit web app** recommends Indian recipes based on the ingredients you have or by finding similar recipes from a dataset.  
It uses **TF-IDF (Term Frequency–Inverse Document Frequency)** and **Cosine Similarity** to match your input with the most relevant recipes.

---

## 🚀 Features

✅ **Ingredient-Based Search** – Get recipes by entering available ingredients  
✅ **Similar Recipe Finder** – Discover recipes similar to one you like  
✅ **Filters** – Filter by cuisine, cooking time, and ingredient count  
✅ **Interactive Interface** – Built with Streamlit for a beautiful and easy-to-use experience  
✅ **Smart Cleaning** – Cleans and preprocesses dataset automatically  

---

## 🧠 Tech Stack

- **Python 3**
- **Streamlit**
- **Pandas / NumPy**
- **Scikit-learn** (TF-IDF & Cosine Similarity)
- **Regex** for text preprocessing

---

## 📂 Dataset

You can use the **Indian Food Recipes Dataset** available on Kaggle:  
🔗 [https://www.kaggle.com/datasets/nehaprabhavalkar/indian-food-recipes](https://www.kaggle.com/datasets/nehaprabhavalkar/indian-food-recipes)

Make sure your CSV file includes the following important columns:
- `TranslatedRecipeName`
- `Cleaned-Ingredients`
- `TranslatedInstructions`
- `Cuisine`
- `TotalTimeInMins`
- `image-url`

---

## ⚙️ Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/<your-username>/indian-recipe-recommendation-bot.git
   cd indian-recipe-recommendation-bot
