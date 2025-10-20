import streamlit as st
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import re
from urllib.error import URLError

# =========================================
# PAGE CONFIGURATION
# =========================================
st.set_page_config(
    page_title="Indian Recipe Recommendation Bot",
    page_icon="🍛",
    layout="wide"
)

# =========================================
# CUSTOM CSS
# =========================================
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #FF6B6B;
        text-align: center;
        margin-bottom: 2rem;
    }
    .recipe-card {
        background-color: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        margin: 1rem 0;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
""", unsafe_allow_html=True)

# =========================================
# DATA LOADING AND CLEANING
# =========================================
@st.cache_data
def load_and_clean_data(file):
    """Load and clean the recipe dataset"""
    try:
        df = pd.read_csv(file)

        # --- Handle missing & inconsistent data ---
        df = df.drop_duplicates(subset=['TranslatedRecipeName'], keep='first')
        df['TranslatedRecipeName'] = df['TranslatedRecipeName'].fillna('Unknown Recipe')
        df['Cleaned-Ingredients'] = df['Cleaned-Ingredients'].fillna('')
        df['TranslatedInstructions'] = df['TranslatedInstructions'].fillna('Instructions not available')
        df['Cuisine'] = df['Cuisine'].fillna('Unknown')
        df['TotalTimeInMins'] = pd.to_numeric(df['TotalTimeInMins'], errors='coerce').fillna(0)
        df['image-url'] = df['image-url'].fillna('')

        # Clean ingredients column
        df['Cleaned-Ingredients'] = df['Cleaned-Ingredients'].apply(clean_ingredients)

        # Ingredient count
        if 'Ingredient-count' not in df.columns:
            df['Ingredient-count'] = df['Cleaned-Ingredients'].apply(
                lambda x: len([i for i in str(x).split(',') if i.strip()])
            )

        # Remove invalid rows
        df = df[df['Cleaned-Ingredients'].str.len() > 0].reset_index(drop=True)
        return df

    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None


def clean_ingredients(text):
    """Clean ingredient text"""
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[^a-z0-9,\s]', '', text)
    text = re.sub(r',+', ',', text)
    text = text.strip(', ')
    return text


# =========================================
# TF-IDF AND RECOMMENDATION FUNCTIONS
# =========================================
def create_tfidf_matrix(df):
    """Create TF-IDF matrix for ingredient matching"""
    tfidf = TfidfVectorizer(
        max_features=500,
        ngram_range=(1, 2),
        stop_words='english'
    )
    tfidf_matrix = tfidf.fit_transform(df['Cleaned-Ingredients'])
    return tfidf, tfidf_matrix


def get_recommendations_by_ingredients(user_ingredients, df, top_n=5):
    """Get recipe recommendations based on user ingredients"""
    user_ingredients = clean_ingredients(user_ingredients)
    tfidf, tfidf_matrix = create_tfidf_matrix(df)
    user_tfidf = tfidf.transform([user_ingredients])
    similarity_scores = cosine_similarity(user_tfidf, tfidf_matrix).flatten()
    top_indices = similarity_scores.argsort()[-top_n:][::-1]
    recommendations = df.iloc[top_indices].copy()
    recommendations['similarity_score'] = similarity_scores[top_indices]
    return recommendations


def get_recommendations_by_recipe(recipe_name, df, top_n=5):
    """Get similar recipes based on recipe name"""
    try:
        tfidf, tfidf_matrix = create_tfidf_matrix(df)
        idx = df[df['TranslatedRecipeName'].str.lower() == recipe_name.lower()].index[0]
        similarity_scores = cosine_similarity(tfidf_matrix[idx], tfidf_matrix).flatten()
        similarity_scores[idx] = -1  # Exclude same recipe
        top_indices = similarity_scores.argsort()[-top_n:][::-1]
        recommendations = df.iloc[top_indices].copy()
        recommendations['similarity_score'] = similarity_scores[top_indices]
        return recommendations
    except Exception as e:
        st.error(f"Error finding similar recipes: {e}")
        return None


# =========================================
# DISPLAY FUNCTION
# =========================================
def display_recipe(recipe, show_score=False):
    """Display a recipe card"""
    col1, col2 = st.columns([1, 2])

    # --- Left column: Image ---
    with col1:
        image_url = str(recipe.get('image-url', '')).strip()

        # Attempt to load image
        try:
            if image_url and image_url.startswith('http'):
                st.image(image_url, use_container_width=True)
            else:
                raise URLError("Invalid URL")
        except Exception:
            st.markdown("""
                <div style='background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                            padding: 80px 20px;
                            text-align: center;
                            border-radius: 10px;
                            color: white;
                            font-size: 60px;'>
                    🍛
                </div>
            """, unsafe_allow_html=True)

    # --- Right column: Details ---
    with col2:
        st.markdown(f"### 🍽️ {recipe['TranslatedRecipeName']}")
        if show_score:
            st.markdown(f"**Match Score:** {recipe['similarity_score']:.2%}")
        st.markdown(f"**Cuisine:** {recipe['Cuisine']}")
        st.markdown(f"**Cooking Time:** {int(recipe['TotalTimeInMins'])} minutes")
        st.markdown(f"**Ingredients Count:** {recipe['Ingredient-count']}")

        with st.expander("📝 View Ingredients"):
            ingredients_list = [ing.strip() for ing in str(recipe['Cleaned-Ingredients']).split(',') if ing.strip()]
            for ing in ingredients_list:
                st.markdown(f"- {ing}")

        with st.expander("👨‍🍳 View Instructions"):
            st.write(recipe['TranslatedInstructions'])

        if 'URL' in recipe and recipe['URL'] and str(recipe['URL']) != 'nan':
            st.markdown(f"[🔗 View Full Recipe]({recipe['URL']})")

    st.markdown("---")


# =========================================
# MAIN APP
# =========================================
def main():
    st.markdown("<h1 class='main-header'>🍛 Indian Recipe Recommendation Bot</h1>", unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload your Indian Food Recipes CSV file", type=['csv'])

    if uploaded_file is not None:
        with st.spinner("Loading and cleaning data..."):
            df = load_and_clean_data(uploaded_file)

        if df is not None:
            st.success(f"✅ Loaded {len(df)} recipes successfully!")

            # Sidebar filters
            st.sidebar.header("🔍 Filters")

            cuisines = ['All'] + sorted(df['Cuisine'].unique().tolist())
            selected_cuisine = st.sidebar.selectbox("Select Cuisine", cuisines)

            max_time = st.sidebar.slider("Maximum Cooking Time (minutes)", 0, int(df['TotalTimeInMins'].max()), int(df['TotalTimeInMins'].max()))
            max_ingredients = st.sidebar.slider("Maximum Ingredients", 1, int(df['Ingredient-count'].max()), int(df['Ingredient-count'].max()))

            filtered_df = df.copy()
            if selected_cuisine != 'All':
                filtered_df = filtered_df[filtered_df['Cuisine'] == selected_cuisine]
            filtered_df = filtered_df[filtered_df['TotalTimeInMins'] <= max_time]
            filtered_df = filtered_df[filtered_df['Ingredient-count'] <= max_ingredients]
            filtered_df = filtered_df.reset_index(drop=True)

            st.sidebar.info(f"📊 {len(filtered_df)} recipes match your filters")

            # Tabs
            tab1, tab2, tab3 = st.tabs(["🥘 Find by Ingredients", "🔎 Find Similar Recipes", "📊 Browse All"])

            # --- Tab 1: Ingredient Search ---
            with tab1:
                st.header("Find Recipes by Your Ingredients")
                user_input = st.text_area("Enter ingredients (comma-separated):", placeholder="e.g., tomato, onion, rice, chicken")
                num_recommendations = st.slider("Number of recommendations", 1, 10, 5)

                if st.button("🔍 Get Recommendations"):
                    if user_input.strip():
                        with st.spinner("Finding best matches..."):
                            recommendations = get_recommendations_by_ingredients(user_input, filtered_df, num_recommendations)
                        if len(recommendations) > 0:
                            st.success(f"Found {len(recommendations)} recipes!")
                            for _, recipe in recommendations.iterrows():
                                display_recipe(recipe, show_score=True)
                        else:
                            st.warning("No matching recipes found.")
                    else:
                        st.warning("Please enter some ingredients!")

            # --- Tab 2: Similar Recipe Search ---
            with tab2:
                st.header("Find Similar Recipes")
                if len(filtered_df) > 0:
                    recipe_names = sorted(filtered_df['TranslatedRecipeName'].tolist())
                    selected_recipe = st.selectbox("Choose a recipe", recipe_names)
                    num_similar = st.slider("Number of similar recipes", 1, 10, 5)

                    if st.button("🔍 Find Similar"):
                        with st.spinner("Finding similar recipes..."):
                            similar_recipes = get_recommendations_by_recipe(selected_recipe, filtered_df, num_similar)
                        if similar_recipes is not None and len(similar_recipes) > 0:
                            for _, recipe in similar_recipes.iterrows():
                                display_recipe(recipe, show_score=True)
                        else:
                            st.warning("No similar recipes found.")
                else:
                    st.warning("No recipes available. Adjust filters.")

            # --- Tab 3: Browse All ---
            with tab3:
                st.header("Browse All Recipes")
                search_term = st.text_input("🔍 Search recipes by name")
                display_df = filtered_df.copy()
                if search_term:
                    display_df = display_df[display_df['TranslatedRecipeName'].str.contains(search_term, case=False, na=False)]

                st.write(f"Showing {len(display_df)} recipes")
                if len(display_df) > 0:
                    recipes_per_page = 10
                    total_pages = (len(display_df) - 1) // recipes_per_page + 1
                    page = st.number_input("Page", 1, total_pages, 1)
                    start_idx = (page - 1) * recipes_per_page
                    end_idx = start_idx + recipes_per_page

                    for _, recipe in display_df.iloc[start_idx:end_idx].iterrows():
                        display_recipe(recipe)
                else:
                    st.info("No recipes found.")
    else:
        st.info("👆 Please upload the Indian Food Recipes CSV file to get started!")
        st.markdown("""
        ### How to use this app:
        1. Download the **Indian Food Recipes Dataset** from Kaggle  
        2. Upload the CSV file here  
        3. Use sidebar filters to refine recipes  
        4. Explore recommendations or similar recipes!
        """)


if __name__ == "__main__":
    main()
