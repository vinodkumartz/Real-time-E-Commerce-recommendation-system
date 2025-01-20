import streamlit as st
import pandas as pd
import pickle
from models import ContentBasedRecommender, HybridRecommender    # Importing the models from models.py

def load_content_based_model(filename='content_based_recommender.pkl'):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded successfully from {filename}.")
    return model

def load_hybrid_model(filename='hybrid_recommender_model.pkl'):
    with open(filename, 'rb') as file:
        model = pickle.load(file)
    print(f"Model loaded successfully from {filename}.")
    return model

# Load all models
content_based_model = load_content_based_model()
hybrid_model = load_hybrid_model()

# Streamlit UI
st.title("E-commerce Recommendation System")

# Sidebar menu for selecting recommendation approach
rec_type = st.sidebar.selectbox("Select Recommendation Type", ["Content-Based", "Hybrid"])

st.header("Get Product Recommendations")
    
# Input for product name
product_name = st.text_input("Enter Product Name:")
    
# Input for the number of recommendations
nbr = st.number_input("Number of Recommendations", min_value=1, max_value=20, value=10)
    
# Input for user ID (for hybrid model)
target_user_id = st.number_input("Enter User ID:", min_value=1, value=1)  # Adjust default user ID accordingly
    
if st.button("Get Recommendations"):
    if product_name.strip() == "":
        st.warning("Please enter a product name.")
    else:
        # Get recommendations based on the selected recommendation type
        if rec_type == "Content-Based":
            recommendations = content_based_model.recommend(item_name=product_name, top_n=nbr)
        elif rec_type == "Hybrid":
            recommendations = hybrid_model.recommend(target_user_id=target_user_id, item_name=product_name, top_n=nbr)
        
        # Display recommendations
        if recommendations.empty:
            st.warning("No recommendations available for this product.")
        else:
            st.subheader("Recommended Products")
            for idx, row in recommendations.iterrows():
                st.write(f"**{row['Name']}** (Brand: {row['Brand']}, Rating: {row['Rating']})")
                st.write(f"Review Count: {row['ReviewCount']}")
