import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the pre-trained model
@st.cache_resource
def load_model():
    with open(r'C:\Users\Admin\Desktop\Yashraj\ML\Product Recommendation\product_recommender_model.pkl', 'rb') as f:
        model_data = pickle.load(f)
    return model_data

# Recommendation function
def get_recommendations(product_id, model_data, num_recommendations=10):
    df = model_data['df']
    cosine_sim = model_data['cosine_similarity_matrix']
    
    if product_id not in df['product_id'].values:
        return None
    
    idx = df[df['product_id'] == product_id].index[0]
    sim_scores = list(enumerate(cosine_sim[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
    sim_scores = sim_scores[1:num_recommendations+1]
    product_indices = [i[0] for i in sim_scores]
    
    return df.iloc[product_indices][['product_id', 'product_name', 'main_category', 
                                   'discounted_price', 'actual_price', 'rating', 'img_link']]

# Streamlit app
def main():
    st.title("Amazon Product Recommendation System")
    
    # Load model
    model_data = load_model()
    df = model_data['df']
    
    # Product selection
    st.header("Select a Product")
    
    # Create a dropdown with product names and IDs
    product_options = [f"{row['product_name'][:50]}... ({row['product_id']})" 
                      for _, row in df.iterrows()]
    
    selected_product = st.selectbox("Choose a product:", product_options)
    selected_product_id = selected_product.split('(')[-1].replace(')', '').strip()
    
    if st.button("Get Recommendations"):
        with st.spinner('Finding similar products...'):
            recommendations = get_recommendations(selected_product_id, model_data)
            
            if recommendations is not None:
                st.header("Recommended Products")
                
                # Display the selected product
                selected_product_data = df[df['product_id'] == selected_product_id].iloc[0]
                st.subheader("Selected Product:")
                col1, col2 = st.columns([1, 2])
                with col1:
                    st.image(selected_product_data['img_link'], width=150)
                with col2:
                    st.write(f"**{selected_product_data['product_name']}**")
                    st.write(f"Category: {selected_product_data['main_category']}")
                    st.write(f"Price: ₹{selected_product_data['discounted_price']} (Original: ₹{selected_product_data['actual_price']})")
                    st.write(f"Rating: {selected_product_data['rating']}/5")
                
                st.divider()
                
                # Display recommendations
                st.subheader("Similar Products:")
                for _, product in recommendations.iterrows():
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.image(product['img_link'], width=120)
                    with col2:
                        st.write(f"**{product['product_name']}**")
                        st.write(f"Category: {product['main_category']}")
                        st.write(f"Price: ₹{product['discounted_price']} (Original: ₹{product['actual_price']})")
                        st.write(f"Rating: {product['rating']}/5")
                    st.divider()
            else:
                st.error("Product not found in database.")

if __name__ == "__main__":
    main()