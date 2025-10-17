import streamlit as st
import pandas as pd
import joblib
import shap
import numpy as np
import matplotlib.pyplot as plt

# ✅ MUST be the first Streamlit command
st.set_page_config(
    page_title="McDonald's Sentiment Analysis Dashboard",
    layout="wide"
)

# -------------------------------
# Load Pre-trained Model
# -------------------------------
@st.cache_resource
def load_model():
    """Loads the pre-trained sentiment model."""
    try:
        # Note: The model is assumed to handle the internal text vectorization
        # and expects a DataFrame with all original features.
        model = joblib.load("best_mcdonalds_sentiment_model_fixed.pkl")
        return model
    except Exception as e:
        # This will only be shown if the file 'best_mcdonalds_sentiment_model_fixed.pkl' is not found.
        st.error(f"❌ Failed to load model. Please ensure 'best_mcdonalds_sentiment_model_fixed.pkl' is in the current directory. Error: {e}")
        return None

model = load_model()

# -------------------------------
# Sidebar Navigation
# -------------------------------
page = st.sidebar.radio(
    "📑 Navigation",
    [
        "1️⃣ Problem Statement & Ethics",
        "2️⃣ Data Collection",
        "3️⃣ Models & Comparison",
        "4️⃣ Predictions, LIME & SHAP",
        "5️⃣ Insights & Real-World Impact"
    ]
)

# -------------------------------
# 1️⃣ Problem Statement & Ethics
# -------------------------------
if page == "1️⃣ Problem Statement & Ethics":
    st.title("🍔 McDonald's Sentiment Analysis")
    st.subheader("Problem Statement")
    st.write("""
    In the fast-food industry, customer feedback is crucial to improving product quality and service.
    This project aims to build an AI-based sentiment analysis model to analyze customer reviews of McDonald's products 
    and services to classify sentiments as **positive** or **negative**, helping the company identify improvement areas.
    """)

    st.subheader("Applications")
    st.write("""
    - Monitor customer satisfaction in real-time  
    - Identify trending complaints or praise points  
    - Support marketing strategies with customer sentiment insights  
    - Assist customer service teams with automated triaging
    """)

    st.subheader("Why this Problem is Important")
    st.write("""
    With thousands of reviews generated daily across platforms, manual analysis is not scalable.
    Automating this process provides faster insights and better decision-making for McDonald's management.
    """)

    st.subheader("Ethical Considerations")
    st.write("""
    - Ensure data is anonymized to protect user privacy  
    - Avoid biased model predictions by regularly auditing model fairness  
    - Provide explainability using SHAP and LIME for transparency  
    """)

# -------------------------------
# 2️⃣ Data Collection
# -------------------------------
elif page == "2️⃣ Data Collection":
    st.title("📊 Data Collection")
    st.write("""
    **Type of Data Collected:**  
    - Text reviews from McDonald’s customers  
    - Each review is labeled as **positive** or **negative**
    """)

    st.write("""
    **How Data is Collected:**  
    - Publicly available datasets (e.g., Kaggle)  
    - Web scraping of McDonald’s review sites and social media posts  
    - Preprocessing includes cleaning, tokenizing, removing stopwords, and labeling
    """)

    sample_data = pd.DataFrame({
        "Review": [
            "The fries were cold and soggy.",
            "I loved the Big Mac!",
            "Service was too slow.",
            "Clean and friendly atmosphere!"
        ],
        "Sentiment": ["Negative", "Positive", "Negative", "Positive"]
    })
    st.dataframe(sample_data)

# -------------------------------
# 3️⃣ Models & Comparison
# -------------------------------
elif page == "3️⃣ Models & Comparison":
    st.title("🤖 Model Comparison & Selection")

    st.write("""
    We trained multiple models including:  
    - Logistic Regression  
    - Naive Bayes  
    - Random Forest  
    - Support Vector Machines (SVM)  

    After evaluation, the best-performing model was selected and saved as a `.pkl` file.
    """)

    # Example results (replace with real values if needed)
    results = {
        "Logistic Regression": 0.94,
        "Naive Bayes": 0.89,
        "Random Forest": 0.87,
        "SVM": 0.91
    }

    df_results = pd.DataFrame(list(results.items()), columns=["Model", "Accuracy"])
    st.bar_chart(df_results.set_index("Model"))

    st.success(f"✅ Best Model Selected: Logistic Regression with Accuracy = {results['Logistic Regression']*100:.2f}%")

# -------------------------------
# 4️⃣ Predictions, LIME & SHAP
# -------------------------------
elif page == "4️⃣ Predictions, LIME & SHAP":
    st.title("🔮 Predictions, LIME & SHAP Explainability")

    if model is None:
        st.warning("⚠️ Model not loaded. Please ensure your `.pkl` file is available.")
    else:
        st.subheader("Enter a Customer Review")
        user_input = st.text_area("✍️ Type a review here", "The burger was amazing and fresh!")

        if st.button("Predict Sentiment"):
            try:
                
                # --- FIX START ---
                # The model expects a DataFrame with all original columns, even if only the text is user-defined.
                required_cols = {
                    'City': 'N/A',
                    'Price': 0.0,
                    'Calories': 0.0,
                    'Offer_Type': 'N/A',
                    'no_of_orders': 0,
                    'Category': 'N/A'
                }

                # Start the input data with the review text
                input_data = {'review_text': [user_input]}
                
                # Add placeholder values for all other required columns
                input_data.update({col: [val] for col, val in required_cols.items()})
                
                # Create the complete DataFrame
                input_df = pd.DataFrame(input_data)
                # --- FIX END ---
                
                prediction = model.predict(input_df)[0]
                probas = model.predict_proba(input_df)[0]
                
                sentiment_label = "Positive" if prediction == 1 else "Negative"
                
                if sentiment_label == "Positive":
                    st.success(f"**Prediction:** {sentiment_label}")
                else:
                    st.error(f"**Prediction:** {sentiment_label}")

                st.info(f"**Confidence:** {np.max(probas)*100:.2f}%")

                # LIME explanation
                st.subheader("LIME Explanation")
                # Define the prediction function for LIME: it must take a list of raw strings and return an array of probabilities
                # Note: The lambda function here must create a full DataFrame for LIME to work.
                def predictor(texts):
                    # LIME will pass a list of strings (texts). We must create a full DataFrame for each string.
                    lime_data = {'review_text': texts}
                    
                    # Add placeholders for the other columns for all rows
                    for col, val in required_cols.items():
                        lime_data[col] = [val] * len(texts)
                    
                    lime_df = pd.DataFrame(lime_data)
                    return model.predict_proba(lime_df)
                
                explainer = LimeTextExplainer(class_names=["Negative", "Positive"])
                exp = explainer.explain_instance(
                    user_input, 
                    predictor, # Use the defined predictor function
                    num_features=5
                )
                # Render LIME explanation HTML
                st.components.v1.html(exp.as_html(), height=400)

                # SHAP explanation
                st.subheader("SHAP Explanation (Feature Importance)")
                
                # We need to run the explainer with a full DataFrame structure.
                # For SHAP text plots, it's often more complex due to tokenization/masking.
                # Since LIME provides a word-level explanation, we will use a simpler SHAP plot (waterfall) 
                # based on the *instance* prediction for demonstration, avoiding the complex text masking setup.
                
                # Predict the probability using the input_df (which has all cols)
                # We need the base value (expected value) and shap_values for the single instance
                # Since we don't have the original data/vectorizer, the full SHAP pipeline is hard to replicate,
                # but we can try the text explainer assuming the model object handles everything.
                
                try:
                    # SHAP Text Explainer (often unstable without full training pipeline)
                    explainer_shap = shap.Explainer(
                        lambda x: model.predict_proba(pd.DataFrame({"review_text": x, **{col: [val] for col, val in required_cols.items()}})),
                        masker=shap.maskers.Text(tokenizer=str.split) # Assuming simple split tokenizer
                    )
                    
                    # For a single prediction, we get one list of SHAP values
                    shap_values = explainer_shap([user_input]) 
                    
                    # Ensure Matplotlib figures are handled correctly
                    st.set_option('deprecation.showPyplotGlobalUse', False)
                    
                    # Generate the plot
                    shap.plots.text(shap_values[0])
                    
                    # Use plt.gcf() to get the current figure and display it
                    fig = plt.gcf() 
                    st.pyplot(fig, bbox_inches='tight')
                    plt.clf() # Clear the plot buffer

                except Exception as shap_e:
                     st.warning(f"Could not generate SHAP text plot due to complexity. LIME explanation is shown above. SHAP Error: {shap_e}")
                

            except Exception as e:
                # The primary prediction error (which was missing columns) is now handled
                st.error(f"❌ Prediction failed: {e}")

# -------------------------------
# 5️⃣ Insights & Real-World Impact
# -------------------------------
elif page == "5️⃣ Insights & Real-World Impact":
    st.title("🌍 Insights & Real-World Applications")

    st.write("""
    **Key Insights:**  
    - Positive sentiments are often associated with food taste, freshness, and cleanliness.  
    - Negative sentiments revolve around slow service, cold food, or hygiene issues.  
    - Using SHAP and LIME, we can explain why a review was classified as positive or negative.

    **Real-World Applications:**  
    - Management can identify problem areas and improve customer experience.  
    - Marketing can promote aspects that customers love.  
    - Automated dashboards can monitor sentiment in real-time to respond quickly.
    """)

    st.success("✅ The AI-powered sentiment analysis system provides valuable insights for business growth and customer satisfaction.")
