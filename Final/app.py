"""
Streamlit Frontend for Suicide Expression Classification
Supports direct text input and Instagram comment analysis with SHAP explainability
"""

import streamlit as st
import pandas as pd
import torch
from pathlib import Path
import os
import time
from datetime import datetime
import matplotlib.pyplot as plt
import io
import base64

# Import project utilities
from src.predict import get_model_and_tokenizer, predict_texts
from src.shap_utils import get_predict_proba, get_shap_explainer
from src.data_collection.instagram_comments import fetch_comments
import shap


# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Suicide Expression Classifier",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 20px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .warning-card {
        background: #fff3cd;
        border: 1px solid #ffc107;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
    .success-card {
        background: #d4edda;
        border: 1px solid #28a745;
        padding: 15px;
        border-radius: 5px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# CACHING & MODEL LOADING
# ============================================================================
@st.cache_resource
def load_model_and_tokenizer():
    """Load model and tokenizer once and cache for all sessions"""
    model_dir = Path(__file__).resolve().parent / 'model_test'
    tokenizer, model = get_model_and_tokenizer(model_dir=model_dir)
    return tokenizer, model


@st.cache_resource
def get_shap_explainer_cached(_tokenizer, _model):
    """Cache SHAP explainer"""
    device = torch.device('cpu')
    predict_proba = get_predict_proba(_model, _tokenizer, device=device)
    explainer = get_shap_explainer(predict_proba, _tokenizer)
    return explainer


# ============================================================================
# PREDICTION & SHAP FUNCTIONS
# ============================================================================
def predict_multiple_texts(texts, tokenizer, model, device='cpu'):
    """Predict labels for multiple texts"""
    results = predict_texts(texts, tokenizer, model, device=device)
    
    output = []
    for r in results:
        label = 'Suicidal' if r['pred'] == 1 else 'Non-Suicidal'
        confidence = r['prob']
        output.append({
            'text': r['text'],
            'cleaned_text': r.get('cleaned_text', r['text']),  # Get cleaned version
            'prediction': label,
            'confidence': confidence,
            'probs': r['probs']
        })
    return output


def generate_shap_explanation(text, tokenizer, model, explainer=None):
    """
    Generate SHAP explanation for a single text
    Currently returning None to avoid infinite loop issues
    """
    try:
        from src.preprocess_text import clean_text
        
        device = torch.device('cpu')
        
        # Validate input
        if not isinstance(text, str):
            text = str(text)
        
        original_text = text
        text = clean_text(text)  # Clean emojis
        text = text.strip()
        
        if not text:
            st.warning("⚠️ Text is empty")
            return None
        
        # Temporary fix: Skip SHAP to avoid infinite loop
        # SHAP with Text masker causes issues in Streamlit
        st.info("ℹ️ Detailed SHAP explanations temporarily disabled due to computation optimization")
        return None
                
    except Exception as e:
        error_msg = f"Error: {str(e)[:80]}"
        st.error(f"❌ {error_msg}")
        return None


# ============================================================================
# SIDEBAR
# ============================================================================
# REMOVED - All content on main page now


# ============================================================================
# MAIN CONTENT
# ============================================================================
st.title("🔍 Suicide Expression Classification")
st.markdown("Analyze text for suicidal expressions with AI-powered classification and SHAP explainability")

# Load model
with st.spinner("Loading model and tokenizer..."):
    tokenizer, model = load_model_and_tokenizer()
    explainer = get_shap_explainer_cached(tokenizer, model)

st.success("✅ Model loaded successfully!")

st.markdown("---")

# Input Mode Selection
col1, col2 = st.columns(2)
with col1:
    input_mode = st.radio(
        "**Select Input Mode:**",
        ["📝 Direct Text Input", "📸 Instagram Link"],
        horizontal=True
    )

with col2:
    # SHAP explanations disabled due to computation optimization
    show_shap = False  # Disabled
    # st.checkbox("🔬 Show SHAP Explanations", value=False)

st.markdown("---")


# ============================================================================
# DIRECT TEXT INPUT MODE
# ============================================================================
if input_mode == "📝 Direct Text Input":
    st.subheader("Enter Text for Classification")
    
    input_type = st.radio(
        "Choose input type:",
        ["Single Text", "Multiple Texts"],
        horizontal=True
    )
    
    if input_type == "Single Text":
        user_text = st.text_area(
            "Enter your text:",
            height=120,
            placeholder="Type or paste the text you want to analyze...",
            label_visibility="collapsed"
        )
        
        if st.button("🔍 Classify", type="primary", use_container_width=True):
            if not user_text.strip():
                st.error("❌ Please enter some text to classify")
            else:
                with st.spinner("Analyzing text..."):
                    device = torch.device('cpu')
                    results = predict_multiple_texts([user_text], tokenizer, model, device=device)
                    result = results[0]
                    
                    st.markdown("---")
                    st.subheader("📊 Classification Results")
                    
                    # Display metrics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Prediction", result['prediction'])
                    with col2:
                        st.metric("Confidence", f"{result['confidence']*100:.1f}%")
                    with col3:
                        risk_level = "🔴 HIGH RISK" if result['prediction'] == 'Suicidal' else "🟢 LOW RISK"
                        st.metric("Risk Level", risk_level)
                    
                    # Results table
                    df_single = pd.DataFrame([{
                        'Text': user_text,
                        'Prediction': result['prediction'],
                        'Confidence': f"{result['confidence']*100:.1f}%"
                    }])
                    st.dataframe(df_single, use_container_width=True)
                    
                    # SHAP Explanation
                    if show_shap:
                        st.markdown("---")
                        st.subheader("🔬 SHAP Explanation - Word Importance")
                        st.markdown("The visualization shows how different words contribute to the classification:")
                        
                        with st.spinner("Generating SHAP explanation..."):
                            shap_html = generate_shap_explanation(user_text, tokenizer, model, explainer)
                            if shap_html:
                                st.components.v1.html(shap_html, height=300, scrolling=True)
    
    else:  # Multiple Texts
        texts_input = st.text_area(
            "Enter texts (one per line):",
            height=120,
            placeholder="Text 1\nText 2\nText 3\n...",
            label_visibility="collapsed"
        )
        
        if st.button("🔍 Classify All", type="primary", use_container_width=True):
            texts_list = [t.strip() for t in texts_input.split('\n') if t.strip()]
            
            if not texts_list:
                st.error("❌ Please enter at least one text")
            else:
                with st.spinner(f"Analyzing {len(texts_list)} texts..."):
                    device = torch.device('cpu')
                    results = predict_multiple_texts(texts_list, tokenizer, model, device=device)
                    
                    st.markdown("---")
                    st.subheader("📊 Classification Results")
                    
                    # Summary statistics
                    suicidal_count = sum(1 for r in results if r['prediction'] == 'Suicidal')
                    non_suicidal_count = len(results) - suicidal_count
                    
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Analyzed", len(results))
                    with col2:
                        st.metric("Suicidal", suicidal_count)
                    with col3:
                        st.metric("Non-Suicidal", non_suicidal_count)
                    
                    # Results table
                    df_results = pd.DataFrame({
                        'Text': [r['text'][:80] + ('...' if len(r['text']) > 80 else '') for r in results],
                        'Prediction': [r['prediction'] for r in results],
                        'Confidence': [f"{r['confidence']*100:.1f}%" for r in results]
                    })
                    
                    st.dataframe(df_results, use_container_width=True, height=400)
                    
                    # SHAP for each text
                    if show_shap:
                        st.markdown("---")
                        st.subheader("🔬 SHAP Explanations")
                        
                        for idx, (text, result) in enumerate(zip(texts_list, results)):
                            with st.expander(f"**Text {idx+1}:** {result['prediction']} ({result['confidence']*100:.1f}%)"):
                                st.text(f"{text}")
                                
                                with st.spinner(f"Generating explanation for text {idx+1}..."):
                                    shap_html = generate_shap_explanation(text, tokenizer, model, explainer)
                                    if shap_html:
                                        st.components.v1.html(shap_html, height=250, scrolling=True)


# ============================================================================
# INSTAGRAM INPUT MODE
# ============================================================================
else:  # Instagram Link
    st.subheader("Analyze Instagram Post Comments")
    
    # Check API token status
    apify_token = os.getenv("APIFY_API_TOKEN")
    if apify_token:
        st.success("✅ APIFY_API_TOKEN is configured")
    else:
        st.warning("⚠️ APIFY_API_TOKEN not configured - Instagram features won't work")
    
    instagram_url = st.text_input(
        "Enter Instagram Post URL:",
        placeholder="https://www.instagram.com/p/ABC123XYZ/",
        help="Full URL to the Instagram post"
    )
    
    comment_limit = st.slider(
        "Number of Comments to Fetch:",
        min_value=1,
        max_value=200,
        value=10,
        step=5
    )
    
    if st.button("🔍 Fetch and Analyze", type="primary", use_container_width=True):
        if not instagram_url.strip():
            st.error("❌ Please enter an Instagram URL")
        else:
            try:
                # Check for API token
                if not os.getenv("APIFY_API_TOKEN"):
                    st.error(
                        "❌ APIFY_API_TOKEN not set. Please set it in your `.env` file to use Instagram features."
                    )
                else:
                    with st.spinner(f"Fetching {comment_limit} comments from Instagram (this may take a moment)..."):
                        comments = fetch_comments(instagram_url, limit=comment_limit, debug=True)
                    
                    if not comments:
                        st.error("❌ No comments found. Check the URL or try again.")
                        
                        with st.expander("🔧 Troubleshooting Help"):
                            st.markdown("""
                            **Possible reasons:**
                            - The Instagram post URL might be invalid
                            - The post might have no comments
                            - The post might be private or deleted
                            - Your APIFY_API_TOKEN might be invalid
                            
                            **Solutions:**
                            1. Verify the URL is a valid Instagram post URL
                            2. Make sure the post is public and has comments
                            3. Try a different post URL
                            4. Check your APIFY_API_TOKEN in the .env file
                            """)
                    else:
                        st.success(f"✅ Fetched {len(comments)} comments from Instagram")
                        st.info(f"📊 Requested: {comment_limit} | Received: {len(comments)}")
                        
                        # Show parameters being used
                        with st.expander("🔍 Debug Info - Scraping Strategy"):
                            st.markdown("""
**Multi-Attempt Strategy Used:**
1. First attempt: maxScrolls=10, resultsLimit up to 2x requested
2. Second attempt: maxScrolls=20, resultsLimit up to 3x requested  
3. Third attempt: maxScrolls=30, resultsLimit up to 5x requested

The scraper tries progressively more aggressive approaches until it gets the requested number of comments or exhausts all attempts.
                            """)
                            
                            with st.expander("📊 Statistics"):
                                st.info(f"""
**Fetch Summary:**
- Comments requested: {comment_limit}
- Comments fetched: {len(comments)}
- Success rate: {(len(comments)/comment_limit)*100:.1f}%

**Note:** If fewer comments are fetched than requested, it means:
- The post has fewer public comments available, OR
- Instagram/Apify rate limited the scraper, OR
- Some comments may be hidden/deleted

**Recommendation:** If you need more comments, try a different post.
                                """)
                        
                        with st.spinner("Classifying comments..."):
                            device = torch.device('cpu')
                            
                            # Prepare data
                            texts = [c.get("text", "") for c in comments]
                            usernames = [c.get("username", "Unknown") for c in comments]
                            
                            # Filter out empty texts
                            valid_indices = [i for i, t in enumerate(texts) if t.strip()]
                            texts = [texts[i] for i in valid_indices]
                            usernames = [usernames[i] for i in valid_indices]
                            
                            if not texts:
                                st.error("❌ No valid comments with text found.")
                            else:
                                st.info(f"📊 Fetched: {len(comments)} | Valid (with text): {len(texts)}")
                                
                                # Predict
                                predictions = predict_multiple_texts(texts, tokenizer, model, device=device)
                                
                                # Create results DataFrame with Username at the end
                                results_data = []
                                for username, comment_text, pred in zip(usernames, texts, predictions):
                                    results_data.append({
                                        'Username': username if username else 'Unknown',
                                        'Comment': comment_text[:150] + ('...' if len(comment_text) > 150 else ''),
                                        'Prediction': pred['prediction'],
                                        'Confidence': f"{pred['confidence']*100:.1f}%",
                                        '_full_text': comment_text,
                                        '_prediction_obj': pred
                                    })
                                
                                df_instagram = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith('_')} for r in results_data])
                                
                                st.markdown("---")
                                st.subheader("📊 Classification Results")
                                
                                # Summary statistics
                                suicidal_count = sum(1 for r in results_data if r['Prediction'] == 'Suicidal')
                                non_suicidal_count = len(results_data) - suicidal_count
                                
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric("Total Analyzed", len(results_data))
                                with col2:
                                    st.metric("Suicidal", suicidal_count)
                                with col3:
                                    st.metric("Non-Suicidal", non_suicidal_count)
                                
                                # Display table
                                st.dataframe(df_instagram, use_container_width=True, height=400)
                                
                                # SHAP Explanations
                                if show_shap:
                                    st.markdown("---")
                                    st.subheader("🔬 SHAP Explanations for Each Comment")
                                    st.markdown("Click to expand and see word-level importance for each comment")
                                    
                                    for idx, result_dict in enumerate(results_data):
                                        comment_text = result_dict['_full_text']
                                        pred = result_dict['_prediction_obj']
                                        username = result_dict['Username']
                                        
                                        with st.expander(
                                            f"**@{username}** - {pred['prediction']} ({pred['confidence']*100:.1f}%)"
                                        ):
                                            st.text(f"Comment: {comment_text}")
                                            
                                            try:
                                                # Validate text
                                                if not isinstance(comment_text, str) or not comment_text.strip():
                                                    st.warning("Cannot generate SHAP explanation for empty text")
                                                else:
                                                    with st.spinner("Generating SHAP explanation..."):
                                                        shap_html = generate_shap_explanation(
                                                            comment_text,
                                                            tokenizer,
                                                            model,
                                                            explainer
                                                        )
                                                        if shap_html:
                                                            st.components.v1.html(shap_html, height=250, scrolling=True)
                                            except Exception as e:
                                                st.error(f"❌ Error generating SHAP explanation: {str(e)[:100]}")
                                
                                # Download results
                                st.markdown("---")
                                csv = df_instagram.to_csv(index=False)
                                st.download_button(
                                    label="📥 Download Results as CSV",
                                    data=csv,
                                    file_name=f"instagram_classification_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv"
                                )
            
            except Exception as e:
                error_msg = str(e)
                st.error(f"❌ Error: {error_msg[:300]}")
                
                # Provide specific solutions based on error type
                if "authentication" in error_msg.lower() or "token" in error_msg.lower():
                    with st.expander("🔧 Authentication Error - Solution"):
                        st.markdown("""
Your APIFY_API_TOKEN is invalid or expired.

**Steps to fix:**
1. Go to https://console.apify.com/account/integrations
2. Copy your API token
3. Update the `.env` file with the new token:
   ```
   APIFY_API_TOKEN=your_new_token_here
   ```
4. Restart the Streamlit app
5. Try again
                        """)
                        
                elif "quota" in error_msg.lower() or "credits" in error_msg.lower():
                    with st.expander("💳 Quota Exceeded - Solution"):
                        st.markdown("""
Your Apify free tier monthly credits have been exceeded ($5 limit).

**Solutions:**
1. **Wait**: Credits reset on the 1st of next month
2. **Upgrade**: Get a paid plan at https://apify.com/pricing
3. **Use different post**: If you want to test with another Instagram post, you may hit limits again

**Current account info:**
- Plan: FREE tier
- Monthly credits: $5
- Status: May be exceeded
                        """)
                else:
                    with st.expander("🔍 Troubleshooting"):
                        st.markdown("""
**Common issues:**
- Invalid Instagram URL format
- Post is private or deleted
- Post has no comments
- Network/connection error
- Apify service temporarily unavailable

**Try:**
1. Verify the URL is correct
2. Make sure the post is public
3. Check your internet connection
4. Wait a moment and try again
                        """)


# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center'>
        <small>
        Suicide Expression Classification System | Powered by DistilBERT + SHAP
        </small>
    </div>
    """,
    unsafe_allow_html=True
)
