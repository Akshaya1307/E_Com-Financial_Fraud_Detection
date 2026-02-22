import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from datetime import datetime
import time

# -----------------------
# PAGE CONFIGURATION
# -----------------------
st.set_page_config(
    page_title="FraudShield AI - E-Commerce Fraud Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    /* Main container styling */
    .main-header {
        background: linear-gradient(90deg, #1a1a2e 0%, #16213e 100%);
        padding: 2rem;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .main-title {
        color: white;
        font-size: 2.5rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .main-subtitle {
        color: #a8dadc;
        font-size: 1.1rem;
        text-align: center;
        margin-bottom: 0;
    }
    
    /* Metric cards styling */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.05);
        border-left: 4px solid #00b4d8;
        margin-bottom: 1rem;
    }
    .metric-label {
        color: #666;
        font-size: 0.9rem;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        color: #1a1a2e;
        font-size: 2rem;
        font-weight: 700;
    }
    
    /* Risk badges */
    .risk-low {
        background: linear-gradient(135deg, #06d6a0 0%, #05b58b 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .risk-medium {
        background: linear-gradient(135deg, #ffd166 0%, #ffb347 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    .risk-high {
        background: linear-gradient(135deg, #ef476f 0%, #d33f63 100%);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: 600;
        display: inline-block;
    }
    
    /* Status colors */
    .fraud-text {
        color: #ef476f;
        font-weight: 600;
    }
    .legit-text {
        color: #06d6a0;
        font-weight: 600;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------
# LOAD MODEL FILES
# -----------------------
@st.cache_resource
def load_models():
    """Load all model files with error handling"""
    try:
        lgb_model = joblib.load("lgb_model.pkl")
        xgb_model = joblib.load("xgb_model.pkl")
        threshold = joblib.load("threshold.pkl")
        feature_columns = joblib.load("feature_columns.pkl")
        return lgb_model, xgb_model, threshold, feature_columns
    except FileNotFoundError as e:
        st.error(f"❌ Missing model file: {e}")
        st.info("Please ensure all model files are in the correct directory.")
        return None, None, None, None
    except Exception as e:
        st.error(f"❌ Error loading models: {e}")
        return None, None, None, None

lgb_model, xgb_model, threshold, feature_columns = load_models()

# -----------------------
# INITIALIZE SESSION STATE
# -----------------------
if 'prediction_history' not in st.session_state:
    st.session_state.prediction_history = []
if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0
if 'fraud_detected' not in st.session_state:
    st.session_state.fraud_detected = 0
if 'page' not in st.session_state:
    st.session_state['page'] = "🔍 Single Transaction"

# -----------------------
# HELPER FUNCTIONS
# -----------------------
def encode_input(df):
    """
    Encode categorical columns using one-hot encoding (get_dummies)
    This matches the training notebook's preprocessing exactly
    """
    categorical_cols = [
        "Payment Method",
        "Product Category",
        "Customer Location",
        "Device Used"
    ]
    
    # Make a copy to avoid modifying original
    df_encoded = df.copy()
    
    # Apply one-hot encoding (same as pd.get_dummies in notebook)
    df_encoded = pd.get_dummies(df_encoded, columns=categorical_cols, drop_first=True)
    
    return df_encoded

def predict_transaction(input_df):
    """
    Make prediction using ensemble model with proper feature alignment
    """
    # Step 1: One-hot encode categorical variables (matches training)
    encoded_df = encode_input(input_df)
    
    # Step 2: Create a dataframe with all training features (initialized to 0)
    full_df = pd.DataFrame(0, index=encoded_df.index, columns=feature_columns)
    
    # Step 3: Update with values from encoded dataframe where columns match
    for col in encoded_df.columns:
        if col in full_df.columns:
            full_df[col] = encoded_df[col]
    
    # Step 4: Ensure correct column order
    full_df = full_df[feature_columns]
    
    # Step 5: Get predictions from both models
    lgb_prob = lgb_model.predict_proba(full_df)[:, 1]
    xgb_prob = xgb_model.predict_proba(full_df)[:, 1]
    
    # Step 6: Weighted ensemble (60% LightGBM, 40% XGBoost)
    final_prob = (0.6 * lgb_prob) + (0.4 * xgb_prob)
    prediction = (final_prob >= threshold).astype(int)
    
    return final_prob[0], prediction[0]

def get_risk_level(probability):
    """Get risk level based on probability"""
    if probability < 0.3:
        return "LOW", "risk-low"
    elif probability < 0.6:
        return "MEDIUM", "risk-medium"
    else:
        return "HIGH", "risk-high"

def create_gauge_chart(probability, threshold_val):
    """Create a gauge chart for risk probability"""
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=probability * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Fraud Risk Score", 'font': {'size': 20}},
        delta={'reference': threshold_val * 100, 'increasing': {'color': "red"}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "darkblue"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 30], 'color': '#06d6a0'},
                {'range': [30, 60], 'color': '#ffd166'},
                {'range': [60, 100], 'color': '#ef476f'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': threshold_val * 100
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font={'color': "#1a1a2e", 'family': "Arial"}
    )
    
    return fig

# -----------------------
# MAIN APP - STOP IF MODELS NOT LOADED
# -----------------------
if lgb_model is None or xgb_model is None:
    st.error("⚠️ Models could not be loaded. Please check your model files.")
    st.stop()

# Header
st.markdown("""
<div class="main-header">
    <h1 class="main-title">🛡️ FraudShield AI</h1>
    <p class="main-subtitle">Advanced E-Commerce Fraud Detection System | Powered by Ensemble ML</p>
</div>
""", unsafe_allow_html=True)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/null/security-checked--v1.png", width=80)
    st.title("Navigation")
    
    page = st.radio(
        "Select Page",
        ["🔍 Single Transaction", "📊 Batch Processing", "📈 Analytics Dashboard", "⚙️ Settings"],
        key="navigation"
    )
    
    st.markdown("---")
    
    # Model info
    st.subheader("📊 Model Info")
    st.info(f"""
    - **Ensemble**: LightGBM (60%) + XGBoost (40%)
    - **Threshold**: {threshold:.3f}
    - **ROC AUC**: 0.8166
    - **Precision (Fraud)**: 0.42
    - **Recall (Fraud)**: 0.39
    - **Features**: {len(feature_columns)}
    """)
    
    # Session stats
    st.subheader("📈 Session Stats")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Checks", st.session_state.total_predictions)
    with col2:
        fraud_rate = (st.session_state.fraud_detected / st.session_state.total_predictions * 100) if st.session_state.total_predictions > 0 else 0
        st.metric("Fraud Rate", f"{fraud_rate:.1f}%")

# -----------------------
# PAGE 1: SINGLE TRANSACTION
# -----------------------
if page == "🔍 Single Transaction":
    st.header("🔍 Single Transaction Analysis")
    st.markdown("Enter transaction details below for real-time fraud detection.")
    
    # Create two columns for input
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("💰 Transaction Details")
        amount = st.number_input("Transaction Amount ($)", min_value=0.0, max_value=10000.0, value=150.0, step=10.0)
        quantity = st.number_input("Quantity", min_value=1, max_value=100, value=1, step=1)
        payment_method = st.selectbox(
            "Payment Method", 
            ["Credit Card", "Debit Card", "UPI", "Bank Transfer"]
        )
        product_category = st.selectbox(
            "Product Category", 
            ["Electronics", "Clothing", "Home", "Beauty"]
        )
    
    with col2:
        st.subheader("👤 Customer Details")
        customer_age = st.number_input("Customer Age", min_value=18, max_value=100, value=35)
        account_age = st.number_input("Account Age (Days)", min_value=0, max_value=3650, value=90)
        customer_location = st.selectbox(
            "Customer Location", 
            ["Urban", "Semi-Urban", "Rural"]
        )
        device_used = st.selectbox(
            "Device Used", 
            ["Mobile", "Desktop", "Tablet"]
        )
        transaction_hour = st.slider("Transaction Hour", 0, 23, 14)
    
    # Predict button
    if st.button("🚀 Analyze Transaction", type="primary", use_container_width=True):
        with st.spinner("Analyzing transaction patterns..."):
            # Create input dataframe
            input_data = pd.DataFrame({
                "Transaction Amount": [amount],
                "Quantity": [quantity],
                "Customer Age": [customer_age],
                "Account Age Days": [account_age],
                "Transaction Hour": [transaction_hour],
                "Payment Method": [payment_method],
                "Product Category": [product_category],
                "Customer Location": [customer_location],
                "Device Used": [device_used]
            })
            
            # Make prediction
            probability, prediction = predict_transaction(input_data)
            
            # Update session stats
            st.session_state.total_predictions += 1
            if prediction == 1:
                st.session_state.fraud_detected += 1
            
            # Store in history
            st.session_state.prediction_history.append({
                'timestamp': datetime.now(),
                'risk_score': probability,
                'is_fraud': prediction,
                'amount': amount,
                'transaction_id': f"TXN{len(st.session_state.prediction_history):06d}"
            })
            
            # Display results
            st.markdown("---")
            st.header("📊 Analysis Results")
            
            # Results layout
            col_res1, col_res2, col_res3 = st.columns([1, 2, 1])
            
            with col_res1:
                st.markdown("### Risk Assessment")
                risk_level, risk_class = get_risk_level(probability)
                st.markdown(f'<div class="{risk_class}">⚠️ {risk_level} RISK</div>', unsafe_allow_html=True)
                st.markdown(f"**Fraud Probability:** {probability:.2%}")
                st.markdown(f"**Threshold:** {threshold:.2%}")
                
                if prediction == 1:
                    st.error("⚠️ This transaction has been flagged as potentially fraudulent!")
                else:
                    st.success("✅ This transaction appears to be legitimate.")
            
            with col_res2:
                # Gauge chart
                gauge_fig = create_gauge_chart(probability, threshold)
                st.plotly_chart(gauge_fig, use_container_width=True)
            
            with col_res3:
                st.markdown("### Risk Factors")
                risk_factors = []
                
                if amount > 500:
                    risk_factors.append(("💰 High Amount", "High value transactions are riskier"))
                if account_age < 30:
                    risk_factors.append(("🆕 New Account", "Accounts under 30 days have higher fraud rate"))
                if transaction_hour < 6 or transaction_hour > 22:
                    risk_factors.append(("🌙 Odd Hours", "Transactions during off-hours are suspicious"))
                if quantity > 5:
                    risk_factors.append(("📦 Bulk Order", "Large quantities may indicate fraud"))
                if customer_location == "Rural" and amount > 300:
                    risk_factors.append(("📍 Location Mismatch", "High amount from rural area"))
                
                if risk_factors:
                    for factor, desc in risk_factors:
                        st.warning(f"**{factor}**: {desc}")
                else:
                    st.success("No significant risk factors detected")

# -----------------------
# PAGE 2: BATCH PROCESSING
# -----------------------
elif page == "📊 Batch Processing":
    st.header("📊 Batch Transaction Processing")
    st.markdown("Upload a CSV file with multiple transactions for bulk analysis.")
    
    # Show expected CSV format
    with st.expander("📋 View Expected CSV Format"):
        st.markdown("""
        Your CSV should contain the following columns:
        - `Transaction Amount` (numeric)
        - `Quantity` (numeric)
        - `Customer Age` (numeric)
        - `Account Age Days` (numeric)
        - `Transaction Hour` (numeric, 0-23)
        - `Payment Method` (Credit Card, Debit Card, UPI, Bank Transfer)
        - `Product Category` (Electronics, Clothing, Home, Beauty)
        - `Customer Location` (Urban, Semi-Urban, Rural)
        - `Device Used` (Mobile, Desktop, Tablet)
        """)
        
        # Sample data
        sample_df = pd.DataFrame({
            'Transaction Amount': [150.00, 750.00, 45.50],
            'Quantity': [1, 3, 2],
            'Customer Age': [35, 28, 52],
            'Account Age Days': [90, 15, 365],
            'Transaction Hour': [14, 3, 10],
            'Payment Method': ['Credit Card', 'UPI', 'Debit Card'],
            'Product Category': ['Electronics', 'Clothing', 'Beauty'],
            'Customer Location': ['Urban', 'Semi-Urban', 'Rural'],
            'Device Used': ['Desktop', 'Mobile', 'Tablet']
        })
        st.dataframe(sample_df)
    
    # File upload
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:")
            st.dataframe(df.head())
            
            # Check for required columns
            required_cols = [
                "Transaction Amount", "Quantity", "Customer Age", 
                "Account Age Days", "Transaction Hour", "Payment Method",
                "Product Category", "Customer Location", "Device Used"
            ]
            missing_cols = [col for col in required_cols if col not in df.columns]
            
            if missing_cols:
                st.error(f"Missing required columns: {', '.join(missing_cols)}")
            else:
                if st.button("🚀 Process Batch", type="primary"):
                    with st.spinner(f"Processing {len(df)} transactions..."):
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        results = []
                        
                        for i, row in df.iterrows():
                            # Create input row
                            input_row = pd.DataFrame([row.to_dict()])
                            
                            # Make prediction
                            probability, prediction = predict_transaction(input_row)
                            
                            results.append({
                                'Row': i + 1,
                                'Risk Score': f"{probability:.4f}",
                                'Prediction': 'FRAUD' if prediction == 1 else 'LEGIT',
                                'Amount': f"${row.get('Transaction Amount', 0):.2f}"
                            })
                            
                            # Add to session state history
                            st.session_state.prediction_history.append({
                                'timestamp': datetime.now(),
                                'risk_score': probability,
                                'is_fraud': prediction,
                                'amount': row.get('Transaction Amount', 100),
                                'transaction_id': f"BATCH{len(st.session_state.prediction_history):06d}"
                            })
                            
                            # Update session stats
                            st.session_state.total_predictions += 1
                            if prediction == 1:
                                st.session_state.fraud_detected += 1
                            
                            # Update progress
                            progress_bar.progress((i + 1) / len(df))
                            status_text.text(f"Processed {i + 1}/{len(df)} transactions")
                        
                        # Display results
                        st.success(f"✅ Successfully processed {len(df)} transactions!")
                        
                        # Results dataframe
                        results_df = pd.DataFrame(results)
                        st.dataframe(results_df)
                        
                        # Summary metrics
                        fraud_count = len([r for r in results if r['Prediction'] == 'FRAUD'])
                        
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Processed", len(df))
                        with col2:
                            fraud_pct = (fraud_count/len(df)*100) if len(df) > 0 else 0
                            st.metric("Fraud Detected", fraud_count, delta=f"{fraud_pct:.1f}%")
                        with col3:
                            avg_risk = pd.to_numeric(results_df['Risk Score']).mean()
                            st.metric("Avg Risk Score", f"{avg_risk:.3f}")
                        
                        # Download results
                        csv = results_df.to_csv(index=False)
                        st.download_button(
                            label="📥 Download Results CSV",
                            data=csv,
                            file_name=f"fraud_detection_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                            mime="text/csv"
                        )
                        
        except Exception as e:
            st.error(f"Error processing file: {str(e)}")

# -----------------------
# PAGE 3: ANALYTICS DASHBOARD
# -----------------------
elif page == "📈 Analytics Dashboard":
    st.header("📈 Analytics Dashboard")
    
    if len(st.session_state.prediction_history) > 0:
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Transactions", 
                st.session_state.total_predictions
            )
        
        with col2:
            fraud_count = st.session_state.fraud_detected
            st.metric(
                "Fraud Detected", 
                fraud_count
            )
        
        with col3:
            fraud_rate = (st.session_state.fraud_detected / st.session_state.total_predictions * 100) if st.session_state.total_predictions > 0 else 0
            st.metric(
                "Fraud Rate", 
                f"{fraud_rate:.1f}%"
            )
        
        with col4:
            avg_risk = np.mean([h['risk_score'] for h in st.session_state.prediction_history])
            st.metric(
                "Avg Risk Score", 
                f"{avg_risk:.3f}"
            )
        
        # Create two columns for charts
        chart_col1, chart_col2 = st.columns(2)
        
        with chart_col1:
            # Risk Score Distribution
            risk_scores = [h['risk_score'] for h in st.session_state.prediction_history]
            fraud_scores = [h['risk_score'] for h in st.session_state.prediction_history if h['is_fraud'] == 1]
            legit_scores = [h['risk_score'] for h in st.session_state.prediction_history if h['is_fraud'] == 0]
            
            fig_dist = go.Figure()
            fig_dist.add_trace(go.Histogram(
                x=legit_scores,
                name='Legitimate',
                marker_color='#06d6a0',
                opacity=0.7,
                nbinsx=20
            ))
            fig_dist.add_trace(go.Histogram(
                x=fraud_scores,
                name='Fraudulent',
                marker_color='#ef476f',
                opacity=0.7,
                nbinsx=20
            ))
            fig_dist.add_vline(x=threshold, line_dash="dash", line_color="orange", 
                             annotation_text=f"Threshold: {threshold:.2f}")
            
            fig_dist.update_layout(
                title="Risk Score Distribution",
                xaxis_title="Risk Score",
                yaxis_title="Count",
                height=350,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                barmode='overlay'
            )
            st.plotly_chart(fig_dist, use_container_width=True)
        
        with chart_col2:
            # Fraud vs Legit Pie Chart
            fraud_count = sum(1 for h in st.session_state.prediction_history if h['is_fraud'] == 1)
            legit_count = len(st.session_state.prediction_history) - fraud_count
            
            fig_pie = go.Figure(data=[go.Pie(
                labels=['Legitimate', 'Fraudulent'],
                values=[legit_count, fraud_count],
                marker_colors=['#06d6a0', '#ef476f'],
                textinfo='label+percent',
                hole=0.4,
                pull=[0, 0.1] if fraud_count > 0 else [0, 0]
            )])
            
            fig_pie.update_layout(
                title="Transaction Classification",
                height=350,
                margin=dict(l=20, r=20, t=50, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                showlegend=False
            )
            st.plotly_chart(fig_pie, use_container_width=True)
        
        # Transaction Timeline
        st.subheader("📊 Recent Transactions Timeline")
        
        # Prepare timeline data
        timeline_df = pd.DataFrame(st.session_state.prediction_history[-20:])
        
        fig_timeline = px.scatter(
            timeline_df,
            x='timestamp',
            y='risk_score',
            color='is_fraud',
            size='amount',
            hover_data=['transaction_id'],
            title="Recent Transactions Risk Timeline",
            labels={'risk_score': 'Risk Score', 'timestamp': 'Time', 'is_fraud': 'Status'},
            color_discrete_map={0: '#06d6a0', 1: '#ef476f'}
        )
        fig_timeline.add_hline(y=threshold, line_dash="dash", line_color="orange", 
                             annotation_text=f"Threshold: {threshold:.2f}")
        
        fig_timeline.update_layout(
            height=400,
            margin=dict(l=20, r=20, t=50, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_timeline, use_container_width=True)
        
        # Recent Transactions Table
        st.subheader("📋 Recent Transactions")
        recent_df = pd.DataFrame(st.session_state.prediction_history[-10:][::-1])
        recent_df['risk_score'] = recent_df['risk_score'].round(4)
        recent_df['status'] = recent_df['is_fraud'].apply(lambda x: '🚨 FRAUD' if x == 1 else '✅ LEGIT')
        recent_df['amount'] = recent_df['amount'].apply(lambda x: f"${x:.2f}")
        
        display_df = recent_df[['timestamp', 'transaction_id', 'amount', 'risk_score', 'status']]
        display_df.columns = ['Time', 'Transaction ID', 'Amount', 'Risk Score', 'Status']
        
        # Style the dataframe
        def color_status(val):
            if 'FRAUD' in val:
                return 'background-color: #ef476f20; color: #ef476f'
            return 'background-color: #06d6a020; color: #06d6a0'
        
        styled_df = display_df.style.map(color_status, subset=['Status'])
        st.dataframe(styled_df, use_container_width=True)
        
        # Download analytics
        if st.button("📥 Download Analytics Report"):
            report_df = pd.DataFrame(st.session_state.prediction_history)
            csv = report_df.to_csv(index=False)
            st.download_button(
                label="Click to Download",
                data=csv,
                file_name=f"fraud_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    else:
        st.info("👋 No transaction history yet. Please run some predictions first!")
        
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("🔍 Go to Single Transaction", use_container_width=True):
                st.session_state['page'] = "🔍 Single Transaction"
                st.rerun()
        with col2:
            if st.button("📊 Go to Batch Processing", use_container_width=True):
                st.session_state['page'] = "📊 Batch Processing"
                st.rerun()
        with col3:
            if st.button("🎲 Load Demo Data", use_container_width=True):
                # Generate sample demo data
                with st.spinner("Generating demo transactions..."):
                    for i in range(25):
                        risk_score = np.random.beta(2, 5)
                        is_fraud = 1 if risk_score > threshold else 0
                        
                        st.session_state.prediction_history.append({
                            'timestamp': datetime.now(),
                            'risk_score': risk_score,
                            'is_fraud': is_fraud,
                            'amount': np.random.uniform(10, 500),
                            'transaction_id': f"DEMO{i+1:04d}"
                        })
                        st.session_state.total_predictions += 1
                        if is_fraud:
                            st.session_state.fraud_detected += 1
                st.rerun()

# -----------------------
# PAGE 4: SETTINGS
# -----------------------
else:
    st.header("⚙️ Settings & Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Model Configuration")
        st.markdown(f"""
        - **LightGBM Weight**: 60%
        - **XGBoost Weight**: 40%
        - **Current Threshold**: {threshold:.3f}
        - **ROC AUC**: 0.8166
        - **Precision (Fraud)**: 0.42
        - **Recall (Fraud)**: 0.39
        - **F1-Score (Fraud)**: 0.40
        """)
        
        st.markdown("### Model Performance (from notebook)")
        st.markdown("""
        ```
        Classification Report:
                      precision    recall  f1-score
                   0       0.97      0.97      0.97
                   1       0.42      0.39      0.40
        
        Confusion Matrix:
        [[271853   7970]
         [  9065   5703]]
        ```
        """)
        
        st.markdown("### Alert Settings")
        email_alerts = st.checkbox("Email Alerts", value=True)
        slack_alerts = st.checkbox("Slack Notifications", value=False)
        sms_alerts = st.checkbox("SMS Alerts", value=False)
    
    with col2:
        st.subheader("System Info")
        st.markdown(f"""
        - **Model Files**:
          - `lgb_model.pkl`
          - `xgb_model.pkl`
          - `threshold.pkl`
          - `feature_columns.pkl`
        - **Number of Features**: {len(feature_columns)}
        - **Ensemble Type**: Weighted Average (60/40)
        - **Encoding Method**: One-Hot Encoding (get_dummies)
        """)
        
        # Show feature columns (first 10)
        st.markdown("### Feature Columns Sample")
        st.code(", ".join(feature_columns[:10]) + "...")
        
        # Reset session button
        if st.button("🔄 Reset Session Data", type="secondary"):
            st.session_state.prediction_history = []
            st.session_state.total_predictions = 0
            st.session_state.fraud_detected = 0
            st.success("✅ Session data reset!")
            st.rerun()

# -----------------------
# FOOTER
# -----------------------
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>🛡️ FraudShield AI • Powered by LightGBM + XGBoost Ensemble • v2.0.0</p>
        <p style="font-size: 0.8rem;">© 2026 FraudShield AI. All rights reserved.</p>
    </div>
    """,
    unsafe_allow_html=True
)