import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
import google.generativeai as genai
import plotly.express as px
import plotly.graph_objects as go

# Load environment variables
load_dotenv()
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# Load XGBoost model only
@st.cache_resource
def load_model_components():
    try:
        model = joblib.load("xgboost_model.pkl")
        st.success("✅ XGBoost model loaded successfully!")
        return model
    except FileNotFoundError as e:
        st.error(f"Model file not found: {str(e)}. Please ensure 'xgboost_model.pkl' is in the current directory.")
        return None

model = load_model_components()

# Define categorical and numerical columns
categorical_cols = ['Gender', 'Region', 'Parental_Education', 'Internet_Access', 'Family_Support', 'School_Support', 'Activities']
numerical_cols = ['Age', 'Family_Income', 'Distance_from_School', 'Absences', 'Failures', 'Study_Time_Category', 'G1', 'G2', 'G3']

def prepare_input_data(age, family_income, distance_from_school, absences, failures, 
                      study_time_category, g1, g2, g3, gender, region, parental_education, 
                      internet_access, family_support, school_support, activities):
    """Convert user inputs to proper format for XGBoost model prediction"""
    
    # Create synthetic input dictionary matching the original format
    synthetic_input = {
        'Gender': gender,
        'Age': age,
        'Region': region,
        'Parental_Education': parental_education,
        'Family_Income': family_income,
        'Distance_from_School': distance_from_school,
        'Internet_Access': internet_access,
        'Absences': absences,
        'Failures': failures,
        'Study_Time_Category': study_time_category,
        'G1': g1,
        'G2': g2,
        'G3': g3,
        'Family_Support': family_support,
        'School_Support': school_support,
        'Activities': activities
    }
    
    # Convert to DataFrame
    synthetic_df = pd.DataFrame([synthetic_input])
    
    # Apply one-hot encoding to categorical columns
    synthetic_encoded = pd.get_dummies(synthetic_df, columns=categorical_cols, drop_first=True)
    
    # Define expected columns (based on typical training setup)
    expected_columns = [
        'Age', 'Family_Income', 'Distance_from_School', 'Absences', 'Failures', 'Study_Time_Category',
        'G1', 'G2', 'G3', 'Gender_Male', 'Region_Urban', 'Parental_Education_Primary', 
        'Parental_Education_Secondary', 'Internet_Access_Yes', 'Family_Support_Yes', 
        'School_Support_Yes', 'Activities_Yes'
    ]
    
    # Add missing columns with 0 values
    for col in expected_columns:
        if col not in synthetic_encoded.columns:
            synthetic_encoded[col] = 0
    
    # Select only the expected columns in the right order
    synthetic_processed = synthetic_encoded[expected_columns]
    
    return synthetic_processed

def get_low_risk_prompt():
    return """
    As an educational counselor, provide comprehensive recommendations for a LOW RISK student:
    
    **Maintenance & Growth Strategy:**
    1. **Academic Excellence Continuation**
       - Advanced learning opportunities and enrichment programs
       - Leadership roles in academic activities
       - Competitive exam preparation
    
    2. **Skill Development Focus**
       - Technology and digital literacy enhancement
       - Communication and presentation skills
       - Career exploration and internships
    
    3. **Peer Mentoring**
       - Mentor struggling students
       - Lead study groups and peer learning sessions
    
    4. **Future Planning**
       - Higher education guidance and scholarship applications
       - Career counseling and goal setting
       - Network building with professionals
    
    **For Female Students Specifically:**
    - STEM career guidance and role model exposure
    - Leadership development programs
    - Women empowerment workshops
    
    **Monitoring:** Monthly check-ins to ensure continued engagement and prevent complacency.
    """

def get_medium_risk_prompt():
    return """
    As an educational counselor, provide comprehensive recommendations for a MEDIUM RISK student:
    
    **Intervention Strategy:**
    1. **Academic Support Enhancement**
       - Personalized tutoring in weak subjects
       - Study skills and time management training
       - Regular progress monitoring
    
    2. **Attendance & Engagement**
       - Attendance tracking system
       - Engaging classroom activities
       - Counseling for attendance issues
    
    3. **Family & Community Engagement**
       - Parent-teacher collaboration
       - Home visit programs if needed
       - Community support network activation
    
    4. **Targeted Support Based on Issues:**
       - **Low Family Income → Scholarship Programs & Financial Aid**
       - **Low Parental Education → Mentor Support & Family Education**
       - **Rural/Distance Issues → Online Learning (NPTEL, DIKSHA, Khan Academy)**
       - **Low Attendance → Counselling & Awareness Programs**
       - **Low Grades → Tutoring & Peer Learning Groups**
    
    **For Female Students Specifically:**
    - Address gender-specific barriers (safety, family expectations)
    - Female mentor assignment
    - Flexible learning options considering household responsibilities
    
    **Monitoring:** Bi-weekly progress reviews and immediate intervention adjustments.
    """

def get_high_risk_prompt():
    return """
    As an educational counselor, provide comprehensive recommendations for a HIGH RISK student:
    
    **Immediate Crisis Intervention:**
    1. **Emergency Academic Support**
       - Intensive one-on-one tutoring
       - Remedial classes and catch-up programs
       - Alternative assessment methods
    
    2. **Psychosocial Support**
       - Counseling for personal/family issues
       - Mental health support
       - Stress management and coping strategies
    
    3. **Financial & Resource Support**
       - Emergency financial assistance
       - Free study materials and resources
       - Nutrition and health support
    
    4. **Comprehensive Intervention Based on Critical Issues:**
       - **Low Family Income → Emergency Scholarships, Fee Waivers, Job Training for Family**
       - **Low Parental Education → Adult Education Programs, Intensive Mentor Support**
       - **Rural/Distance Issues → Residential Support, Transportation Assistance, Online Learning Infrastructure**
       - **High Absences → Home Visits, Health Check-ups, Flexible Timing**
       - **Multiple Failures → Alternative Learning Paths, Vocational Training Options**
    
    5. **Alternative Pathways**
       - Vocational training options
       - Skill-based certification programs
       - Part-time study opportunities
    
    **For Female Students Specifically:**
    - Address safety and mobility concerns
    - Work with family on girl child education importance
    - Provide female-only support groups
    - Connect with women's empowerment NGOs
    
    **Monitoring:** Daily check-ins and comprehensive weekly reviews with multi-disciplinary team.
    
    **Crisis Response:** 24/7 support hotline and emergency intervention team.
    """

# Streamlit page configuration
st.set_page_config(
    page_title="Student Dropout Risk Predictor",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 5px;
        padding: 0.5rem 2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.markdown("""
<div class="main-header">
    <h1>🎓 Student Dropout Risk Prediction System</h1>
    <p>Advanced AI-powered analysis to identify at-risk students and provide actionable recommendations</p>
</div>
""", unsafe_allow_html=True)

if model is None:
    st.stop()

# Sidebar for input
st.sidebar.header("📋 Student Information")
st.sidebar.markdown("Please fill in all the student details below:")

# Demographic Information
st.sidebar.subheader("👤 Demographics")
age = st.sidebar.slider("Age", min_value=15, max_value=22, value=17, help="Student's current age")
gender = st.sidebar.selectbox("Gender", ["Female", "Male"])
region = st.sidebar.selectbox("Region", ["Rural", "Urban"], help="Student's residential area")

# Family Information
st.sidebar.subheader("👨‍👩‍👧‍👦 Family Background")
family_income = st.sidebar.number_input("Family Income (₹)", min_value=0, max_value=100000, value=15000, step=1000, 
                                       help="Annual family income in rupees")
parental_education = st.sidebar.selectbox("Parental Education Level", 
                                         ["Primary", "Secondary", "Higher Education"])
family_support = st.sidebar.selectbox("Family Support", ["Yes", "No"], 
                                     help="Does the family provide educational support?")

# School Information
st.sidebar.subheader("🏫 School & Academic")
distance_from_school = st.sidebar.slider("Distance from School (km)", min_value=0.0, max_value=50.0, value=5.0, step=0.5)
absences = st.sidebar.number_input("Number of Absences", min_value=0, max_value=100, value=5, 
                                  help="Total absences in current academic year")
failures = st.sidebar.number_input("Previous Failures", min_value=0, max_value=10, value=0, 
                                  help="Number of past class failures")

# Study Information
st.sidebar.subheader("📚 Study Patterns")
study_time_category = st.sidebar.selectbox("Study Time Category", 
                                          [1, 2, 3, 4], 
                                          format_func=lambda x: f"Level {x} ({'<2 hours' if x==1 else '2-5 hours' if x==2 else '5-10 hours' if x==3 else '>10 hours'} per week)",
                                          index=1)

# Grades
st.sidebar.subheader("📊 Academic Performance")
g1 = st.sidebar.slider("First Period Grade (G1)", min_value=0, max_value=20, value=10, help="Grade in first period")
g2 = st.sidebar.slider("Second Period Grade (G2)", min_value=0, max_value=20, value=10, help="Grade in second period")
g3 = st.sidebar.slider("Final Grade (G3)", min_value=0, max_value=20, value=10, help="Final grade")

# Support & Activities
st.sidebar.subheader("🎯 Support & Activities")
internet_access = st.sidebar.selectbox("Internet Access", ["Yes", "No"])
school_support = st.sidebar.selectbox("School Support", ["Yes", "No"], 
                                     help="Does school provide extra educational support?")
activities = st.sidebar.selectbox("Extracurricular Activities", ["Yes", "No"], 
                                 help="Participates in extracurricular activities?")

# Main content area
col1, col2 = st.columns([2, 1])

with col1:
    st.subheader("📊 Student Profile Summary")
    
    # Create summary metrics
    col_a, col_b, col_c, col_d = st.columns(4)
    
    with col_a:
        st.metric("Age", f"{age} years")
        st.metric("Family Income", f"₹{family_income:,}")
    
    with col_b:
        st.metric("Distance to School", f"{distance_from_school} km")
        st.metric("Absences", absences)
    
    with col_c:
        st.metric("Previous Failures", failures)
        avg_grade = round((g1 + g2 + g3) / 3, 1)
        st.metric("Average Grade", f"{avg_grade}/20")
    
    with col_d:
        support_score = sum([
            1 if family_support == "Yes" else 0,
            1 if school_support == "Yes" else 0,
            1 if internet_access == "Yes" else 0,
            1 if activities == "Yes" else 0
        ])
        st.metric("Support Score", f"{support_score}/4")

with col2:
    st.subheader("🔍 Quick Stats")
    st.info(f"**Gender:** {gender}")
    st.info(f"**Region:** {region}")
    st.info(f"**Parental Education:** {parental_education}")
    st.info(f"**Study Time:** Level {study_time_category}")

# Prediction section
st.markdown("---")
col1, col2, col3 = st.columns([1, 2, 1])

with col2:
    if st.button("🔮 Predict Dropout Risk", type="primary", use_container_width=True):
        # Prepare input data
        test_input = prepare_input_data(
            age, family_income, distance_from_school, absences, failures,
            study_time_category, g1, g2, g3, gender, region, parental_education,
            internet_access, family_support, school_support, activities
        )
        
        try:
            # Make prediction using the processed input
            pred_class_encoded = model.predict(test_input)[0]
            pred_proba = model.predict_proba(test_input)[0]
            
            # Map encoded predictions to risk levels (assuming 0=Low, 1=Medium, 2=High)
            risk_mapping = {0: "Low Risk", 1: "Medium Risk", 2: "High Risk"}
            pred_class = risk_mapping.get(pred_class_encoded, f"Risk Level {pred_class_encoded}")
            confidence = max(pred_proba) * 100
            
            # Display prediction results
            st.markdown("---")
            st.subheader("🎯 Prediction Results")
            
            # Risk level with color coding
            if pred_class == "High Risk":
                st.error(f"🚨 **High Dropout Risk** (Confidence: {confidence:.1f}%)")
                risk_color = "red"
            elif pred_class == "Medium Risk":
                st.warning(f"⚠️ **Medium Dropout Risk** (Confidence: {confidence:.1f}%)")
                risk_color = "orange"
            else:
                st.success(f"✅ **Low Dropout Risk** (Confidence: {confidence:.1f}%)")
                risk_color = "green"
            
            # Debug information to help understand model output
            with st.expander("🔍 Debug Information"):
                st.write("**Processed Input Shape:**", test_input.shape)
                st.write("**Input Features:**", list(test_input.columns))
                st.write("**Predicted Class (encoded):**", pred_class_encoded)
                st.write("**Prediction Probabilities:**", pred_proba)
                st.write("**Sample Input Values:**")
                st.dataframe(test_input.head(), use_container_width=True)
            
            # Show all probabilities
            st.subheader("📊 Risk Probability Breakdown")
            
            # Create risk level names for display
            risk_levels = ["Low Risk", "Medium Risk", "High Risk"]
            prob_df = pd.DataFrame({
                'Risk Level': risk_levels[:len(pred_proba)],
                'Probability': pred_proba
            }).sort_values('Probability', ascending=False)
            
            fig = px.bar(prob_df, x='Risk Level', y='Probability', 
                        title='Dropout Risk Probabilities',
                        color='Probability', color_continuous_scale='RdYlGn_r')
            st.plotly_chart(fig, use_container_width=True)
            
            # AI-Generated Recommendations based on risk level
            if GEMINI_API_KEY:
                st.markdown("---")
                st.subheader("🤖 AI-Generated Personalized Recommendations")
                
                with st.spinner("Generating personalized recommendations..."):
                    # Select appropriate prompt based on risk level
                    if pred_class == "Low Risk":
                        base_prompt = get_low_risk_prompt()
                    elif pred_class == "Medium Risk":
                        base_prompt = get_medium_risk_prompt()
                    else:  # High Risk
                        base_prompt = get_high_risk_prompt()
                    
                    # Add student-specific context
                    context = f"""
                    **Student Context:**
                    - Age: {age}, Gender: {gender}, Region: {region}
                    - Family Income: ₹{family_income}, Parental Education: {parental_education}
                    - Average Grade: {avg_grade}/20, Absences: {absences}, Failures: {failures}
                    - Family Support: {family_support}, School Support: {school_support}
                    - Internet Access: {internet_access}, Activities: {activities}
                    - Distance from School: {distance_from_school} km
                    
                    Based on this profile and the {pred_class} prediction, provide specific, actionable recommendations:
                    
                    {base_prompt}
                    
                    **Make recommendations culturally sensitive and practical for Indian educational context.**
                    """
                    
                    try:
                        model_gen = genai.GenerativeModel("gemini-2.0-flash")
                        response = model_gen.generate_content(context)
                        st.markdown(response.text)
                    except Exception as e:
                        st.error(f"Failed to generate AI recommendations: {str(e)}")
            else:
                st.warning("⚠️ Gemini API key not found. AI recommendations are not available.")
                
                # Show basic recommendations based on risk level
                st.subheader("📋 Basic Recommendations")
                if pred_class == "Low Risk":
                    st.success("""
                    **Low Risk - Maintenance Strategy:**
                    - Continue current academic excellence
                    - Explore advanced learning opportunities  
                    - Consider peer mentoring roles
                    - Focus on career planning and skill development
                    """)
                elif pred_class == "Medium Risk":
                    st.warning("""
                    **Medium Risk - Intervention Needed:**
                    - Enhance academic support and tutoring
                    - Address attendance issues through counseling
                    - Strengthen family-school communication
                    - Provide targeted support based on specific challenges
                    """)
                else:
                    st.error("""
                    **High Risk - Immediate Action Required:**
                    - Intensive academic and psychosocial support
                    - Emergency financial/resource assistance
                    - Daily monitoring and intervention
                    - Consider alternative learning pathways
                    - Involve multiple stakeholders immediately
                    """)
            
        except Exception as e:
            st.error(f"Prediction failed: {str(e)}")
            st.write("Please check if the XGBoost model file is compatible and properly trained.")

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem;">
    <p>Built with ❤️ using Streamlit, XGBoost, and Gemini AI</p>
    <p><small>This tool is designed to support educational decision-making and should be used alongside professional judgment.</small></p>
</div>
""", unsafe_allow_html=True)