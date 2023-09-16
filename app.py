import numpy as np
import pickle
import streamlit as st
from xgboost import XGBClassifier

st.set_page_config(
    page_title="Lab 01 App",
    page_icon="🌻",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'About': '''Quách Xuân Nam - 20020541 - IUH\n
        https://www.facebook.com/20020541.nam'''
    }
)
model = None
with open('best_model.pkl', 'rb') as f:
    model = pickle.load(f)

def predict(features):
    decode = {0: '1', 1: '2', 2: '3'}
    X_new = np.array([features])#.reshape(1,-1)
    prediction = model.predict(X_new)
    return decode[prediction[0]]
    
def input_features():
    feature_1 = st.sidebar.number_input('Feature 01', min_value=0.0, max_value=50.0, value=5.4, step=0.1)
    feature_2 = st.sidebar.number_input('Feature 02', min_value=0.0, max_value=50.0, value=3.4, step=0.1)
    feature_3 = st.sidebar.number_input('Feature 03', min_value=0.0, max_value=50.0, value=1.3, step=0.1)
    feature_4 = st.sidebar.number_input('Feature 04', min_value=0.0, max_value=50.0, value=0.2, step=0.1)
    feature_5 = st.sidebar.number_input('Feature 05', min_value=0.0, max_value=50.0, value=0.2, step=0.1)
    feature_6 = st.sidebar.number_input('Feature 06', min_value=0.0, max_value=50.0, value=0.2, step=0.1)
    feature_7 = st.sidebar.number_input('Feature 07', min_value=0.0, max_value=50.0, value=0.2, step=0.1)

    return [feature_1, feature_2, feature_3, feature_4, feature_5, feature_6, feature_7]

def main():
    st.markdown('<h1 style="text-align: center;">Classification</h1>', unsafe_allow_html=True)
    st.markdown('---')
    data = input_features()
    pred = predict(data)
    st.markdown('<h1 style="text-align: center;">Prediction</h1>', unsafe_allow_html=True)
    st.markdown(f'<h1 style="text-align: center; border: 5px solid green; ">{pred}</h1>', unsafe_allow_html=True)
    st.caption('Modify by :blue[qxnam]')
if __name__ == '__main__':
    main()