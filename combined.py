import streamlit as st
from pred_AI import app as app1
from nlp import app as app2
from statistic import app as app3


st.title("ULearn Dashboard")
st.header("Student Activity and Performance Analysis")

PAGES = {
    "Artififcial Intelligence": app1,
    "Natural Language Processing": app2,
    "Statistics and Probablity": app3
}

st.sidebar.title("Navigation")
selection = st.sidebar.radio("Go to", list(PAGES.keys()))

page = PAGES[selection]
page()
