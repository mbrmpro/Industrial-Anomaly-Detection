import streamlit as st

from utils.config import *


def show_header():

    st.title(PROJECT_NAME)

    st.caption(PROJECT_SUBTITLE)

    st.divider()