import pandas as pd
import numpy as np
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import loguniform
from scipy.stats import randint as sp_randint
from scipy.stats import uniform
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import streamlit as st
from sklearn.preprocessing import StandardScaler
from joblib import dump, load
import os


def warn(*args, **kwargs):
    pass


import warnings

warnings.warn = warn
# warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore")


def load_data():
    # return pd.read_csv('data/children_anemia_cleaned_phase3.csv')
    return pd.read_csv("../data/children_anemia_cleaned.csv")


st.title("Anemia Level Prediction")

# Input features
with st.form(key="input_form"):

    age_group = st.slider("Mother's current age", min_value=10, max_value=80, value=5)
    past_births = st.slider(
        "Births in last five years", min_value=0, max_value=10, value=2
    )
    # first_birth_age = st.slider('First Birth Age', min_value=15, max_value=50, value=25)
    first_birth_age = st.slider(
        "Age of the mother at 1st birth", min_value=9, max_value=age_group, value=25
    )
    adj_hemo_alt = st.slider("Hemoglobin level", min_value=10, max_value=300, value=10)
    adj_hemo_altsmoke = st.slider(
        "Adjusted hemoglobin level due to smoking",
        min_value=10,
        max_value=300,
        value=10,
    )
    residence_type = st.selectbox("Residence Type", ("Rural", "Urban", "Unknown"))
    if residence_type == "Rural":
        residence_rural = True
        residence_urban = False
    elif residence_type == "Urban":
        residence_rural = False
        residence_urban = True
    else:  # Handle "Unknown" case if necessary, or assume both false
        residence_rural = False
        residence_urban = False

    # residence_rural_choice = st.radio("Rural Residence", ("Yes", "No", "Unknown"))
    # residence_rural = residence_rural_choice == "Yes"
    # residence_urban_choice = st.radio("Urban Residence", ("Yes", "No", "Unknown"))
    # residence_urban = residence_urban_choice == "Yes"
    wealth_level = st.radio(
        "Wealth level of the mother",
        ("Poorest", "Poorer", "Middle", "Richer", "Richest"),
    )
    wealth_middle = False
    wealth_poorer = False
    wealth_poorest = False
    wealth_richer = False
    wealth_richest = False
    if wealth_level == "Middle":
        wealth_middle = True
        wealth_poorer = False
        wealth_poorest = False
        wealth_richer = False
        wealth_richest = False
    elif wealth_level == "Poorer":
        wealth_middle = False
        wealth_poorer = True
        wealth_poorest = False
        wealth_richer = False
        wealth_richest = False
    elif wealth_level == "Poorest":
        wealth_middle = False
        wealth_poorer = False
        wealth_poorest = True
        wealth_richer = False
        wealth_richest = False
    elif wealth_level == "Richer":
        wealth_middle = False
        wealth_poorer = False
        wealth_poorest = False
        wealth_richer = True
        wealth_richest = False
    else:
        wealth_middle = False
        wealth_poorer = False
        wealth_poorest = False
        wealth_richer = False
        wealth_richest = True

    net_available_options = ("No", "Yes")
    net_available_choice = st.selectbox(
        "Do you have mosquito bed net for sleeping?", net_available_options
    )
    if net_available_choice == "No":
        net_available_no = True
        net_available_yes = False
    else:
        net_available_no = False
        net_available_yes = True
    net_available_options_smoker = ("No", "Yes")
    net_available_choice_smoker = st.selectbox(
        "Does the mother smoke cigarettes?", net_available_options_smoker
    )
    if net_available_choice_smoker == "No":
        is_smoker_no = True
        is_smoker_yes = False
    else:
        is_smoker_no = False
        is_smoker_yes = True

    net_available_options_education = ("No education", "Primary", "Secondary", "Higher")
    net_available_choice_education = st.selectbox(
        "What's the highest educational level of the mother?",
        net_available_options_education,
    )
    if net_available_choice_education == "No education":
        education_higher = False
        education_no = True
        education_primary = False
        education_secondary = False
    elif net_available_choice_education == "Primary":
        education_higher = False
        education_no = False
        education_primary = True
        education_secondary = False
    elif net_available_choice_education == "Secondary":
        education_higher = False
        education_no = False
        education_primary = False
        education_secondary = True
    else:
        education_higher = False
        education_no = False
        education_primary = False
        education_secondary = True

    marital_status = st.radio(
        "What's the marital status of mother?",
        ("Never", "Living", "Married", "Divorced", "Seperated", "Widowed"),
    )
    marital_status_divorced = False
    marital_status_living = False
    marital_status_married = False
    marital_status_never = False
    marital_status_separated = False
    marital_status_widowed = False

    if marital_status == "Never":
        marital_status_divorced = False
        marital_status_living = False
        marital_status_married = False
        marital_status_never = True
        marital_status_separated = False
        marital_status_widowed = False
    elif marital_status == "Living":
        marital_status_divorced = False
        marital_status_living = True
        marital_status_married = False
        marital_status_never = False
        marital_status_separated = False
        marital_status_widowed = False
    elif marital_status == "Married":
        marital_status_divorced = False
        marital_status_living = False
        marital_status_married = True
        marital_status_never = False
        marital_status_separated = False
        marital_status_widowed = False
    elif marital_status == "Divorced":
        marital_status_divorced = True
        marital_status_living = False
        marital_status_married = False
        marital_status_never = False
        marital_status_separated = False
        marital_status_widowed = False
    elif marital_status == "Seperated":
        marital_status_divorced = False
        marital_status_living = False
        marital_status_married = False
        marital_status_never = False
        marital_status_separated = True
        marital_status_widowed = False
    else:
        marital_status_divorced = False
        marital_status_living = False
        marital_status_married = False
        marital_status_never = False
        marital_status_separated = False
        marital_status_widowed = True

    net_available_options_partner = ("Elsewhere", "Unknown", "Yes")
    net_available_choice_partner = st.selectbox(
        "Is the partner co-living with the mother?", net_available_options_partner
    )
    if net_available_choice_partner == "Elsewhere":
        partner_coliving_elsewhere = True
        partner_coliving_unknown = False
        partner_coliving_yes = False
    elif net_available_choice_partner == "Yes":
        partner_coliving_elsewhere = False
        partner_coliving_yes = True
        partner_coliving_unknown = False
    else:
        partner_coliving_elsewhere = False
        partner_coliving_yes = False
        partner_coliving_unknown = True

    net_available_options_fever = ("No", "Yes", "Unknown")
    net_available_choice_fever = st.selectbox(
        "Are you aware of any fever history of your child in last two weeks?",
        net_available_options_fever,
    )
    if net_available_choice_fever == "No":
        fever_history_no = True
        fever_history_yes = False
        fever_history_unknown = False
    elif net_available_choice_fever == "Yes":
        fever_history_no = False
        fever_history_yes = True
        fever_history_unknown = False
    else:
        fever_history_no = False
        fever_history_yes = False
        fever_history_unknown = True

    net_available_options_ironpill = ("No", "Yes", "Unknown")
    net_available_choice_ironpill = st.selectbox(
        "Does you child take any iron pill, sprinkles or syrup?",
        net_available_options_ironpill,
    )
    if net_available_choice_ironpill == "No":
        ironpill_taken_no = True
        ironpill_taken_yes = False
        ironpill_taken_unknown = False
    elif net_available_choice_ironpill == "Yes":
        ironpill_taken_no = False
        ironpill_taken_yes = True
        ironpill_taken_unknown = False
    else:
        ironpill_taken_no = False
        ironpill_taken_yes = False
        ironpill_taken_unknown = True

    family_choice = st.radio("Family", ("Yes", "No"))
    family = family_choice == "Yes"
    predict_button = st.form_submit_button(label="Predict")
if predict_button:
    scaler = StandardScaler()
    input_features = [
        age_group,
        past_births,
        first_birth_age,
        adj_hemo_altsmoke,
        adj_hemo_alt,
        residence_rural,
        residence_urban,
        wealth_middle,
        wealth_poorer,
        wealth_poorest,
        wealth_richer,
        wealth_richest,
        net_available_no,
        net_available_yes,
        is_smoker_no,
        is_smoker_yes,
        education_higher,
        education_no,
        education_primary,
        education_secondary,
        marital_status_divorced,
        marital_status_living,
        marital_status_married,
        marital_status_never,
        marital_status_separated,
        marital_status_widowed,
        partner_coliving_elsewhere,
        partner_coliving_unknown,
        partner_coliving_yes,
        fever_history_no,
        fever_history_unknown,
        fever_history_yes,
        ironpill_taken_no,
        ironpill_taken_unknown,
        ironpill_taken_yes,
        family,
    ]
    model_path = "gbm_model.joblib"
    if os.path.isfile(model_path):
        with open(model_path, "rb") as fo:
            model_data = load(fo)
            clf = model_data["model"]
            scaler = model_data["scaler"]
            columns = model_data["columns"]
    else:
        df = load_data()
        for col in [
            "breastfed",
            "anemia_level1",
            "adj_hemo_altbirth",
            "adj_hemo_altsmkbirth",
        ]:
            df = df.drop(col, axis=1)
        X = df.drop("anemia_level", axis=1)
        y = df["anemia_level"]

        X = X.fillna(X.mean())

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=73939133
        )

        scaler = StandardScaler()
        # scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)

        gbm = GradientBoostingClassifier(
            n_estimators=100, learning_rate=0.1, max_depth=3, random_state=73939133
        )
        gbm.fit(X_train_scaled, y_train)
        columns = X.columns
        model_data = {"model": gbm, "scaler": scaler, "columns": columns}
        with open(model_path, "wb") as fo:
            dump(model_data, fo)

        clf = gbm

    input_df = pd.DataFrame(data=[input_features], columns=columns)
    input_scaled = scaler.transform(input_df)

    y_pred_gbm = clf.predict(input_scaled)
    # From phase 1 notebook
    anemia_level_mapper = {"Not anemic": 1, "Mild": 2, "Moderate": 3, "Severe": 4}
    anemia_level_mapper_reverse = {v: k for k, v in anemia_level_mapper.items()}
    pred_val = anemia_level_mapper_reverse.get(y_pred_gbm[0], "Unidentified")
    st.write(f"Predicted Anemia Level: {pred_val}")
