import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# Load dataset
data = pd.read_csv("spam_dataset.tsv", sep="\t", names=["label","message"])

data["label"] = data["label"].map({"ham":0,"spam":1})

X = data["message"]
y = data["label"]


# Train test split
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# TF-IDF
vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# Train model
model = MultinomialNB()

model.fit(X_train_vec,y_train)


# Accuracy
pred = model.predict(X_test_vec)

accuracy = accuracy_score(y_test,pred)


# Suspicious link detection
def detect_link(text):

    pattern = r"http[s]?://"

    if re.search(pattern,text):

        return True

    return False


# Phishing keywords
def detect_phishing(text):

    keywords = ["bank","password","login","verify","account","urgent","otp"]

    text = text.lower()

    for word in keywords:

        if word in text:

            return True

    return False


# Risk level
def risk_level(prob):

    if prob > 80:
        return "HIGH RISK"

    elif prob > 50:
        return "MEDIUM RISK"

    else:
        return "LOW RISK"


# -----------------------
# STREAMLIT DASHBOARD
# -----------------------

st.title("AI Email Security Dashboard")

col1, col2 = st.columns(2)

col1.metric("Model Accuracy", f"{accuracy*100:.2f}%")

col2.metric("Total Emails", len(data))


# Graph
spam_counts = data["label"].value_counts()

fig, ax = plt.subplots()

ax.pie(spam_counts, labels=["Ham","Spam"], autopct='%1.1f%%')

ax.set_title("Email Distribution")

st.pyplot(fig)


# Email Input
st.subheader("Check Email Security")

msg = st.text_area("Enter Email Message")


if st.button("Analyze Email"):

    vec = vectorizer.transform([msg])

    result = model.predict(vec)

    prob = model.predict_proba(vec)

    spam_prob = prob[0][1]*100

    risk = risk_level(spam_prob)


    st.write(f"Spam Probability: {spam_prob:.2f}%")

    if result[0] == 1:

        st.error("Spam Email Detected")

    else:

        st.success("Email appears safe")


    st.info(f"Risk Level: {risk}")


    if detect_link(msg):

        st.warning("Suspicious link detected")


    if detect_phishing(msg):

        st.warning("Possible phishing attempt detected")