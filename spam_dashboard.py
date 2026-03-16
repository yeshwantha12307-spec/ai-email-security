import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import re

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


# ---------------------------
# Load Dataset
# ---------------------------
data = pd.read_csv("spam_dataset.tsv", sep="\t", names=["label","message"])

data["label"] = data["label"].map({"ham":0,"spam":1})

X = data["message"]
y = data["label"]


# ---------------------------
# Train Test Split
# ---------------------------
X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2)


# ---------------------------
# Feature Extraction
# ---------------------------
vectorizer = TfidfVectorizer()

X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)


# ---------------------------
# Train Model
# ---------------------------
model = MultinomialNB()

model.fit(X_train_vec,y_train)


# ---------------------------
# Accuracy
# ---------------------------
pred = model.predict(X_test_vec)

acc = accuracy_score(y_test,pred)


# ---------------------------
# Suspicious Link Detection
# ---------------------------
def detect_link(text):

    pattern = r"http[s]?://"

    if re.search(pattern,text):

        return True

    return False


# ---------------------------
# Phishing Keyword Detection
# ---------------------------
def detect_phishing(text):

    keywords = ["bank","password","login","verify","account","urgent","otp"]

    text = text.lower()

    for word in keywords:

        if word in text:

            return True

    return False


# ---------------------------
# Risk Level
# ---------------------------
def risk_level(prob):

    if prob > 80:

        return "HIGH RISK"

    elif prob > 50:

        return "MEDIUM RISK"

    else:

        return "LOW RISK"


# ---------------------------
# STREAMLIT UI
# ---------------------------
st.title("AI Email Security System")

st.subheader("Model Accuracy")

st.write(round(acc*100,2), "%")


# ---------------------------
# Graph
# ---------------------------
spam_count = data["label"].value_counts()

fig, ax = plt.subplots()

ax.bar(["Ham","Spam"], spam_count)

ax.set_title("Spam vs Ham Emails")

st.pyplot(fig)


# ---------------------------
# User Input
# ---------------------------
msg = st.text_area("Enter Email Message")


if st.button("Check Email"):

    vec = vectorizer.transform([msg])

    result = model.predict(vec)

    prob = model.predict_proba(vec)

    spam_prob = prob[0][1]*100

    risk = risk_level(spam_prob)


    if result[0] == 1:

        st.error(f"Spam Email (Probability {spam_prob:.2f}%)")

    else:

        st.success(f"Safe Email (Spam Probability {spam_prob:.2f}%)")


    st.info(f"Risk Level: {risk}")


    if detect_link(msg):

        st.warning("Suspicious link detected")


    if detect_phishing(msg):

        st.warning("Possible phishing email detected")