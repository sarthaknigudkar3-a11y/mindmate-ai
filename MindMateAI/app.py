import streamlit as st
import pandas as pd
import sqlite3
from datetime import datetime
import pickle
import matplotlib.pyplot as plt

# Load trained model
model = pickle.load(open("emotion_model.pkl", "rb"))
vectorizer = pickle.load(open("vectorizer.pkl", "rb"))

# Database connection
conn = sqlite3.connect("mood_tracker.db", check_same_thread=False)
cursor = conn.cursor()

cursor.execute("""
CREATE TABLE IF NOT EXISTS moods (
    date TEXT,
    mood TEXT,
    note TEXT
)
""")
conn.commit()

# Emergency keywords
emergency_keywords = [
    "want to die", "kill myself", "hurt myself",
    "suicide", "end my life"
]

def detect_emotion(text):
    text_vec = vectorizer.transform([text])
    prediction = model.predict(text_vec)[0]
    return prediction

def chatbot_response(emotion):
    responses = {
        "joy": "Glad you're feeling positive today 😊",
        "sadness": "I'm sorry you're feeling low. Try talking to a friend or journaling.",
        "anger": "Take a short break and do deep breathing for 5 minutes.",
        "fear": "Try focusing on what you can control today.",
        "stress": "Break tasks into smaller parts and take regular study breaks."
    }
    return responses.get(emotion, "Take care of yourself.")

st.title("🧠 MindMate AI - Student Wellness Assistant")

menu = st.sidebar.selectbox(
    "Menu",
    ["Chatbot", "Mood Tracker", "Mood Analytics"]
)

if menu == "Chatbot":
    st.subheader("Mental Health Chatbot")

    user_input = st.text_area("How are you feeling today?")

    if st.button("Analyze"):
        if any(word in user_input.lower() for word in emergency_keywords):
            st.error("⚠️ Please contact a trusted person or emergency helpline immediately.")
            st.write("India Helpline: Tele-MANAS: 14416")
        else:
            emotion = detect_emotion(user_input)
            response = chatbot_response(emotion)

            st.success(f"Detected Emotion: {emotion}")
            st.info(response)

if menu == "Mood Tracker":
    st.subheader("Daily Mood Tracker")

    mood = st.selectbox(
        "Select your mood",
        ["Happy", "Neutral", "Stressed", "Sad", "Anxious"]
    )

    note = st.text_input("Optional note")

    if st.button("Save Mood"):
        cursor.execute(
            "INSERT INTO moods VALUES (?, ?, ?)",
            (str(datetime.now().date()), mood, note)
        )
        conn.commit()
        st.success("Mood saved successfully!")

if menu == "Mood Analytics":
    st.subheader("Mood Analytics Dashboard")

    df = pd.read_sql_query("SELECT * FROM moods", conn)

    if not df.empty:
        mood_counts = df["mood"].value_counts()

        fig, ax = plt.subplots()
        mood_counts.plot(kind="bar", ax=ax)
        plt.xlabel("Mood")
        plt.ylabel("Count")
        plt.title("Mood Distribution")

        st.pyplot(fig)
        st.dataframe(df)
    else:
        st.warning("No mood data found.")