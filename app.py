import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import time
from sklearn.metrics import mean_squared_error, r2_score

# Cache the data loading
@st.cache_data
def load_data():
    calories = pd.read_csv("calories.csv")
    exercise = pd.read_csv("exercise.csv")
    return calories, exercise

calories, exercise = load_data()

# Merge and preprocess data
def preprocess_data(calories, exercise):
    exercise_df = exercise.merge(calories, on="User_ID")
    exercise_df.drop(columns="User_ID", inplace=True)
    exercise_df["BMI"] = exercise_df["Weight"] / ((exercise_df["Height"] / 100) ** 2)
    exercise_df["BMI"] = round(exercise_df["BMI"], 2)
    return exercise_df

exercise_df = preprocess_data(calories, exercise)

# User input features
def user_input_features():
    st.sidebar.header("User Input Parameters: ")
    age = st.sidebar.slider("Age: ", 10, 100, 21)
    bmi = st.sidebar.slider("BMI: ", 15, 40, 20)
    duration = st.sidebar.slider("Duration (min): ", 0, 60, 30)
    heart_rate = st.sidebar.slider("Heart Rate: ", 60, 150, 90)
    body_temp = st.sidebar.slider("Body Temperature (C): ", 36, 42, 38)
    gender_button = st.sidebar.radio("Gender: ", ("Male", "Female"))

    gender = 1 if gender_button == "Male" else 0

    data_model = {
        "Age": age,
        "BMI": bmi,
        "Duration": duration,
        "Heart_Rate": heart_rate,
        "Body_Temp": body_temp,
        "Gender_male": gender
    }

    features = pd.DataFrame(data_model, index=[0])
    return features

df = user_input_features()

# Add Calorie Goal Tracker Input
st.sidebar.header("Calorie Goal Tracker")
calorie_goal = st.sidebar.number_input("Set your Calorie Burn Goal:", min_value=0, max_value=1000, value=200)

st.write("---")
st.header("Your Parameters:")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)
st.write(df)

# Train-test split
exercise_train_data, exercise_test_data = train_test_split(exercise_df, test_size=0.2, random_state=1)

# Prepare the training and testing sets
def prepare_data(data):
    data = data[["Gender", "Age", "BMI", "Duration", "Heart_Rate", "Body_Temp", "Calories"]]
    data = pd.get_dummies(data, drop_first=True)
    return data

exercise_train_data = prepare_data(exercise_train_data)
exercise_test_data = prepare_data(exercise_test_data)

# Separate features and labels
X_train = exercise_train_data.drop("Calories", axis=1)
y_train = exercise_train_data["Calories"]

X_test = exercise_test_data.drop("Calories", axis=1)
y_test = exercise_test_data["Calories"]

# Train the model
random_reg = RandomForestRegressor(n_estimators=1000, max_features=3, max_depth=6)
random_reg.fit(X_train, y_train)

# Align prediction data columns with training data
df = df.reindex(columns=X_train.columns, fill_value=0)

# Make prediction
prediction = random_reg.predict(df)

st.write("---")
st.header("Prediction:")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

st.write(f"{round(prediction[0], 2)} kilocalories")

# Provide Feedback on User Progress
st.write("---")
st.header("Calorie Goal Feedback:")

if prediction[0] >= calorie_goal:
    st.success(f"Congratulations! You've exceeded your calorie burn goal of {calorie_goal} kilocalories by burning {round(prediction[0], 2)} kilocalories.")
else:
    st.warning(f"Keep going! You are {round(calorie_goal - prediction[0], 2)} kilocalories away from reaching your goal of {calorie_goal} kilocalories.")

# Suggestions for Improvement
st.write("Suggestions to achieve your goal:")
suggestions = [
    "Increase exercise duration.",
    "Consider higher-intensity workouts.",
    "Ensure a balanced diet to maintain energy levels.",
    "Monitor heart rate and body temperature to optimize performance.",
]
for suggestion in suggestions:
    st.write("- " + suggestion)

# Visualization
st.write("---")
st.header("Similar Results Visualization:")
calorie_range = [prediction[0] - 10, prediction[0] + 10]
similar_data = exercise_df[(exercise_df["Calories"] >= calorie_range[0]) & (exercise_df["Calories"] <= calorie_range[1])]

fig, ax = plt.subplots()
sns.histplot(similar_data["Calories"], bins=10, kde=True, ax=ax)
ax.set_title("Distribution of Similar Calorie Burns")
st.pyplot(fig)

st.write("---")
st.header("Similar Results:")
latest_iteration = st.empty()
bar = st.progress(0)
for i in range(100):
    bar.progress(i + 1)
    time.sleep(0.01)

st.write(similar_data.sample(5))

st.write("---")
st.header("General Information:")

# Boolean logic for age, duration, etc., compared to the user's input
boolean_age = (exercise_df["Age"] < df["Age"].values[0]).tolist()
boolean_duration = (exercise_df["Duration"] < df["Duration"].values[0]).tolist()
boolean_body_temp = (exercise_df["Body_Temp"] < df["Body_Temp"].values[0]).tolist()
boolean_heart_rate = (exercise_df["Heart_Rate"] < df["Heart_Rate"].values[0]).tolist()

st.write("You are older than", round(sum(boolean_age) / len(boolean_age), 2) * 100, "% of other people.")
st.write("Your exercise duration is higher than", round(sum(boolean_duration) / len(boolean_duration), 2) * 100, "% of other people.")
st.write("You have a higher heart rate than", round(sum(boolean_heart_rate) / len(boolean_heart_rate), 2) * 100, "% of other people during exercise.")
st.write("You have a higher body temperature than", round(sum(boolean_body_temp) / len(boolean_body_temp), 2) * 100, "% of other people during exercise.")

# Scatter Plot Visualization
st.write("---")
st.header("Scatter Plot Visualization:")

# Scatter plot of Age vs Calories
fig, ax = plt.subplots()
sns.scatterplot(x="Age", y="Calories", data=exercise_df, ax=ax)
ax.set_title("Age vs Calories Burned")
st.pyplot(fig)

# Box Plot Visualization
st.write("---")
st.header("Box Plot Visualization:")

# Box plot of BMI vs Calories
fig, ax = plt.subplots()
sns.boxplot(x="BMI", y="Calories", data=exercise_df, ax=ax)
ax.set_title("BMI vs Calories Burned")
st.pyplot(fig)

# Model Evaluation Metrics
st.write("---")
st.header("Model Evaluation Metrics:")

# Predictions on test data
y_pred = random_reg.predict(X_test)

# Calculate evaluation metrics
rmse = mean_squared_error(y_test, y_pred)
r_squared = r2_score(y_test, y_pred)

st.write(f"RMSE: {rmse:.2f}")
st.write(f"R-squared: {r_squared:.2f}")

# Download Results
st.write("---")
st.header("Download Results:")

# Prepare data for download
download_data = df.copy()
download_data['Predicted Calories'] = round(prediction[0], 2)

# Convert DataFrame to CSV
csv = download_data.to_csv(index=False)

st.download_button(
    label="Download Prediction as CSV",
    data=csv,
    file_name='prediction.csv',
    mime='text/csv',
)

# Feedback Form
st.write("---")
st.header("Feedback:")

feedback = st.text_area("How accurate was the prediction? Please provide your feedback:")
if st.button("Submit Feedback"):
    st.success("Thank you for your feedback!")