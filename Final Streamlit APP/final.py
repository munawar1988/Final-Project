import streamlit as st
import joblib
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import PIL.Image
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt

# Load Models and Data
model_odi = joblib.load('rf_model.pkl')
label_encoders_odi = joblib.load('label_encoders.pkl')
data_odi = pd.read_csv('odi.csv')

model_t20 = joblib.load('rf_t20_model.pkl')
label_encoders_t20 = joblib.load('t20_label_encoders.pkl')
data_t20 = pd.read_csv('t20.csv')

# Load the Run Out Prediction model
run_out_model = tf.keras.models.load_model('patch.h5')
class_labels = {0: 'not out', 1: 'out'}

# HTML and CSS for Custom Styling
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    body {
        font-family: 'Roboto', sans-serif;
    }
    .title {
        color: #4CAF50;
        text-align: center;
        animation: fadeIn 2s;
    }
    .header {
        color: #f44336;
        text-align: center;
        animation: fadeIn 2s;
    }
    .subheader {
        color: #2196F3;
        text-align: center;
        animation: fadeIn 3s;
    }
    .description {
        color: #000;
        text-align: justify;
        animation: fadeIn 3s;
    }
    .img-container {
        text-align: center;
    }
    .fadeIn {
        animation: fadeIn 2s;
    }
    @keyframes fadeIn {
        0% { opacity: 0; }
        100% { opacity: 1; }
    }
    .btn {
        background-color: #4CAF50;
        color: white;
        border: none;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 16px;
    }
    </style>
""", unsafe_allow_html=True)

# Streamlit app layout
st.markdown('<h1 class="title">Cricket AI Web App</h1>', unsafe_allow_html=True)

# Sidebar for navigation
st.sidebar.title("Navigation")

page = st.sidebar.radio("Go to", ["Home", "ODI Analysis", "T20 Analysis", "ODI Match Predictor", "T20 Match Predictor", "Run Out Prediction"])

# Home Page
# Enhanced Home Page with Detailed Project Description and Visuals

if page == "Home":
    st.markdown('<h2 class="header">Welcome to Cricket AI Web App</h2>', unsafe_allow_html=True)

    # Project Introduction
    st.markdown("""
    <p class="description">
    This web application leverages advanced data analytics and machine learning techniques to provide detailed insights into cricket matches. Whether you are an enthusiast looking to analyze ODI and T20 matches, or a data scientist interested in predicting match outcomes, this platform offers a comprehensive suite of tools.
    </p>
    """, unsafe_allow_html=True)

    # Project Details Section
    st.markdown('<h3 class="subheader">Project Features:</h3>', unsafe_allow_html=True)

    # Project Details Table
    st.markdown("""
    <table style="width:100%">
      <tr>
        <th>Feature</th>
        <th>Description</th>
      </tr>
      <tr>
        <td>ODI Analysis</td>
        <td>Analyze historical ODI match data including team performance, venue statistics, and more.</td>
      </tr>
      <tr>
        <td>T20 Analysis</td>
        <td>Explore T20 match data with interactive visualizations and team comparisons.</td>
      </tr>
      <tr>
        <td>ODI Match Predictor</td>
        <td>Predict the winner of an ODI match based on team and venue selections.</td>
      </tr>
      <tr>
        <td>T20 Match Predictor</td>
        <td>Predict the winner of a T20 match using machine learning models.</td>
      </tr>
      <tr>
        <td>Run Out Prediction</td>
        <td>Upload an image and use AI to predict if a player is out or not out.</td>
      </tr>
    </table>
    """, unsafe_allow_html=True)

    # Adding an Image for Visual Appeal
    st.image('cricket_image.jpg', caption='Cricket Analysis & Prediction', use_column_width=True)

    # Additional Information Section
    st.markdown("""
    <p class="description">
    <strong>Technology Stack:</strong> This app is built using Python and Streamlit, leveraging powerful libraries such as TensorFlow, scikit-learn, and Plotly for machine learning and data visualization. The models have been trained on extensive cricket datasets to ensure accurate predictions.
    </p>
    <p class="description">
    <strong>How to Use:</strong> Navigate through the different sections using the sidebar. Select the desired feature and interact with the inputs to explore cricket data or predict match outcomes.
    </p>
    """, unsafe_allow_html=True)

# ODI Analysis Page
elif page == "ODI Analysis":
    st.title(":bar_chart: ODI Cricket Matches Data Analysis")
    
    # Reading Data 
    data = pd.read_csv('ODI_Match_info.csv')

    st.header("Get Desired Filters")

    # Team 1 and Team 2 Analysis 
    team1_unique = data['team1'].unique()
    team2_unique = data['team2'].unique()
    all_teams = list(set(team1_unique) | set(team2_unique))

    col1, col2 = st.columns(2)

    with col1:
        Team_1 = st.selectbox('Select Team 1', all_teams)
    with col2:
        Team_2 = st.selectbox('Select Team 2', all_teams)

    venues = data['venue'].unique()
    selected_venue = st.selectbox('Select Venue', venues)

    if st.button("Analyze"):
        st.markdown('<h4 class="hover-effect">Analysis Result</h4>', unsafe_allow_html=True) 
        st.markdown('<br><br>', unsafe_allow_html=True)
        
        # Team 1 Analysis
        st.subheader(f"Matches Played and Won by {Team_1}")
        filtered_data_team1 = data[(data['team1'] == Team_1) | (data['team2'] == Team_1)]
        matches_played_team1 = len(filtered_data_team1)
        matches_won_team1 = len(filtered_data_team1[filtered_data_team1['winner'] == Team_1])
        
        df_team1 = pd.DataFrame({
            'Category': [f'Matches Played by {Team_1}', f'Matches Won By {Team_1}'],
            'Value': [matches_played_team1, matches_won_team1]
        })
        
        color_map_team1 = {
            'Matches Played': 'red',
            'Matches Won': 'orange'
        }
        
        df_team1 = df_team1.sort_values(by='Value', ascending=False)
        fig_team1 = px.bar(df_team1, x='Category', y='Value', color='Category', color_discrete_map=color_map_team1)
        st.plotly_chart(fig_team1)

        # Team 2 Analysis
        st.subheader(f"Matches Played and Won by {Team_2}")
        filtered_data_team2 = data[(data['team1'] == Team_2) | (data['team2'] == Team_2)]
        matches_played_team2 = len(filtered_data_team2)
        matches_won_team2 = len(filtered_data_team2[filtered_data_team2['winner'] == Team_2])
        
        df_team2 = pd.DataFrame({
            'Category': [f'Matches Played by {Team_2}', f'Matches Won By {Team_2}'],
            'Value': [matches_played_team2, matches_won_team2]
        })
        
        color_map_team2 = {
            'Matches Played': 'blue',
            'Matches Won': 'green'
        }
        
        df_team2 = df_team2.sort_values(by='Value', ascending=False)
        fig_team2 = px.bar(df_team2, x='Category', y='Value', color='Category', color_discrete_map=color_map_team2)
        st.plotly_chart(fig_team2)

        # Venue Analysis
        st.subheader(f"Matches at {selected_venue}")
        filtered_data_venue = data[data['venue'] == selected_venue]
        matches_at_venue = len(filtered_data_venue)
        
        st.write(f"Total matches played at {selected_venue}: {matches_at_venue}")
        
        if matches_at_venue > 0:
            fig_venue = px.histogram(filtered_data_venue, x='winner', title=f'Winners at {selected_venue}')
            st.plotly_chart(fig_venue)

# T20 Analysis Page
elif page == "T20 Analysis":
    st.title("T20 Cricket Matches Data Analysis")

    # Load Data
    @st.cache_data
    def load_data():
        data = pd.read_csv('t20.csv')
        return data

    # Data Preprocessing
    def preprocess_data(data):
        data['Date'] = pd.to_datetime(data['Date'], format='%m/%d/%Y')
        data['Year'] = data['Date'].dt.year
        return data

    # Load and preprocess data
    data = load_data()
    data = preprocess_data(data)

    # Main Page Filters
    st.header("Filter Options")

    teams = pd.concat([data['Bat First'], data['Bat Second']]).unique()
    selected_team = st.selectbox("Select Team", teams)

    opponent_teams = ['All'] + list(teams[teams != selected_team])
    selected_opponent = st.selectbox("Select Opponent Team", opponent_teams)

    # Filter venues based on selected team and opponent team
    if selected_team:
        team_matches = data[(data['Bat First'] == selected_team) | (data['Bat Second'] == selected_team)]
    if selected_opponent != 'All':
        opponent_matches = team_matches[(team_matches['Bat First'] == selected_opponent) | (team_matches['Bat Second'] == selected_opponent)]
        venues = ['All'] + list(opponent_matches['Venue'].unique())
    else:
        venues = ['All'] + list(team_matches['Venue'].unique())

    selected_venue = st.selectbox("Select Venue", venues)

    # Filter data based on selection
    if selected_team:
        data = data[(data['Bat First'] == selected_team) | (data['Bat Second'] == selected_team)]
    if selected_opponent != 'All':
        data = data[(data['Bat First'] == selected_opponent) | (data['Bat Second'] == selected_opponent)]
    if selected_venue != 'All':
        data = data[data['Venue'] == selected_venue]

    # Display the filtered data
    st.subheader("Filtered Dataset")
    st.write(data.head())

    # Show basic statistics
    st.subheader("Basic Statistics")
    st.write(data.describe())

    # Visualizations
    st.subheader("Visualizations")

    # Plotting the distribution of wins by year
    st.write("### Distribution of Wins by Year")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.countplot(data=data, x='Year', hue='Winner', ax=ax)
    ax.set_title(f'Distribution of Wins by Year for {selected_team}')
    st.pyplot(fig)

    # Plotting the total number of matches played by venue
    st.write(f"### Number of Matches Played at Different Venues by {selected_team}")
    fig, ax = plt.subplots(figsize=(10, 6))
    venue_count = data['Venue'].value_counts()
    sns.barplot(x=venue_count.index, y=venue_count.values, ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title(f'Number of Matches Played at Different Venues by {selected_team}')
    st.pyplot(fig)

    # Plotting the head-to-head results between the selected team and opponent
    if selected_opponent != 'All':
        st.write(f"### Head-to-Head Results: {selected_team} vs {selected_opponent}")
        head_to_head_data = data[((data['Bat First'] == selected_team) & (data['Bat Second'] == selected_opponent)) |
                                 ((data['Bat First'] == selected_opponent) & (data['Bat Second'] == selected_team))]
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.countplot(data=head_to_head_data, x='Winner', ax=ax)
        ax.set_title(f'Head-to-Head Results: {selected_team} vs {selected_opponent}')
        st.pyplot(fig)

# ODI Match Predictor Page
elif page == "ODI Match Predictor":
    st.title('ODI Match Winner Predictor')
    # Team1 selection
    team1 = st.selectbox('Select Team 1', label_encoders_odi['team1'].classes_)

    # Filter team2 options based on team1 selection
    team2_options = [team for team in label_encoders_odi['team2'].classes_ if team != team1]
    team2 = st.selectbox('Select Team 2', team2_options)

    # Filter venue options based on team1 and team2 selection
    venues = data_odi[(data_odi['team1'] == team1) | (data_odi['team2'] == team1) |
                      (data_odi['team1'] == team2) | (data_odi['team2'] == team2)]['venue'].unique()
    venue = st.selectbox('Select Venue', venues)

    # Predict button
    if st.button('Predict Winner'):
        if team1 not in label_encoders_odi['team1'].classes_ or \
           team2 not in label_encoders_odi['team2'].classes_ or \
           venue not in label_encoders_odi['venue'].classes_:
            st.write("Error: One of the inputs is not recognized.")
        else:
            team1_enc = label_encoders_odi['team1'].transform([team1])[0]
            team2_enc = label_encoders_odi['team2'].transform([team2])[0]
            venue_enc = label_encoders_odi['venue'].transform([venue])[0]
            prediction_enc = model_odi.predict([[team1_enc, team2_enc, venue_enc]])[0]
            prediction = label_encoders_odi['winner'].inverse_transform([prediction_enc])[0]
            st.write(f'Predicted Winner: {prediction}')

    # Instructions
    st.write("""
    ## Instructions
    1. Select Team 1.
    2. Select Team 2 (Team 2 options will not include the selected Team 1).
    3. Select the venue (Venue options are limited to where both selected teams have played).
    4. Click the 'Predict Winner' button to see the prediction.
    """)

# T20 Match Predictor Page
elif page == "T20 Match Predictor":
    st.title("T20 Match Winner Predictor")
    # Input fields from dataset columns
    bat_first = st.selectbox('Select Team Batting First', sorted(data_t20['Bat First'].unique()))
    bat_second_options = sorted([team for team in data_t20['Bat Second'].unique() if team != bat_first])
    bat_second = st.selectbox('Select Team Batting Second', bat_second_options)

    # Filter venues based on selected teams
    venues_both_teams_played = data_t20[data_t20['Bat First'].isin([bat_first, bat_second]) & 
                                         data_t20['Bat Second'].isin([bat_first, bat_second])]['Venue'].unique()
    venue = st.selectbox('Select Venue', sorted(venues_both_teams_played))

    def predict_winner(bat_first, bat_second, venue):
        bat_first_enc = label_encoders_t20['Bat First'].transform([bat_first])[0]
        bat_second_enc = label_encoders_t20['Bat Second'].transform([bat_second])[0]
        venue_enc = label_encoders_t20['Venue'].transform([venue])[0]
        prediction_enc = model_t20.predict([[bat_first_enc, bat_second_enc, venue_enc]])[0]
        prediction = label_encoders_t20['Winner'].inverse_transform([prediction_enc])[0]
        return prediction

    if st.button('Predict Winner'):
        result = predict_winner(bat_first, bat_second, venue)
        st.write(f'Predicted winner: {result}')

# Run Out Prediction Page
elif page == "Run Out Prediction":
    st.title("Run Out Prediction")
    st.markdown('<h3 class="subheader">Run Out Prediction</h3>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Choose an image for run-out prediction...", type="jpg")
    if uploaded_file is not None:
        try:
            st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
            img = image.load_img(uploaded_file, target_size=(224, 224))
            img_array = image.img_to_array(img)
            img_array = np.expand_dims(img_array, axis=0)
            img_array /= 255.0

            prediction = run_out_model.predict(img_array)
            predicted_class = int(round(prediction[0][0]))
            predicted_label = class_labels[predicted_class]

            st.markdown('<h4 class="subheader">Prediction Result:</h4>', unsafe_allow_html=True)
            st.markdown(f'<p class="description">The model predicts: {predicted_label} (Class {predicted_class})</p>', unsafe_allow_html=True)
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")
