import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

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

# Main
def main():
    st.title("T20 Cricket Matches Data Analysis")

    # Load and preprocess data
    data = load_data()
    data = preprocess_data(data)

    # Sidebar for filters
    st.sidebar.header("Filter Options")
    teams = pd.concat([data['Bat First'], data['Bat Second']]).unique()
    selected_team = st.sidebar.selectbox("Select Team", teams)
    opponent_teams = ['All'] + list(teams[teams != selected_team])
    selected_opponent = st.sidebar.selectbox("Select Opponent Team", opponent_teams)
    
    # Filter venues based on selected team and opponent team
    if selected_team:
        team_matches = data[(data['Bat First'] == selected_team) | (data['Bat Second'] == selected_team)]
    if selected_opponent != 'All':
        opponent_matches = team_matches[(team_matches['Bat First'] == selected_opponent) | (team_matches['Bat Second'] == selected_opponent)]
        venues = ['All'] + list(opponent_matches['Venue'].unique())
    else:
        venues = ['All'] + list(team_matches['Venue'].unique())
    
    selected_venue = st.sidebar.selectbox("Select Venue", venues)

    # Filter data based on selection
    if selected_team:
        data = data[(data['Bat First'] == selected_team) | (data['Bat Second'] == selected_team)]
    if selected_opponent != 'All':
        data = data[(data['Bat First'] == selected_opponent) | (data['Bat Second'] == selected_opponent)]
    if selected_venue != 'All':
        data = data[data['Venue'] == selected_venue]

    st.subheader("Dataset")
    st.write(data.head())

    # Show basic statistics
    st.subheader("Basic Statistics")
    st.write(data.describe())

    # Data Visualization
    st.subheader("Data Visualization")

    # Matches per year
    st.write("### Number of Matches Per Year")
    matches_per_year = data['Year'].value_counts().sort_index()
    plt.figure(figsize=(10, 6))
    sns.lineplot(x=matches_per_year.index, y=matches_per_year.values, marker='o')
    plt.title('Number of Matches Per Year')
    plt.xlabel('Year')
    plt.ylabel('Number of Matches')
    st.pyplot(plt)

    # Matches won by teams
    st.write("### Matches Won by Teams")
    matches_won_by_team = data['Winner'].value_counts()
    plt.figure(figsize=(12, 8))
    sns.barplot(x=matches_won_by_team.values, y=matches_won_by_team.index)
    plt.title('Matches Won by Teams')
    plt.xlabel('Number of Matches Won')
    plt.ylabel('Teams')
    st.pyplot(plt)

    # Venue distribution
    st.write("### Distribution of Venues")
    venue_distribution = data['Venue'].value_counts().head(10)
    plt.figure(figsize=(12, 8))
    sns.barplot(x=venue_distribution.values, y=venue_distribution.index)
    plt.title('Top 10 Venues by Number of Matches')
    plt.xlabel('Number of Matches')
    plt.ylabel('Venue')
    st.pyplot(plt)

    # Winning percentage of teams
    st.write("### Winning Percentage of Teams")
    teams = pd.concat([data['Bat First'], data['Bat Second']]).unique()
    winning_percentage = {}
    for team in teams:
        total_matches = data[(data['Bat First'] == team) | (data['Bat Second'] == team)].shape[0]
        total_wins = data[data['Winner'] == team].shape[0]
        if total_matches > 0:
            winning_percentage[team] = (total_wins / total_matches) * 100
    winning_percentage = pd.Series(winning_percentage).sort_values(ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x=winning_percentage.values, y=winning_percentage.index)
    plt.title('Winning Percentage of Teams')
    plt.xlabel('Winning Percentage')
    plt.ylabel('Teams')
    st.pyplot(plt)

if __name__ == "__main__":
    main()
