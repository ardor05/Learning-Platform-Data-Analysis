import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def app():
    # Load the dataset
    file_path = r'C:/Users/ACER/Downloads/capstone/cleaned_logs_Statistics1.0.csv'
    data = pd.read_csv(file_path)

    # Data preprocessing for clustering
    data['time'] = pd.to_datetime(data['time'])
    data['date'] = data['time'].dt.date
    daily_activity = data.groupby(['user_full_name', 'date']).size().reset_index(name='activity_count')

    # Standardize data for clustering
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(daily_activity[['activity_count']])

    # Clustering using KMeans
    kmeans = KMeans(n_clusters=3, random_state=42)
    daily_activity['cluster'] = kmeans.fit_predict(data_scaled)

    # Streamlit dashboard
    #st.title("E-Learning Platform Dashboard")
    #st.header("Student Activity and Performance Analysis")

    # Clustering visualization with guidelines
    st.subheader("Clustering Analysis")
    st.markdown("""
    **Cluster Distribution:**

    - **Cluster 0 (Purple):** Represents days with relatively low activity counts. These dots are mostly concentrated near the bottom of the y-axis, indicating low activity.
    - **Cluster 1 (Yellow):** Represents days with moderate activity counts. These dots are spread out more along the y-axis, indicating a moderate level of student activity.
    - **Cluster 2 (Teal):** Represents days with high activity counts. These dots are mostly found higher up on the y-axis, indicating higher activity levels.

  
    """)

    fig, ax = plt.subplots(figsize=(12, 6))
    sns.scatterplot(data=daily_activity, x='date', y='activity_count', hue='cluster', palette='viridis', ax=ax)
    ax.set_title('Daily Activity Clustering')
    ax.set_xlabel('Date')
    ax.set_ylabel('Activity Count')
    st.pyplot(fig)

    # Performance Analysis by Cluster (can be retained or enhanced as needed)
    st.subheader("Student Performance Analysis")

    # Cluster-wise Student Progress Analysis (can be retained or enhanced as needed)
    st.subheader("Cluster-wise Student Progress Analysis")

    for cluster in sorted(daily_activity['cluster'].unique()):
        st.subheader(f"Cluster {cluster} - Student Progress")

        cluster_data = daily_activity[daily_activity['cluster'] == cluster]
        student_activity = cluster_data.groupby('user_full_name')['activity_count'].sum().sort_values(ascending=False)

        st.write(f"Top Students in Cluster {cluster} (based on activity count):")
        st.write(student_activity.head(10))

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.barplot(x=student_activity.head(10).values, y=student_activity.head(10).index, palette='viridis', ax=ax)
        ax.set_title(f'Top Students Activity Count in Cluster {cluster}')
        ax.set_xlabel('Activity Count')
        ax.set_ylabel('Student')
        st.pyplot(fig)

if __name__ == '__main__':
    st.set_page_config(layout="wide")
    app()
