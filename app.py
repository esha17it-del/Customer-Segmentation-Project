
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

st.set_page_config(
    page_title="Customer Segmentation",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Customer Segmentation Dashboard")
st.write("Analyze customer behavior using Machine Learning.")

# Upload dataset
uploaded_file = st.file_uploader(
    "Upload Customer CSV File",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)
else:
    try:
        df = pd.read_csv("customer_data.csv")
    except FileNotFoundError:
        st.error("Please upload a customer CSV file.")
        st.stop()

st.subheader("Customer Dataset")
st.dataframe(df, use_container_width=True)

# Check required columns
required = [
    "CustomerID",
    "Gender",
    "Age",
    "Annual Income (k$)",
    "Spending Score (1-100)"
]

if not all(col in df.columns for col in required):
    st.error("Dataset columns are missing or incorrect.")
    st.stop()

if df.empty or df[required].isnull().any().any():
    st.error("Dataset is empty or contains missing values.")
    st.stop()

# Customer statistics
st.subheader("📌 Customer Overview")

col1, col2, col3 = st.columns(3)

col1.metric("Total Customers", len(df))
col2.metric("Average Age", round(df["Age"].mean(), 1))
col3.metric(
    "Average Spending",
    round(df["Spending Score (1-100)"].mean(), 1)
)

# Clustering
st.subheader("🤖 K-Means Customer Clustering")

features = df[["Annual Income (k$)", "Spending Score (1-100)"]]

scaler = StandardScaler()
scaled_data = scaler.fit_transform(features)

clusters = st.slider("Select Number of Clusters", 2, 8, 5)

model = KMeans(
    n_clusters=clusters,
    random_state=42,
    n_init=10
)

df["Cluster"] = model.fit_predict(scaled_data)

# Visualization
st.subheader("📈 Customer Segmentation Graph")

fig, ax = plt.subplots(figsize=(10, 5))

sns.scatterplot(
    data=df,
    x="Annual Income (k$)",
    y="Spending Score (1-100)",
    hue="Cluster",
    palette="viridis",
    s=100,
    ax=ax
)

ax.set_title("Customer Segments")
st.pyplot(fig)
plt.close(fig)

# Cluster summary
st.subheader("📋 Cluster Analysis")

summary = df.groupby("Cluster").agg({
    "CustomerID": "count",
    "Age": "mean",
    "Annual Income (k$)": "mean",
    "Spending Score (1-100)": "mean"
}).rename(columns={"CustomerID": "Customer Count"})

st.dataframe(summary.round(2), use_container_width=True)

# Download result
csv = df.to_csv(index=False).encode("utf-8")

st.download_button(
    "⬇️ Download Segmented Customer Data",
    data=csv,
    file_name="customer_segments.csv",
    mime="text/csv"
)

st.success("Customer segmentation completed successfully!")
