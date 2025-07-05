# Mall Customer Segmentation using K-Means

This project uses the K-Means Clustering algorithm to group customers of a retail mall based on their purchase patterns.

## 📊 Dataset

**Mall_Customers.csv**  
Synthetic dataset with 100 entries and the following features:
- CustomerID
- Gender
- Age
- Annual Income (k$)
- Spending Score (1-100)

## 🧠 Algorithm

- Feature Scaling using `StandardScaler`
- K-Means clustering with `sklearn`
- Elbow Method to determine optimal k
- Visualization of clusters

## 📈 Output

- Elbow plot to determine optimal number of clusters
- Cluster visualization in 2D space

## 💻 Run the Project

```bash
python kmeans_mall_customers.py
