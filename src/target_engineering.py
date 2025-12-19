"""
src/target_engineering.py
Create proxy target variable for credit risk using RFM clustering.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class RFMClustering:
    """
    Creates a proxy credit risk target using K-Means clustering on RFM metrics.
    
    Based on Task 4 requirements:
    1. Calculate RFM metrics from transaction history
    2. Scale features appropriately
    3. Cluster into 3 groups using K-Means
    4. Identify high-risk cluster (least engaged)
    5. Create binary is_high_risk target
    """
    
    def __init__(self, n_clusters=3, random_state=42):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.scaler = RobustScaler()  # Robust to outliers found in EDA
        self.kmeans = None
        self.cluster_labels = None
        self.rfm_data = None
        self.high_risk_cluster = None
        
    def calculate_rfm_from_transactions(self, df, customer_id='CustomerId'):
        """
        Calculate RFM metrics directly from raw transaction data.
        Alternative: Use pre-calculated aggregates from Task 3.
        """
        # Use snapshot date as day after last transaction
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
        
        # Calculate RFM
        rfm = df.groupby(customer_id).agg({
            'TransactionStartTime': lambda x: (snapshot_date - x.max()).days,  # Recency
            'TransactionId': 'count',                                          # Frequency
            'Amount': 'sum'                                                    # Monetary (Use MEDIAN if outliers are extreme)
        }).reset_index()
        
        rfm.columns = [customer_id, 'recency', 'frequency', 'monetary']
        
        # Handle negative monetary values (refunds)
        rfm['monetary'] = rfm['monetary'].abs()
        
        return rfm
    
    def calculate_rfm_from_aggregates(self, df, customer_id='CustomerId'):
        """
        Calculate RFM from the aggregate features created in Task 3.
        This is the recommended approach since you already have these features.
        """
        # Identify RFM columns from your Task 3 output
        rfm_cols = {
            'recency': 'recency_days',
            'frequency': 'frequency_per_month',  # Or 'transaction_count'
            'monetary': 'value_median'          # Use MEDIAN as per EDA recommendation
        }
        
        # Check which columns exist
        available_cols = {}
        for rfm_name, col_name in rfm_cols.items():
            if col_name in df.columns:
                available_cols[rfm_name] = col_name
            else:
                print(f"⚠️ Warning: {col_name} not found. Using alternative...")
        
        # Create RFM DataFrame
        rfm = df[[customer_id]].drop_duplicates().copy()
        
        for rfm_name, col_name in available_cols.items():
            # Get the latest value per customer
            latest_vals = df.groupby(customer_id)[col_name].last().reset_index()
            rfm = rfm.merge(latest_vals, on=customer_id, how='left')
            rfm = rfm.rename(columns={col_name: rfm_name})
        
        print(f"✅ RFM metrics created: {list(rfm.columns)}")
        return rfm
    
    def prepare_for_clustering(self, rfm_df, features=['recency', 'frequency', 'monetary']):
        """
        Prepare RFM data for clustering with appropriate preprocessing.
        """
        # 1. Handle missing values
        self.rfm_data = rfm_df.copy()
        
        for feature in features:
            if feature in self.rfm_data.columns:
                # Fill missing with median
                self.rfm_data[feature] = self.rfm_data[feature].fillna(
                    self.rfm_data[feature].median()
                )
            else:
                print(f"⚠️ Feature {feature} not available. Using alternatives.")
        
        # 2. Apply logarithmic transformation for skewed features (based on EDA)
        if 'monetary' in self.rfm_data.columns:
            self.rfm_data['monetary_log'] = np.log1p(self.rfm_data['monetary'])
            features.append('monetary_log')
            features.remove('monetary')
        
        if 'frequency' in self.rfm_data.columns:
            self.rfm_data['frequency_log'] = np.log1p(self.rfm_data['frequency'])
            features.append('frequency_log')
            features.remove('frequency')
        
        # 3. Scale features (RobustScaler handles outliers better)
        X = self.rfm_data[features].copy()
        X_scaled = self.scaler.fit_transform(X)
        
        print(f"✅ Prepared {len(features)} features for clustering")
        print(f"   Features: {features}")
        
        return X_scaled, features
    
    def perform_clustering(self, X_scaled):
        """
        Perform K-Means clustering and analyze results.
        """
        # Initialize and fit K-Means
        self.kmeans = KMeans(
            n_clusters=self.n_clusters,
            random_state=self.random_state,
            n_init=10  # Important for stability
        )
        
        self.cluster_labels = self.kmeans.fit_predict(X_scaled)
        
        # Calculate silhouette score (measure of clustering quality)
        if len(np.unique(self.cluster_labels)) > 1:
            score = silhouette_score(X_scaled, self.cluster_labels)
            print(f"✅ Clustering complete. Silhouette Score: {score:.3f}")
        else:
            print("✅ Clustering complete.")
        
        # Add cluster labels to RFM data
        self.rfm_data['cluster'] = self.cluster_labels
        
        return self.cluster_labels
    
    def analyze_clusters(self, features=['recency', 'frequency', 'monetary']):
        """
        Analyze cluster characteristics to identify high-risk group.
        """
        if self.rfm_data is None or 'cluster' not in self.rfm_data.columns:
            raise ValueError("Clustering must be performed first.")
        
        # Calculate cluster statistics
        cluster_stats = self.rfm_data.groupby('cluster')[features].agg(['mean', 'median', 'std'])
        
        print("\n" + "="*60)
        print("CLUSTER ANALYSIS")
        print("="*60)
        
        # Simplified display
        for feature in features:
            if feature in self.rfm_data.columns:
                print(f"\n📊 {feature.upper()} by Cluster (Median):")
                median_vals = self.rfm_data.groupby('cluster')[feature].median()
                for cluster, val in median_vals.items():
                    print(f"  Cluster {cluster}: {val:.2f}")
        
        # Identify high-risk cluster based on business logic
        # High-risk typically means: High Recency (inactive), Low Frequency, Low Monetary
        print("\n🔍 Identifying High-Risk Cluster...")
        
        # Calculate risk score for each cluster
        cluster_risk_scores = {}
        
        for cluster in sorted(self.rfm_data['cluster'].unique()):
            cluster_data = self.rfm_data[self.rfm_data['cluster'] == cluster]
            
            # Normalize metrics (higher = more risky)
            risk_factors = []
            
            if 'recency' in features:
                # Higher recency (more days since last purchase) = more risky
                recency_score = cluster_data['recency'].median() / self.rfm_data['recency'].max()
                risk_factors.append(recency_score)
            
            if 'frequency' in features:
                # Lower frequency = more risky (inverse relationship)
                freq_score = 1 - (cluster_data['frequency'].median() / self.rfm_data['frequency'].max())
                risk_factors.append(freq_score)
            
            if 'monetary' in features:
                # Lower monetary value = more risky (inverse relationship)
                monetary_score = 1 - (cluster_data['monetary'].median() / self.rfm_data['monetary'].max())
                risk_factors.append(monetary_score)
            
            # Average risk factors
            cluster_risk_scores[cluster] = np.mean(risk_factors) if risk_factors else 0
        
        # Cluster with highest risk score is high-risk
        self.high_risk_cluster = max(cluster_risk_scores, key=cluster_risk_scores.get)
        
        print("\n📈 Cluster Risk Scores:")
        for cluster, score in cluster_risk_scores.items():
            risk_level = "🚨 HIGH-RISK" if cluster == self.high_risk_cluster else "✅ Low-Risk"
            print(f"  Cluster {cluster}: {score:.3f} - {risk_level}")
        
        # Show composition
        cluster_sizes = self.rfm_data['cluster'].value_counts().sort_index()
        high_risk_pct = (cluster_sizes[self.high_risk_cluster] / len(self.rfm_data)) * 100
        
        print(f"\n👥 Cluster Composition:")
        for cluster in sorted(cluster_sizes.index):
            size = cluster_sizes[cluster]
            pct = (size / len(self.rfm_data)) * 100
            label = " (HIGH-RISK)" if cluster == self.high_risk_cluster else ""
            print(f"  Cluster {cluster}: {size} customers ({pct:.1f}%){label}")
        
        return self.high_risk_cluster, cluster_stats
    
    def create_target_variable(self):
        """
        Create binary is_high_risk target based on cluster analysis.
        """
        if self.high_risk_cluster is None:
            raise ValueError("Must analyze clusters first to identify high-risk group.")
        
        # Create binary target
        self.rfm_data['is_high_risk'] = (self.rfm_data['cluster'] == self.high_risk_cluster).astype(int)
        
        risk_pct = (self.rfm_data['is_high_risk'].sum() / len(self.rfm_data)) * 100
        print(f"\n🎯 Target Variable Created:")
        print(f"   High-risk customers: {self.rfm_data['is_high_risk'].sum():,} ({risk_pct:.1f}%)")
        print(f"   Low-risk customers: {(self.rfm_data['is_high_risk'] == 0).sum():,} ({100-risk_pct:.1f}%)")
        
        return self.rfm_data[['CustomerId', 'is_high_risk', 'cluster']]
    
    def visualize_clusters(self, features=['recency', 'frequency', 'monetary']):
        """
        Create visualizations to understand cluster characteristics.
        """
        if 'cluster' not in self.rfm_data.columns:
            raise ValueError("Clustering must be performed first.")
        
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        fig.suptitle('RFM Cluster Analysis for Credit Risk Proxy', fontsize=16, y=1.02)
        
        # Plot 1: Recency vs Frequency
        scatter = axes[0, 0].scatter(
            self.rfm_data['recency'],
            self.rfm_data['frequency'],
            c=self.rfm_data['cluster'],
            cmap='viridis',
            alpha=0.6,
            s=20
        )
        axes[0, 0].set_xlabel('Recency (days since last purchase)')
        axes[0, 0].set_ylabel('Frequency')
        axes[0, 0].set_title('Recency vs Frequency')
        
        # Plot 2: Monetary vs Frequency
        axes[0, 1].scatter(
            self.rfm_data['monetary'],
            self.rfm_data['frequency'],
            c=self.rfm_data['cluster'],
            cmap='viridis',
            alpha=0.6,
            s=20
        )
        axes[0, 1].set_xlabel('Monetary Value')
        axes[0, 1].set_ylabel('Frequency')
        axes[0, 1].set_title('Monetary vs Frequency')
        axes[0, 1].set_xscale('log')  # Log scale for monetary
        
        # Plot 3: Recency vs Monetary
        axes[0, 2].scatter(
            self.rfm_data['recency'],
            self.rfm_data['monetary'],
            c=self.rfm_data['cluster'],
            cmap='viridis',
            alpha=0.6,
            s=20
        )
        axes[0, 2].set_xlabel('Recency')
        axes[0, 2].set_ylabel('Monetary Value')
        axes[0, 2].set_title('Recency vs Monetary')
        axes[0, 2].set_yscale('log')  # Log scale for monetary
        
        # Plot 4: Cluster distribution (bar chart)
        cluster_counts = self.rfm_data['cluster'].value_counts().sort_index()
        colors = ['red' if i == self.high_risk_cluster else 'green' for i in cluster_counts.index]
        axes[1, 0].bar(cluster_counts.index.astype(str), cluster_counts.values, color=colors)
        axes[1, 0].set_xlabel('Cluster')
        axes[1, 0].set_ylabel('Number of Customers')
        axes[1, 0].set_title('Cluster Distribution')
        
        # Plot 5: Risk distribution (pie chart)
        risk_counts = self.rfm_data['is_high_risk'].value_counts()
        labels = ['Low Risk', 'High Risk']
        colors_pie = ['lightgreen', 'lightcoral']
        axes[1, 1].pie(risk_counts.values, labels=labels, colors=colors_pie, autopct='%1.1f%%')
        axes[1, 1].set_title('Risk Distribution')
        
        # Plot 6: Feature distributions by cluster (box plot for recency)
        cluster_data = [self.rfm_data[self.rfm_data['cluster'] == i]['recency'] 
                       for i in sorted(self.rfm_data['cluster'].unique())]
        axes[1, 2].boxplot(cluster_data, labels=[f'Cluster {i}' for i in sorted(self.rfm_data['cluster'].unique())])
        axes[1, 2].set_ylabel('Recency (days)')
        axes[1, 2].set_title('Recency Distribution by Cluster')
        axes[1, 2].tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.show()
        
        # Save the figure
        plt.savefig('../reports/rfm_clustering_analysis.png', dpi=150, bbox_inches='tight')
        print("💾 Visualization saved to: ../reports/rfm_clustering_analysis.png")
    
    def merge_target_with_data(self, original_df, target_df, customer_id='CustomerId'):
        """
        Merge the target variable back to the original dataset.
        """
        # Merge target variable
        merged_df = original_df.merge(
            target_df[[customer_id, 'is_high_risk']],
            on=customer_id,
            how='left'
        )
        
        # Fill any missing values (customers not in clustering) as low risk
        merged_df['is_high_risk'] = merged_df['is_high_risk'].fillna(0).astype(int)
        
        print(f"\n✅ Target variable merged with main dataset")
        print(f"   Original shape: {original_df.shape}")
        print(f"   Merged shape: {merged_df.shape}")
        print(f"   High-risk rows in merged data: {merged_df['is_high_risk'].sum():,}")
        
        return merged_df


def create_proxy_target(input_path, output_path=None, use_existing_aggregates=True):
    """
    Main function to create proxy target variable.
    
    Parameters:
    -----------
    input_path : str
        Path to processed data from Task 3
    output_path : str, optional
        Path to save data with target variable
    use_existing_aggregates : bool
        Whether to use Task 3 aggregates (True) or calculate from scratch (False)
        
    Returns:
    --------
    target_data : DataFrame
        DataFrame with CustomerId and is_high_risk
    merged_data : DataFrame
        Full dataset with target variable merged
    """
    
    print("="*70)
    print("TASK 4: PROXY TARGET VARIABLE ENGINEERING")
    print("="*70)
    
    # 1. Load the processed data from Task 3
    print("\n📂 Loading data...")
    processed_data = pd.read_csv(input_path)
    
    print(f"   Data shape: {processed_data.shape}")
    print(f"   Columns: {list(processed_data.columns)[:10]}...")
    
    # 2. Initialize RFM clustering
    rfm_cluster = RFMClustering(n_clusters=3, random_state=42)
    
    # 3. Calculate or extract RFM metrics
    print("\n📊 Calculating RFM metrics...")
    if use_existing_aggregates and 'recency_days' in processed_data.columns:
        # Use aggregates from Task 3
        rfm_data = rfm_cluster.calculate_rfm_from_aggregates(processed_data)
    else:
        # Calculate from scratch (requires raw transaction data)
        print("   Using raw transaction data for RFM calculation...")
        # You would need to load raw data here
        # raw_data = pd.read_csv('../data/raw/transaction_data.csv')
        # rfm_data = rfm_cluster.calculate_rfm_from_transactions(raw_data)
        raise NotImplementedError("Raw transaction calculation requires raw data path.")
    
    # 4. Prepare for clustering
    print("\n⚙️ Preparing data for clustering...")
    features_for_clustering = ['recency', 'frequency', 'monetary']
    X_scaled, features_used = rfm_cluster.prepare_for_clustering(rfm_data, features_for_clustering)
    
    # 5. Perform clustering
    print("\n🤖 Performing K-Means clustering (k=3)...")
    cluster_labels = rfm_cluster.perform_clustering(X_scaled)
    
    # 6. Analyze clusters and identify high-risk group
    print("\n🔍 Analyzing cluster characteristics...")
    high_risk_cluster, cluster_stats = rfm_cluster.analyze_clusters(['recency', 'frequency', 'monetary'])
    
    # 7. Create target variable
    print("\n🎯 Creating binary target variable...")
    target_data = rfm_cluster.create_target_variable()
    
    # 8. Visualize results
    print("\n📈 Creating visualizations...")
    try:
        rfm_cluster.visualize_clusters()
    except Exception as e:
        print(f"   Visualization skipped: {e}")
    
    # 9. Merge target with original data
    print("\n🔄 Merging target variable with main dataset...")
    merged_data = rfm_cluster.merge_target_with_data(processed_data, target_data)
    
    # 10. Save results
    if output_path:
        print(f"\n💾 Saving data with target variable to: {output_path}")
        merged_data.to_csv(output_path, index=False)
        
        # Also save target mapping separately
        target_mapping_path = output_path.replace('.csv', '_target_mapping.csv')
        target_data.to_csv(target_mapping_path, index=False)
        print(f"💾 Target mapping saved to: {target_mapping_path}")
    
    print("\n" + "="*70)
    print("✅ TASK 4 COMPLETE: Proxy target variable created successfully!")
    print("="*70)
    
    return target_data, merged_data


# For standalone execution
if __name__ == "__main__":
    # Update these paths based on your project structure
    input_file = "data/processed/model_ready_features.csv"  # From Task 3
    output_file = "data/processed/data_with_target.csv"
    
    try:
        target_data, merged_data = create_proxy_target(
            input_file, 
            output_file,
            use_existing_aggregates=True
        )
        
        # Print final summary
        print("\n📋 FINAL SUMMARY:")
        print(f"   Total customers: {len(target_data):,}")
        print(f"   High-risk customers: {target_data['is_high_risk'].sum():,}")
        print(f"   High-risk percentage: {(target_data['is_high_risk'].sum() / len(target_data) * 100):.1f}%")
        print(f"   Data ready for Task 5 (Model Training)")
        
    except FileNotFoundError as e:
        print(f"\n❌ Error: {e}")
        print("Please ensure:")
        print("1. Task 3 is complete and model_ready_features.csv exists")
        print("2. The file path is correct in target_engineering.py")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")