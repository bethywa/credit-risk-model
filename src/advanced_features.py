# src/advanced_features.py
"""
ADVANCED FEATURE ENGINEERING for Credit Risk Model
Creates 50+ customer-level features from raw transaction data.
"""
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedFeatureEngineer:
    """
    Creates comprehensive customer behavior features for credit risk modeling.
    """
    
    def __init__(self, snapshot_date=None):
        """
        Initialize feature engineer.
        
        Parameters:
        -----------
        snapshot_date : datetime
            Reference date for recency calculations
        """
        self.snapshot_date = snapshot_date
        self.customer_features = None
        
        print("✅ AdvancedFeatureEngineer initialized")
        print("   Will create 50+ customer behavior features")
    
    def load_raw_data(self, data_path):
        """Load raw transaction data."""
        print(f"📂 Loading raw data from: {data_path}")
        self.df = pd.read_csv(data_path, parse_dates=['TransactionStartTime'])
        
        # Set snapshot date for recency
        if self.snapshot_date is None:
            self.snapshot_date = self.df['TransactionStartTime'].max() + pd.Timedelta(days=1)
        
        print(f"   Data shape: {self.df.shape}")
        print(f"   Date range: {self.df['TransactionStartTime'].min()} to {self.df['TransactionStartTime'].max()}")
        print(f"   Snapshot date: {self.snapshot_date}")
        
        return self.df
    
    def engineer_all_features(self):
        """
        Create ALL advanced features (50+ features).
        
        Returns:
        --------
        features_df : DataFrame
            DataFrame with 50+ features per customer
        """
        print("\n" + "="*70)
        print("🛠️ ENGINEERING 50+ ADVANCED FEATURES")
        print("="*70)
        
        # Initialize empty DataFrame for features
        customer_ids = self.df['CustomerId'].unique()
        features_df = pd.DataFrame({'CustomerId': customer_ids})
        
        # Create feature categories
        print("\n1. Creating RFM Features...")
        features_df = self._create_rfm_features(features_df)
        
        print("\n2. Creating Spending Behavior Features...")
        features_df = self._create_spending_features(features_df)
        
        print("\n3. Creating Temporal Pattern Features...")
        features_df = self._create_temporal_features(features_df)
        
        print("\n4. Creating Product Behavior Features...")
        features_df = self._create_product_features(features_df)
        
        print("\n5. Creating Channel & Payment Features...")
        features_df = self._create_channel_features(features_df)
        
        print("\n6. Creating Derived & Risk Indicator Features...")
        features_df = self._create_derived_features(features_df)
        
        print("\n7. Creating Fraud & Risk Correlation Features...")
        features_df = self._create_fraud_features(features_df)
        
        # Final processing
        print("\n8. Finalizing features...")
        features_df = self._finalize_features(features_df)
        
        self.customer_features = features_df
        
        print(f"\n✅ Feature engineering complete!")
        print(f"   Total customers: {len(features_df)}")
        print(f"   Total features created: {features_df.shape[1] - 1} (excluding CustomerId)")
        
        # Show feature categories
        self._show_feature_summary(features_df)
        
        return features_df
    
    def _create_rfm_features(self, features_df):
        """Create enhanced RFM features (15+ features)."""
        # Group by customer
        grouped = self.df.groupby('CustomerId')
        
        # Recency features
        last_transaction = grouped['TransactionStartTime'].max()
        features_df['recency_days'] = (self.snapshot_date - last_transaction).dt.days
        
        # Multiple recency metrics
        features_df['recency_weeks'] = features_df['recency_days'] / 7
        features_df['recency_category'] = pd.cut(features_df['recency_days'], 
                                                bins=[0, 7, 30, 90, 365, np.inf],
                                                labels=['very_active', 'active', 'inactive', 'dormant', 'lost'])
        features_df['is_active_30days'] = (features_df['recency_days'] <= 30).astype(int)
        features_df['is_active_7days'] = (features_df['recency_days'] <= 7).astype(int)
        
        # Frequency features
        features_df['transaction_count'] = grouped.size()
        features_df['transaction_frequency'] = grouped['TransactionId'].count()
        
        # Calculate transactions per time unit
        first_transaction = grouped['TransactionStartTime'].min()
        customer_tenure = (self.snapshot_date - first_transaction).dt.days
        features_df['customer_tenure_days'] = customer_tenure
        
        # Avoid division by zero
        features_df['transactions_per_day'] = np.where(
            customer_tenure > 0,
            features_df['transaction_count'] / customer_tenure,
            0
        )
        features_df['transactions_per_week'] = features_df['transactions_per_day'] * 7
        features_df['transactions_per_month'] = features_df['transactions_per_day'] * 30
        
        # Monetary features
        features_df['total_amount'] = grouped['Amount'].sum()
        features_df['avg_transaction_amount'] = grouped['Amount'].mean()
        features_df['median_transaction_amount'] = grouped['Amount'].median()
        features_df['max_transaction_amount'] = grouped['Amount'].max()
        features_df['min_transaction_amount'] = grouped['Amount'].min()
        features_df['std_transaction_amount'] = grouped['Amount'].std()
        
        # Handle NaN for std
        features_df['std_transaction_amount'] = features_df['std_transaction_amount'].fillna(0)
        
        # Spending power indicators
        features_df['spending_range'] = features_df['max_transaction_amount'] - features_df['min_transaction_amount']
        features_df['amount_variability'] = features_df['std_transaction_amount'] / (features_df['avg_transaction_amount'].abs() + 1e-6)
        
        return features_df
    
    def _create_spending_features(self, features_df):
        """Create spending behavior features (10+ features)."""
        grouped = self.df.groupby('CustomerId')
        
        # Positive vs negative amounts (purchases vs refunds)
        positive_transactions = self.df[self.df['Amount'] > 0].groupby('CustomerId')
        negative_transactions = self.df[self.df['Amount'] < 0].groupby('CustomerId')
        
        # Purchase behavior
        features_df['purchase_count'] = positive_transactions.size()
        features_df['total_purchase_amount'] = positive_transactions['Amount'].sum()
        features_df['avg_purchase_amount'] = positive_transactions['Amount'].mean()
        
        # Refund behavior
        features_df['refund_count'] = negative_transactions.size()
        features_df['total_refund_amount'] = negative_transactions['Amount'].sum().abs()
        
        # Calculate ratios
        features_df['refund_ratio'] = features_df['refund_count'] / (features_df['transaction_count'] + 1e-6)
        features_df['refund_amount_ratio'] = features_df['total_refund_amount'] / (features_df['total_purchase_amount'].abs() + 1e-6)
        
        # Large transaction indicators
        large_threshold = self.df['Amount'].quantile(0.75)  # Top 25%
        large_transactions = self.df[self.df['Amount'] > large_threshold].groupby('CustomerId')
        
        features_df['large_transaction_count'] = large_transactions.size()
        features_df['large_transaction_ratio'] = features_df['large_transaction_count'] / (features_df['transaction_count'] + 1e-6)
        
        # Small transaction indicators
        small_threshold = self.df['Amount'].quantile(0.25)  # Bottom 25%
        small_transactions = self.df[self.df['Amount'] < small_threshold].groupby('CustomerId')
        
        features_df['small_transaction_count'] = small_transactions.size()
        features_df['small_transaction_ratio'] = features_df['small_transaction_count'] / (features_df['transaction_count'] + 1e-6)
        
        # Spending consistency
        features_df['spending_consistency'] = 1 / (features_df['std_transaction_amount'] + 1)
        
        return features_df
    
    def _create_temporal_features(self, features_df):
        """Create time-based pattern features (10+ features)."""
        df = self.df.copy()
        df['transaction_hour'] = df['TransactionStartTime'].dt.hour
        df['transaction_day'] = df['TransactionStartTime'].dt.day
        df['transaction_dayofweek'] = df['TransactionStartTime'].dt.dayofweek
        df['transaction_week'] = df['TransactionStartTime'].dt.isocalendar().week
        df['transaction_month'] = df['TransactionStartTime'].dt.month
        
        grouped = df.groupby('CustomerId')
        
        # Time of day preferences
        features_df['avg_transaction_hour'] = grouped['transaction_hour'].mean()
        features_df['preferred_hour'] = grouped['transaction_hour'].agg(lambda x: x.mode()[0] if not x.mode().empty else np.nan)
        
        # Day of week preferences
        features_df['weekday_transactions'] = grouped.apply(lambda x: (x['transaction_dayofweek'] < 5).sum())
        features_df['weekend_transactions'] = grouped.apply(lambda x: (x['transaction_dayofweek'] >= 5).sum())
        features_df['weekend_ratio'] = features_df['weekend_transactions'] / (features_df['transaction_count'] + 1e-6)
        
        # Time interval features
        def calculate_time_intervals(series):
            if len(series) > 1:
                sorted_times = series.sort_values()
                intervals = sorted_times.diff().dt.total_seconds() / 3600  # hours
                return intervals.mean(), intervals.std()
            return np.nan, np.nan
        
        interval_stats = grouped['TransactionStartTime'].apply(lambda x: calculate_time_intervals(x))
        features_df['avg_hours_between_transactions'] = interval_stats.apply(lambda x: x[0])
        features_df['std_hours_between_transactions'] = interval_stats.apply(lambda x: x[1])
        
        # Regularity score (inverse of std of intervals)
        features_df['transaction_regularity'] = 1 / (features_df['std_hours_between_transactions'] + 1)
        
        # Time of day categories
        def get_time_category(hour):
            if 5 <= hour < 12:
                return 'morning'
            elif 12 <= hour < 17:
                return 'afternoon'
            elif 17 <= hour < 22:
                return 'evening'
            else:
                return 'night'
        
        df['time_category'] = df['transaction_hour'].apply(get_time_category)
        time_pref = df.groupby(['CustomerId', 'time_category']).size().unstack(fill_value=0)
        
        for category in ['morning', 'afternoon', 'evening', 'night']:
            if category in time_pref.columns:
                features_df[f'{category}_transactions'] = time_pref[category]
                features_df[f'{category}_ratio'] = time_pref[category] / (features_df['transaction_count'] + 1e-6)
        
        # Month consistency
        features_df['unique_transaction_months'] = grouped['transaction_month'].nunique()
        features_df['month_consistency'] = features_df['unique_transaction_months'] / (features_df['customer_tenure_days'] / 30 + 1e-6)
        
        return features_df
    
    def _create_product_features(self, features_df):
        """Create product behavior features (8+ features)."""
        grouped = self.df.groupby('CustomerId')
        
        # Category diversity
        features_df['unique_product_categories'] = grouped['ProductCategory'].nunique()
        features_df['unique_products'] = grouped['ProductId'].nunique()
        
        # Calculate category preferences
        category_counts = self.df.groupby(['CustomerId', 'ProductCategory']).size().unstack(fill_value=0)
        
        # Find favorite category
        def get_favorite_category(row):
            if len(row) > 0:
                return row.idxmax()
            return None
        
        features_df['favorite_category'] = category_counts.apply(get_favorite_category, axis=1)
        
        # Calculate concentration in favorite category
        def get_category_concentration(row):
            if len(row) > 0:
                return row.max() / row.sum()
            return 0
        
        features_df['category_concentration'] = category_counts.apply(get_category_concentration, axis=1)
        
        # Calculate entropy (diversity measure)
        from scipy.stats import entropy
        
        def calculate_entropy(row):
            if row.sum() > 0:
                probabilities = row / row.sum()
                return entropy(probabilities)
            return 0
        
        features_df['category_entropy'] = category_counts.apply(calculate_entropy, axis=1)
        
        # Flag for specific categories (e.g., airtime might indicate different behavior)
        if 'airtime' in self.df['ProductCategory'].unique():
            airtime_counts = self.df[self.df['ProductCategory'] == 'airtime'].groupby('CustomerId').size()
            features_df['airtime_transactions'] = features_df['CustomerId'].map(airtime_counts).fillna(0)
            features_df['airtime_ratio'] = features_df['airtime_transactions'] / (features_df['transaction_count'] + 1e-6)
        
        if 'financial_services' in self.df['ProductCategory'].unique():
            fs_counts = self.df[self.df['ProductCategory'] == 'financial_services'].groupby('CustomerId').size()
            features_df['financial_services_transactions'] = features_df['CustomerId'].map(fs_counts).fillna(0)
            features_df['financial_services_ratio'] = features_df['financial_services_transactions'] / (features_df['transaction_count'] + 1e-6)
        
        return features_df
    
    def _create_channel_features(self, features_df):
        """Create channel and payment behavior features (5+ features)."""
        grouped = self.df.groupby('CustomerId')
        
        # Channel diversity
        features_df['unique_channels'] = grouped['ChannelId'].nunique()
        
        # Channel preferences
        channel_counts = self.df.groupby(['CustomerId', 'ChannelId']).size().unstack(fill_value=0)
        
        # Find favorite channel
        def get_favorite_channel(row):
            if len(row) > 0:
                return row.idxmax()
            return None
        
        features_df['favorite_channel'] = channel_counts.apply(get_favorite_channel, axis=1)
        
        # Channel concentration
        def get_channel_concentration(row):
            if len(row) > 0:
                return row.max() / row.sum()
            return 0
        
        features_df['channel_concentration'] = channel_counts.apply(get_channel_concentration, axis=1)
        
        # Provider diversity
        features_df['unique_providers'] = grouped['ProviderId'].nunique()
        
        return features_df
    
    def _create_derived_features(self, features_df):
        """Create derived composite features (8+ features)."""
        # RFM Score (combine recency, frequency, monetary)
        # Normalize each component
        from sklearn.preprocessing import MinMaxScaler
        
        # Recency score (lower recency = better)
        features_df['recency_score'] = 1 - MinMaxScaler().fit_transform(
            features_df[['recency_days']]
        ).flatten()
        
        # Frequency score (higher frequency = better)
        features_df['frequency_score'] = MinMaxScaler().fit_transform(
            features_df[['transactions_per_month']]
        ).flatten()
        
        # Monetary score (higher monetary = better)
        features_df['monetary_score'] = MinMaxScaler().fit_transform(
            features_df[['total_amount'].abs().values.reshape(-1, 1)]
        ).flatten()
        
        # Combined RFM score
        features_df['rfm_score'] = (
            features_df['recency_score'] * 0.4 +
            features_df['frequency_score'] * 0.3 +
            features_df['monetary_score'] * 0.3
        )
        
        # Customer engagement score
        features_df['engagement_score'] = (
            features_df['frequency_score'] * features_df['monetary_score']
        ) / (features_df['recency_score'] + 1e-6)
        
        # Customer value score
        features_df['customer_value_score'] = (
            features_df['total_amount'].abs() * features_df['transactions_per_month']
        ) / (features_df['recency_days'] + 1e-6)
        
        # Risk indicator scores
        features_df['high_refund_risk'] = (features_df['refund_ratio'] > 0.2).astype(int)
        features_df['inactivity_risk'] = (features_df['recency_days'] > 60).astype(int)
        features_df['low_frequency_risk'] = (features_df['transactions_per_month'] < 2).astype(int)
        features_df['high_variability_risk'] = (features_df['amount_variability'] > 2).astype(int)
        
        # Composite risk score
        features_df['composite_risk_score'] = (
            features_df['high_refund_risk'] * 0.25 +
            features_df['inactivity_risk'] * 0.25 +
            features_df['low_frequency_risk'] * 0.25 +
            features_df['high_variability_risk'] * 0.25
        )
        
        # Customer lifecycle stage
        def get_lifecycle_stage(row):
            if row['recency_days'] <= 30 and row['transactions_per_month'] >= 4:
                return 'active'
            elif row['recency_days'] <= 90:
                return 'warm'
            elif row['recency_days'] <= 180:
                return 'cold'
            else:
                return 'lost'
        
        features_df['lifecycle_stage'] = features_df.apply(get_lifecycle_stage, axis=1)
        
        return features_df
    
    def _create_fraud_features(self, features_df):
        """Create fraud correlation features (5+ features)."""
        if 'FraudResult' in self.df.columns:
            fraud_grouped = self.df[self.df['FraudResult'] == 1].groupby('CustomerId')
            
            # Fraud history
            features_df['fraud_transaction_count'] = fraud_grouped.size()
            features_df['fraud_ratio'] = features_df['fraud_transaction_count'] / (features_df['transaction_count'] + 1e-6)
            
            # Fraud amount
            fraud_amounts = fraud_grouped['Amount'].agg(['sum', 'mean', 'max']).add_prefix('fraud_')
            features_df = features_df.merge(fraud_amounts, left_on='CustomerId', right_index=True, how='left')
            
            # Fill NaN for customers with no fraud
            fraud_cols = ['fraud_transaction_count', 'fraud_ratio', 'fraud_sum', 'fraud_mean', 'fraud_max']
            for col in fraud_cols:
                if col in features_df.columns:
                    features_df[col] = features_df[col].fillna(0)
            
            # Fraud risk flag
            features_df['has_fraud_history'] = (features_df['fraud_transaction_count'] > 0).astype(int)
        
        return features_df
    
    def _finalize_features(self, features_df):
        """Final processing of all features."""
        # Fill remaining NaN values
        numeric_cols = features_df.select_dtypes(include=[np.number]).columns
        features_df[numeric_cols] = features_df[numeric_cols].fillna(0)
        
        # Remove any infinite values
        features_df.replace([np.inf, -np.inf], 0, inplace=True)
        
        # Drop CustomerId for now (will add back)
        customer_ids = features_df['CustomerId']
        features_without_id = features_df.drop(columns=['CustomerId'])
        
        # Remove constant columns
        constant_cols = [col for col in features_without_id.columns if features_without_id[col].nunique() <= 1]
        if constant_cols:
            print(f"   Removing {len(constant_cols)} constant columns")
            features_without_id = features_without_id.drop(columns=constant_cols)
        
        # Add CustomerId back
        features_without_id['CustomerId'] = customer_ids.values
        
        # Reorder columns with CustomerId first
        cols = ['CustomerId'] + [col for col in features_without_id.columns if col != 'CustomerId']
        features_df = features_without_id[cols]
        
        return features_df
    
    def _show_feature_summary(self, features_df):
        """Display summary of created features."""
        print("\n📊 FEATURE CATEGORIES CREATED:")
        print("-" * 50)
        
        # Count features by category
        feature_categories = {
            'RFM Features': [col for col in features_df.columns if 'recency' in col.lower() or 
                           'frequency' in col.lower() or 'monetary' in col.lower() or
                           'rfm' in col.lower() or 'tenure' in col.lower()],
            'Spending Behavior': [col for col in features_df.columns if 'amount' in col.lower() or 
                                'spend' in col.lower() or 'purchase' in col.lower() or
                                'refund' in col.lower()],
            'Temporal Patterns': [col for col in features_df.columns if 'hour' in col.lower() or 
                                'day' in col.lower() or 'week' in col.lower() or
                                'month' in col.lower() or 'time' in col.lower() or
                                'regular' in col.lower()],
            'Product Behavior': [col for col in features_df.columns if 'product' in col.lower() or 
                               'category' in col.lower() or 'entropy' in col.lower()],
            'Channel/Provider': [col for col in features_df.columns if 'channel' in col.lower() or 
                               'provider' in col.lower()],
            'Risk & Fraud': [col for col in features_df.columns if 'risk' in col.lower() or 
                           'fraud' in col.lower() or 'inactiv' in col.lower()],
            'Derived Scores': [col for col in features_df.columns if 'score' in col.lower() or 
                             'engagement' in col.lower() or 'value' in col.lower()]
        }
        
        for category, cols in feature_categories.items():
            if cols:
                print(f"  {category}: {len(cols)} features")
                # Show 3 example features
                example_features = cols[:3]
                print(f"    Examples: {', '.join(example_features)}")
        
        print("-" * 50)
        print(f"  TOTAL FEATURES: {features_df.shape[1] - 1}")
    
    def save_features(self, output_path):
        """Save engineered features to CSV."""
        if self.customer_features is not None:
            self.customer_features.to_csv(output_path, index=False)
            print(f"\n💾 Features saved to: {output_path}")
        else:
            print("❌ No features to save. Run engineer_all_features() first.")


# Main execution function
def create_advanced_features_pipeline(input_path, output_path):
    """
    Complete pipeline for advanced feature engineering.
    
    Parameters:
    -----------
    input_path : str
        Path to raw transaction data
    output_path : str
        Path to save engineered features
    """
    print("="*70)
    print("ADVANCED FEATURE ENGINEERING PIPELINE")
    print("="*70)
    
    # Initialize engineer
    engineer = AdvancedFeatureEngineer()
    
    # Load raw data
    raw_df = engineer.load_raw_data(input_path)
    
    # Engineer all features
    features_df = engineer.engineer_all_features()
    
    # Save features
    engineer.save_features(output_path)
    
    # Show sample of features
    print("\n📋 Sample of engineered features:")
    print(features_df.head())
    
    print(f"\n✅ Pipeline complete!")
    print(f"   Input:  {raw_df.shape[0]:,} transactions")
    print(f"   Output: {features_df.shape[0]:,} customers × {features_df.shape[1]-1} features")
    
    return features_df


if __name__ == "__main__":
    # Example usage
    input_file = "data/raw/transaction_data.csv"
    output_file = "data/processed/advanced_customer_features.csv"
    
    try:
        features = create_advanced_features_pipeline(input_file, output_file)
    except FileNotFoundError:
        print(f"❌ File not found: {input_file}")
        print("Please update the file path in advanced_features.py")
    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()