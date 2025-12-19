# src/data_processing.py
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import warnings
warnings.filterwarnings('ignore')

class DataLoader:
    """Loads and initializes the raw transaction data."""
    def __init__(self, filepath):
        self.filepath = filepath
        
    def load(self):
        df = pd.read_csv(self.filepath, parse_dates=['TransactionStartTime'])
        print(f"✅ Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        return df


class AggregateCustomerFeatures(BaseEstimator, TransformerMixin):
    """
    Creates customer-level RFM and behavioral aggregates from transaction data.
    This MUST be run before other transformations as it changes dataframe granularity.
    """
    def __init__(self, customer_id='CustomerId'):
        self.customer_id = customer_id
        self.agg_features_ = None
        
    def fit(self, X, y=None):
        # Calculate aggregates from training data
        self.agg_features_ = self._calculate_aggregates(X)
        return self
    
    def transform(self, X):
        # Merge pre-calculated aggregates back to transactions
        X_transformed = X.merge(self.agg_features_, 
                               on=self.customer_id, 
                               how='left')
        print(f"✅ Customer aggregates merged. Shape: {X_transformed.shape}")
        return X_transformed
    
    def _calculate_aggregates(self, df):
        """Internal method to compute RFM and other aggregates."""
        # Use MEDIAN for Monetary due to extreme outliers
        agg_funcs = {
            'Amount': ['count', 'median', 'std'],
            'Value': ['median'],
            'TransactionStartTime': ['max', 'min']
        }
        
        agg_df = df.groupby(self.customer_id).agg(agg_funcs).round(2)
        agg_df.columns = ['_'.join(col).strip() for col in agg_df.columns.values]
        
        # Rename for clarity
        agg_df = agg_df.rename(columns={
            'Amount_count': 'transaction_count',
            'Amount_median': 'amount_median',
            'Amount_std': 'amount_std',
            'Value_median': 'value_median',
            'TransactionStartTime_max': 'last_transaction',
            'TransactionStartTime_min': 'first_transaction'
        })
        
        # Calculate Recency (days since last transaction)
        snapshot_date = df['TransactionStartTime'].max() + pd.Timedelta(days=1)
        agg_df['recency_days'] = (snapshot_date - agg_df['last_transaction']).dt.days
        
        # Calculate Frequency (transactions per month)
        transaction_span = (agg_df['last_transaction'] - agg_df['first_transaction']).dt.days
        agg_df['frequency_per_month'] = np.where(
            transaction_span > 0,
            agg_df['transaction_count'] / (transaction_span / 30),
            0  # Avoid division by zero for single-transaction customers
        )
        
        return agg_df.reset_index()


class OutlierHandler(BaseEstimator, TransformerMixin):
    """
    Handles extreme values in Amount/Value using capping and log transformation.
    Based on EDA findings: 25.5% outliers with bounds [-4325, 7075].
    """
    def __init__(self, lower_bound=-4325, upper_bound=7075):
        self.lower_bound = lower_bound
        self.upper_bound = upper_bound
        
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        
        # 1. Cap outliers
        X['Amount_capped'] = X['Amount'].clip(
            lower=self.lower_bound, 
            upper=self.upper_bound
        )
        X['Value_capped'] = X['Value'].clip(
            lower=0,  # Value is absolute amount
            upper=self.upper_bound
        )
        
        # 2. Log transformation (add 1 to handle negative/zero values)
        X['Amount_log'] = np.log1p(X['Amount_capped'] - X['Amount_capped'].min() + 1)
        X['Value_log'] = np.log1p(X['Value_capped'])
        
        print(f"✅ Outliers handled. Original range: [{X['Amount'].min():.0f}, {X['Amount'].max():.0f}]")
        print(f"   Capped range: [{X['Amount_capped'].min():.0f}, {X['Amount_capped'].max():.0f}]")
        
        return X


class DateTimeFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extracts temporal features from TransactionStartTime."""
    def fit(self, X, y=None):
        return self
    
    def transform(self, X):
        X = X.copy()
        if 'TransactionStartTime' in X.columns:
            X['transaction_hour'] = X['TransactionStartTime'].dt.hour
            X['transaction_day'] = X['TransactionStartTime'].dt.day
            X['transaction_month'] = X['TransactionStartTime'].dt.month
            X['transaction_dayofweek'] = X['TransactionStartTime'].dt.dayofweek
            X['is_weekend'] = X['transaction_dayofweek'].isin([5, 6]).astype(int)
            print("✅ DateTime features extracted")
        return X


def create_feature_pipeline():
    """
    Creates the complete sklearn pipeline for feature engineering.
    Note: Run AggregateCustomerFeatures FIRST on raw data before this pipeline.
    """
    # Define column groups
    numeric_features = [
        'amount_median', 'amount_std', 'value_median',
        'recency_days', 'frequency_per_month',
        'Amount_log', 'Value_log'
    ]
    
    categorical_features = ['ProductCategory', 'ChannelId', 'CountryCode']
    
    temporal_features = [
        'transaction_hour', 'transaction_day', 
        'transaction_month', 'transaction_dayofweek'
    ]
    
    # Create preprocessing pipelines for different feature types
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])
    
    temporal_transformer = Pipeline(steps=[
        ('scaler', StandardScaler())
    ])
    
    # Combine all preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features),
            ('temp', temporal_transformer, temporal_features)
        ],
        remainder='drop'  # Drop columns not explicitly handled
    )
    
    # Main pipeline
    pipeline = Pipeline(steps=[
        ('outlier_handler', OutlierHandler()),
        ('datetime_extractor', DateTimeFeatureExtractor()),
        ('preprocessor', preprocessor)
    ])
    
    return pipeline


def process_data(input_path, output_path=None):
    """
    Main function to run the complete feature engineering process.
    
    Parameters:
    -----------
    input_path : str
        Path to raw transaction data CSV
    output_path : str, optional
        Path to save processed data
        
    Returns:
    --------
    X_processed : numpy array
        Processed feature matrix
    feature_names : list
        Names of all processed features
    customer_ids : array
        Customer IDs for reference
    """
    # 1. Load data
    loader = DataLoader(input_path)
    raw_df = loader.load()
    
    # 2. Create customer aggregates (FIRST STEP - changes granularity)
    print("\n" + "="*50)
    print("STEP 1: Creating customer-level aggregates")
    print("="*50)
    aggregator = AggregateCustomerFeatures()
    df_with_aggregates = aggregator.fit_transform(raw_df)
    
    # Save customer IDs before processing
    customer_ids = df_with_aggregates['CustomerId'].values
    
    # 3. Apply feature engineering pipeline
    print("\n" + "="*50)
    print("STEP 2: Applying feature engineering pipeline")
    print("="*50)
    pipeline = create_feature_pipeline()
    
    # Fit and transform
    X_processed = pipeline.fit_transform(df_with_aggregates)
    
    # 4. Get feature names for interpretability
    feature_names = []
    
    # Get numeric feature names
    feature_names.extend([
        'amount_median', 'amount_std', 'value_median',
        'recency_days', 'frequency_per_month',
        'Amount_log', 'Value_log'
    ])
    
    # Get categorical feature names (from one-hot encoding)
    categorical_transformer = pipeline.named_steps['preprocessor'].transformers_[1][1]
    onehot = categorical_transformer.named_steps['onehot']
    
    # Get original categories
    cat_imputer = categorical_transformer.named_steps['imputer']
    # Note: In practice, you'd need to fit this separately to get categories
    # For now, we'll create placeholder names
    
    # Get temporal feature names
    feature_names.extend([
        'transaction_hour', 'transaction_day', 
        'transaction_month', 'transaction_dayofweek'
    ])
    
    print(f"\n✅ Processing complete!")
    print(f"   Input shape: {raw_df.shape}")
    print(f"   Output shape: {X_processed.shape}")
    print(f"   Number of features created: {X_processed.shape[1]}")
    
    # 5. Save processed data if output path provided
    if output_path:
        # Create a DataFrame for saving
        processed_df = pd.DataFrame(
            X_processed, 
            columns=[f'feature_{i}' for i in range(X_processed.shape[1])]
        )
        processed_df['CustomerId'] = customer_ids
        processed_df.to_csv(output_path, index=False)
        print(f"💾 Processed data saved to: {output_path}")
    
    return X_processed, feature_names, customer_ids


# For standalone execution
if __name__ == "__main__":
    # Example usage
    input_file = "../data/raw/transaction_data.csv"  # Update this path
    output_file = "../data/processed/features_engineered.csv"
    
    try:
        X_processed, feature_names, customer_ids = process_data(
            input_file, output_file
        )
    except FileNotFoundError:
        print("❌ Input file not found. Please update the path in data_processing.py")
    except Exception as e:
        print(f"❌ Error during processing: {str(e)}")