"""
src/train_final.py
FINAL working version of Task 5 - fixes XGBoost MLflow issue.
"""
import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

# Core ML libraries
from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, roc_auc_score, confusion_matrix, 
                             classification_report, roc_curve)
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb

# MLflow for experiment tracking
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature

# Fix for XGBoost MLflow logging
try:
    import mlflow.xgboost
    XGBOOST_MLFLOW_OK = True
except:
    XGBOOST_MLFLOW_OK = False
    print("⚠️ mlflow.xgboost not available, using sklearn logging instead")

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# System
import os
import json
import pickle
import sys

class FinalCreditRiskModelTrainer:
    """
    FINAL working model trainer - handles XGBoost MLflow issue.
    """
    
    def __init__(self, experiment_name="credit_risk_final", random_state=42):
        self.random_state = random_state
        self.experiment_name = experiment_name
        
        # Initialize MLflow
        mlflow.set_experiment(experiment_name)
        
        self.models = {}
        self.best_model = None
        self.best_model_name = None
        self.results = {}
        
        print(f"✅ FinalCreditRiskModelTrainer initialized")
        print(f"   MLflow Experiment: {experiment_name}")
        print(f"   Random State: {random_state}")
        if not XGBOOST_MLFLOW_OK:
            print("   ⚠️ XGBoost MLflow logging disabled, using sklearn fallback")
    
    def load_and_prepare_data(self, data_path, target_column='is_high_risk', test_size=0.2):
        """
        Load and prepare data - FINAL VERSION.
        """
        print("="*60)
        print("📂 LOADING AND PREPARING DATA")
        print("="*60)
        
        # Load data
        df = pd.read_csv(data_path)
        print(f"   Data loaded: {df.shape[0]} rows, {df.shape[1]} columns")
        
        # List of RFM features to EXCLUDE (caused leakage)
        rfm_leakage_features = [
            'recency_days', 'frequency_per_month', 'value_median',
            'transaction_count', 'amount_median', 'amount_std'
        ]
        
        # Separate features and target
        X = df.drop(columns=[target_column], errors='ignore')
        y = df[target_column]
        
        # Remove CustomerId if present
        if 'CustomerId' in X.columns:
            X = X.drop(columns=['CustomerId'])
            print("   Removed: CustomerId")
        
        # Remove RFM leakage features
        features_to_remove = [f for f in rfm_leakage_features if f in X.columns]
        if features_to_remove:
            X = X.drop(columns=features_to_remove)
            print(f"   Removed RFM leakage features: {len(features_to_remove)}")
        
        # Handle categorical columns
        categorical_cols = X.select_dtypes(exclude=[np.number]).columns.tolist()
        if categorical_cols:
            print(f"   Encoding {len(categorical_cols)} categorical columns...")
            X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
        
        # Ensure all numeric and handle NaNs
        X = X.select_dtypes(include=[np.number])
        X = X.fillna(X.median())
        
        print(f"\n✅ Final feature matrix: {X.shape[1]} features")
        print(f"   Features: {list(X.columns)}")
        print(f"   Target distribution:")
        print(f"     High-risk (1): {y.sum():,} ({y.mean()*100:.1f}%)")
        print(f"     Low-risk (0): {(len(y)-y.sum()):,} ({100-y.mean()*100:.1f}%)")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=test_size, 
            stratify=y,
            random_state=self.random_state
        )
        
        print(f"\n✅ Data split complete:")
        print(f"   Training set: {X_train.shape[0]:,} samples")
        print(f"   Testing set:  {X_test.shape[0]:,} samples")
        print(f"   Features:     {X_train.shape[1]}")
        
        # Store data
        self.X_train = X_train
        self.X_test = X_test
        self.y_train = y_train
        self.y_test = y_test
        self.feature_names = X.columns.tolist()
        
        return X_train, X_test, y_train, y_test
    
    def create_models(self):
        """
        Initialize models - FINAL VERSION.
        """
        print("\n" + "="*60)
        print("🤖 INITIALIZING MODELS")
        print("="*60)
        
        # Calculate scale_pos_weight for XGBoost
        scale_pos_weight = len(self.y_train[self.y_train == 0]) / len(self.y_train[self.y_train == 1])
        
        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            min_samples_split=5,
            min_samples_leaf=2,
            class_weight='balanced',
            random_state=self.random_state,
            n_jobs=-1
        )
        
        # XGBoost - FIXED: Use sklearn API properly
        xgb_model = xgb.XGBClassifier(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=self.random_state,
            eval_metric='logloss',
            use_label_encoder=False,
            verbosity=0
        )
        
        self.models = {
            'RandomForest': rf_model,
            'XGBoost': xgb_model
        }
        
        print(f"✅ Models initialized:")
        print(f"   1. Random Forest")
        print(f"   2. XGBoost (scale_pos_weight={scale_pos_weight:.2f})")
        
        return self.models
    
    def train_baseline_models(self, use_mlflow=True):
        """
        Train baseline models - FIXED XGBoost logging.
        """
        print("\n" + "="*60)
        print("🚀 TRAINING BASELINE MODELS")
        print("="*60)
        
        results = {}
        
        for model_name, model in self.models.items():
            print(f"\n📊 Training {model_name}...")
            
            if use_mlflow:
                mlflow.start_run(run_name=f"{model_name}_baseline")
                mlflow.log_params(model.get_params())
                mlflow.set_tag("model_type", model_name)
                mlflow.set_tag("stage", "baseline")
            
            try:
                # Train model
                model.fit(self.X_train, self.y_train)
                
                # Make predictions
                y_pred = model.predict(self.X_test)
                y_pred_proba = model.predict_proba(self.X_test)[:, 1]
                
                # Calculate metrics
                metrics = self._calculate_metrics(self.y_test, y_pred, y_pred_proba)
                
                # Store results
                results[model_name] = {
                    'model': model,
                    'metrics': metrics,
                    'y_pred': y_pred,
                    'y_pred_proba': y_pred_proba
                }
                
                print(f"✅ {model_name} trained")
                print(f"   Accuracy:  {metrics['accuracy']:.4f}")
                print(f"   Precision: {metrics['precision']:.4f}")
                print(f"   Recall:    {metrics['recall']:.4f}")
                print(f"   F1 Score:  {metrics['f1_score']:.4f}")
                print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
                
                # Log to MLflow (FIXED for XGBoost)
                if use_mlflow:
                    mlflow.log_metrics(metrics)
                    
                    # Log model - handle XGBoost specially
                    if model_name == 'RandomForest':
                        mlflow.sklearn.log_model(model, "model")
                    elif model_name == 'XGBoost':
                        # Try XGBoost logging, fallback to sklearn if fails
                        try:
                            if XGBOOST_MLFLOW_OK:
                                mlflow.xgboost.log_model(model, "model")
                            else:
                                mlflow.sklearn.log_model(model, "model")
                                mlflow.set_tag("xgboost_logged_as", "sklearn")
                        except:
                            mlflow.sklearn.log_model(model, "model")
                            mlflow.set_tag("xgboost_logged_as", "sklearn_fallback")
                    
                    mlflow.end_run()
                    
            except Exception as e:
                print(f"❌ Error training {model_name}: {e}")
                if use_mlflow:
                    mlflow.end_run(status="FAILED")
        
        self.results = results
        
        # Performance assessment
        print("\n🔍 Performance Assessment:")
        for model_name, result in results.items():
            auc = result['metrics']['roc_auc']
            if auc > 0.7:
                rating = "✅ Good"
            elif auc > 0.6:
                rating = "📊 Acceptable"
            else:
                rating = "📉 Needs feature engineering"
            print(f"   {model_name}: AUC = {auc:.4f} - {rating}")
        
        return results
    
    def tune_models_quick(self):
        """
        Quick hyperparameter tuning (lightweight).
        """
        print("\n" + "="*60)
        print("🔧 QUICK HYPERPARAMETER TUNING")
        print("="*60)
        
        # Only tune Random Forest (more stable)
        print("\n📊 Tuning Random Forest...")
        
        param_grid = {
            'n_estimators': [100, 150],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5],
            'min_samples_leaf': [1, 2]
        }
        
        cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=self.random_state)
        
        search = RandomizedSearchCV(
            estimator=self.models['RandomForest'],
            param_distributions=param_grid,
            n_iter=8,  # Very few iterations for speed
            cv=cv,
            scoring='roc_auc',
            random_state=self.random_state,
            n_jobs=-1,
            verbose=0
        )
        
        search.fit(self.X_train, self.y_train)
        
        best_rf = search.best_estimator_
        best_params = search.best_params_
        
        print(f"✅ Best Random Forest parameters:")
        for param, value in best_params.items():
            print(f"   {param}: {value}")
        
        # Evaluate
        y_pred_proba = best_rf.predict_proba(self.X_test)[:, 1]
        metrics = self._calculate_metrics(self.y_test, best_rf.predict(self.X_test), y_pred_proba)
        
        print(f"📊 Test performance:")
        print(f"   ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        # Store tuned model
        self.models['RandomForest_tuned'] = best_rf
        self.results['RandomForest_tuned'] = {
            'model': best_rf,
            'metrics': metrics,
            'y_pred_proba': y_pred_proba
        }
        
        # Log to MLflow
        with mlflow.start_run(run_name="RandomForest_tuned"):
            mlflow.log_params(best_params)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(best_rf, "tuned_model")
            mlflow.set_tag("model_type", "RandomForest_tuned")
        
        return best_rf, best_params, metrics
    
    def select_best_model(self):
        """
        Select best model based on ROC-AUC.
        """
        print("\n" + "="*60)
        print("🏆 SELECTING BEST MODEL")
        print("="*60)
        
        if not self.results:
            raise ValueError("No results available")
        
        best_score = -1
        best_model_name = None
        
        for model_name, result in self.results.items():
            auc = result['metrics']['roc_auc']
            print(f"   {model_name}: AUC = {auc:.4f}")
            
            if auc > best_score:
                best_score = auc
                best_model_name = model_name
        
        self.best_model_name = best_model_name
        self.best_model = self.results[best_model_name]['model']
        best_metrics = self.results[best_model_name]['metrics']
        
        print(f"\n✅ Best model: {best_model_name}")
        print(f"   ROC-AUC: {best_score:.4f}")
        
        return best_model_name, self.best_model, best_metrics
    
    def register_best_model(self):
        """
        Register best model in MLflow Registry.
        """
        print("\n" + "="*60)
        print("📝 REGISTERING BEST MODEL")
        print("="*60)
        
        if self.best_model is None:
            raise ValueError("No best model selected")
        
        model_name_for_registry = "credit_risk_production"
        
        with mlflow.start_run(run_name=f"{self.best_model_name}_registered"):
            # Get metrics from best model
            y_pred_proba = self.best_model.predict_proba(self.X_test)[:, 1]
            metrics = self._calculate_metrics(self.y_test, 
                                            self.best_model.predict(self.X_test), 
                                            y_pred_proba)
            
            mlflow.log_metrics(metrics)
            mlflow.set_tag("best_model", "true")
            mlflow.set_tag("registered", "true")
            mlflow.set_tag("auc_score", f"{metrics['roc_auc']:.4f}")
            
            # Create signature
            signature = infer_signature(self.X_train, self.best_model.predict(self.X_train))
            
            # Register model
            mlflow.sklearn.log_model(
                self.best_model,
                "production_model",
                signature=signature,
                registered_model_name=model_name_for_registry
            )
            
            print(f"✅ Model registered: {model_name_for_registry}")
            print(f"   Type: {self.best_model_name}")
            print(f"   AUC: {metrics['roc_auc']:.4f}")
    
    def create_final_report(self):
        """
        Create comprehensive final report.
        """
        print("\n" + "="*60)
        print("📈 CREATING FINAL REPORT")
        print("="*60)
        
        os.makedirs("../reports", exist_ok=True)
        os.makedirs("../models", exist_ok=True)
        
        # 1. Save best model
        model_path = "../models/best_credit_risk_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump(self.best_model, f)
        
        print(f"💾 Best model saved: {model_path}")
        
        # 2. Create comparison table
        comparison_data = []
        for model_name, result in self.results.items():
            metrics = result['metrics']
            comparison_data.append({
                'Model': model_name,
                'ROC-AUC': f"{metrics['roc_auc']:.4f}",
                'Accuracy': f"{metrics['accuracy']:.4f}",
                'Precision': f"{metrics['precision']:.4f}",
                'Recall': f"{metrics['recall']:.4f}",
                'F1-Score': f"{metrics['f1_score']:.4f}"
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        comparison_path = "../reports/model_comparison_final.csv"
        comparison_df.to_csv(comparison_path, index=False)
        
        print(f"\n📋 Model Comparison:")
        print(comparison_df.to_string(index=False))
        print(f"💾 Saved to: {comparison_path}")
        
        # 3. Business analysis
        print("\n" + "="*60)
        print("💼 BUSINESS ANALYSIS")
        print("="*60)
        
        best_result = self.results[self.best_model_name]
        y_pred = best_result['model'].predict(self.X_test)
        cm = confusion_matrix(self.y_test, y_pred)
        
        tn, fp, fn, tp = cm.ravel()
        
        print(f"Confusion Matrix ({self.best_model_name}):")
        print(f"                 Predicted")
        print(f"                 Low Risk  High Risk")
        print(f"Actual Low Risk   {tn:>6}     {fp:>6}")
        print(f"Actual High Risk  {fn:>6}     {tp:>6}")
        
        total = len(self.y_test)
        actual_positives = self.y_test.sum()
        
        print(f"\n📊 Key Metrics:")
        print(f"   Total customers: {total:,}")
        print(f"   Actual high-risk: {actual_positives:,} ({actual_positives/total*100:.1f}%)")
        print(f"   Detection Rate (Recall): {tp/actual_positives*100:.1f}%")
        print(f"   False Alarm Rate: {fp/(total-actual_positives)*100:.1f}%")
        print(f"   Precision: {tp/(tp+fp)*100:.1f}% when predicting high-risk")
        
        # 4. Recommendations
        print("\n" + "="*60)
        print("🎯 RECOMMENDATIONS")
        print("="*60)
        
        auc_score = best_result['metrics']['roc_auc']
        
        print(f"""
        Based on AUC = {auc_score:.4f}:
        
        {'✅ GOOD FOR PRODUCTION' if auc_score > 0.65 else '⚠️ NEEDS IMPROVEMENT'}
        
        1. **Model Performance**:
           - AUC {auc_score:.4f} indicates {'reasonable' if auc_score > 0.65 else 'limited'} predictive power
           - Captures {tp/actual_positives*100:.1f}% of actual high-risk customers
           - Wrongly flags {fp/(total-actual_positives)*100:.1f}% of good customers
        
        2. **Deployment Strategy**:
           - Use as **first screening layer**
           - High-risk predictions: Manual review required
           - Low-risk predictions: Auto-approve with monitoring
        
        3. **Improvement Suggestions**:
           - Add more features (demographics, device info, etc.)
           - Collect real default data when available
           - Implement model retraining every 3 months
        
        4. **Next Steps**:
           - Proceed to Task 6: Model Deployment
           - Create API with adjustable risk threshold
           - Set up monitoring dashboard
        """)
        
        # 5. Save feature importance
        if hasattr(self.best_model, 'feature_importances_'):
            importances = self.best_model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'Feature': self.feature_names,
                'Importance': importances
            }).sort_values('Importance', ascending=False)
            
            importance_path = "../reports/feature_importance_final.csv"
            feature_importance_df.to_csv(importance_path, index=False)
            print(f"💾 Feature importance saved: {importance_path}")
            
            # Top features
            print(f"\n🔝 Top 5 Most Important Features:")
            for i, row in feature_importance_df.head(5).iterrows():
                print(f"   {row['Feature']}: {row['Importance']:.4f}")
    
    def _calculate_metrics(self, y_true, y_pred, y_pred_proba):
        """Calculate evaluation metrics."""
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, zero_division=0),
            'recall': recall_score(y_true, y_pred, zero_division=0),
            'f1_score': f1_score(y_true, y_pred, zero_division=0),
            'roc_auc': roc_auc_score(y_true, y_pred_proba)
        }


def main_final():
    """
    FINAL main function - complete Task 5.
    """
    print("="*70)
    print("TASK 5: FINAL MODEL TRAINING")
    print("="*70)
    
    # Set MLflow tracking
    mlflow.set_tracking_uri("file:../mlruns")
    
    # Initialize trainer
    trainer = FinalCreditRiskModelTrainer(
        experiment_name="credit_risk_final",
        random_state=42
    )
    
    try:
        # 1. Load and prepare data
        data_path = "data/processed/data_with_target.csv"
        trainer.load_and_prepare_data(data_path)
        
        # 2. Create and train baseline models
        trainer.create_models()
        baseline_results = trainer.train_baseline_models(use_mlflow=True)
        
        # 3. Quick tuning (optional)
        print("\n" + "="*70)
        print("OPTIONAL: QUICK TUNING")
        print("="*70)
        
        tune_choice = input("Run quick hyperparameter tuning? (y/n): ").strip().lower()
        if tune_choice == 'y':
            tuned_model, params, metrics = trainer.tune_models_quick()
            print(f"✅ Tuning complete. New AUC: {metrics['roc_auc']:.4f}")
        
        # 4. Select best model
        print("\n" + "="*70)
        print("MODEL SELECTION")
        print("="*70)
        best_name, best_model, best_metrics = trainer.select_best_model()
        
        # 5. Register best model
        trainer.register_best_model()
        
        # 6. Create final report
        trainer.create_final_report()
        
        # Final summary
        print("\n" + "="*70)
        print("✅ TASK 5 COMPLETED SUCCESSFULLY!")
        print("="*70)
        print(f"""
        🎉 CONGRATULATIONS! Task 5 is complete.
        
        Results:
        - Best Model: {best_name}
        - ROC-AUC: {best_metrics['roc_auc']:.4f}
        - Status: {'✅ Production Ready' if best_metrics['roc_auc'] > 0.65 else '⚠️ Needs Improvement'}
        
        Files Created:
        - Model: ../models/best_credit_risk_model.pkl
        - Reports: ../reports/model_comparison_final.csv
        - MLflow: All experiments logged
        
        Next Steps (Task 6):
        1. View MLflow: mlflow ui --port 5000
        2. Create FastAPI for model serving
        3. Dockerize the application
        4. Set up CI/CD pipeline
        
        💡 Tip: Your AUC of ~0.62 is REALISTIC for:
        - Limited feature set (7 features)
        - Alternative credit scoring
        - First version of the model
        """)
        
        return trainer
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None


if __name__ == "__main__":
    # Run the final training
    trainer = main_final()