"""
Tier A: The Statistician
Multi-class classifier using XGBoost and Random Forest on statistical features
to distinguish between Human, AI-Generic, and AI-Styled text.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    confusion_matrix, 
    accuracy_score,
    f1_score,
    precision_recall_fscore_support
)
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Set style for better visualizations
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 8)


class TierAStatistician:
    """
    Tier A Detector: Uses only numerical statistical features
    to classify text as Human, AI-Generic, or AI-Styled
    """
    
    def __init__(self, data_path='fingerprints.csv'):
        """Initialize the classifier and load data"""
        self.data_path = data_path
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.feature_names = None
        
        # Models
        self.rf_model = None
        self.xgb_model = None
        
    def load_and_prepare_data(self):
        """Load data and prepare features"""
        print("=" * 70)
        print("TIER A: THE STATISTICIAN")
        print("=" * 70)
        print("\n📊 Loading fingerprint data...")
        
        self.df = pd.read_csv(self.data_path)
        print(f"✓ Loaded {len(self.df)} samples")
        
        # Display class distribution
        print("\n📈 Class Distribution:")
        class_dist = self.df['class'].value_counts()
        for class_name, count in class_dist.items():
            print(f"   {class_name}: {count} samples ({count/len(self.df)*100:.1f}%)")
        
        # Define statistical features (only numerical, no text-based features)
        self.feature_names = [
            'ttr',                    # Type-Token Ratio
            'hapax_count',            # Hapax Legomena count
            'hapax_percentage',       # Hapax percentage
            'adj_noun_ratio',         # Adjective to Noun ratio
            'dep_tree_depth',         # Dependency tree depth
            'avg_sentence_length',    # Average sentence length
            'flesch_kincaid',         # Flesch-Kincaid Grade Level
            'punct_comma',            # Comma density
            'punct_period',           # Period density
            'punct_semicolon',        # Semicolon density
            'punct_colon',            # Colon density
            'punct_exclamation',      # Exclamation mark density
            'punct_question',         # Question mark density
            'word_count'              # Total word count
        ]
        
        print(f"\n🔢 Using {len(self.feature_names)} statistical features:")
        for i, feat in enumerate(self.feature_names, 1):
            print(f"   {i:2d}. {feat}")
        
        return self
    
    def explore_features(self):
        """Perform exploratory data analysis on features"""
        print("\n" + "=" * 70)
        print("EXPLORATORY DATA ANALYSIS")
        print("=" * 70)
        
        # Feature statistics by class
        print("\n📊 Feature Statistics by Class:\n")
        for feature in self.feature_names[:7]:  # Show first 7 features
            print(f"\n{feature.upper()}:")
            stats = self.df.groupby('class')[feature].describe()[['mean', 'std', '50%']]
            print(stats)
        
        # Check for missing values
        missing = self.df[self.feature_names].isnull().sum()
        if missing.sum() > 0:
            print("\n⚠️  Missing values detected:")
            print(missing[missing > 0])
        else:
            print("\n✓ No missing values found")
        
        # Correlation analysis
        print("\n🔗 Analyzing feature correlations...")
        self._plot_correlation_matrix()
        
        # Feature distributions
        print("📉 Generating feature distribution plots...")
        self._plot_feature_distributions()
        
        return self
    
    def _plot_correlation_matrix(self):
        """Plot correlation matrix of features"""
        plt.figure(figsize=(14, 12))
        corr_matrix = self.df[self.feature_names].corr()
        
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        sns.heatmap(
            corr_matrix, 
            mask=mask,
            annot=True, 
            fmt='.2f', 
            cmap='coolwarm',
            center=0,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8}
        )
        plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold', pad=20)
        
    
    def _plot_feature_distributions(self):
        """Plot distributions of key features by class"""
        key_features = ['ttr', 'hapax_percentage', 'adj_noun_ratio', 
                       'dep_tree_depth', 'flesch_kincaid', 'punct_semicolon']
        
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.ravel()
        
        for idx, feature in enumerate(key_features):
            for class_name in self.df['class'].unique():
                class_data = self.df[self.df['class'] == class_name][feature]
                axes[idx].hist(class_data, alpha=0.5, label=class_name, bins=30)
            
            axes[idx].set_xlabel(feature.replace('_', ' ').title(), fontsize=11)
            axes[idx].set_ylabel('Frequency', fontsize=11)
            axes[idx].legend()
            axes[idx].grid(True, alpha=0.3)
        
        
    def prepare_train_test_split(self, test_size=0.2, random_state=42):
        """Split data into training and testing sets"""
        print("\n" + "=" * 70)
        print("DATA PREPARATION")
        print("=" * 70)
        
        X = self.df[self.feature_names].values
        y = self.df['class'].values
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"\n📊 Label Encoding:")
        for i, label in enumerate(self.label_encoder.classes_):
            print(f"   {label} → {i}")
        
        # Train-test split
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y_encoded, 
            test_size=test_size, 
            random_state=random_state,
            stratify=y_encoded
        )
        
        print(f"\n✓ Train set: {len(self.X_train)} samples")
        print(f"✓ Test set:  {len(self.X_test)} samples")
        
        # Scale features
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        
        print("✓ Features scaled using StandardScaler")
        
        return self
    
    def train_random_forest(self, use_grid_search=True):
        """Train Random Forest classifier"""
        print("\n" + "=" * 70)
        print("TRAINING: RANDOM FOREST")
        print("=" * 70)
        
        if use_grid_search:
            print("\n🔍 Performing Grid Search for hyperparameter tuning...")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [10, 20, 30, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'max_features': ['sqrt', 'log2']
            }
            
            rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
            grid_search = GridSearchCV(
                rf_base, 
                param_grid, 
                cv=5, 
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(self.X_train, self.y_train)
            
            self.rf_model = grid_search.best_estimator_
            print(f"\n✓ Best parameters: {grid_search.best_params_}")
            print(f"✓ Best CV F1 Score: {grid_search.best_score_:.4f}")
        else:
            print("\n🌲 Training Random Forest with default parameters...")
            self.rf_model = RandomForestClassifier(
                n_estimators=200,
                max_depth=20,
                random_state=42,
                n_jobs=-1
            )
            self.rf_model.fit(self.X_train, self.y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(self.rf_model, self.X_train, self.y_train, cv=5, scoring='f1_weighted')
        print(f"\n📊 Cross-Validation F1 Scores: {cv_scores}")
        print(f"   Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return self
    
    def train_xgboost(self, use_grid_search=True):
        """Train XGBoost classifier"""
        print("\n" + "=" * 70)
        print("TRAINING: XGBOOST")
        print("=" * 70)
        
        if use_grid_search:
            print("\n🔍 Performing Grid Search for hyperparameter tuning...")
            param_grid = {
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1],
                'n_estimators': [100, 200, 300],
                'subsample': [0.8, 0.9, 1.0],
                'colsample_bytree': [0.8, 0.9, 1.0]
            }
            
            xgb_base = xgb.XGBClassifier(random_state=42, n_jobs=-1, eval_metric='mlogloss')
            grid_search = GridSearchCV(
                xgb_base, 
                param_grid, 
                cv=5, 
                scoring='f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(self.X_train, self.y_train)
            
            self.xgb_model = grid_search.best_estimator_
            print(f"\n✓ Best parameters: {grid_search.best_params_}")
            print(f"✓ Best CV F1 Score: {grid_search.best_score_:.4f}")
        else:
            print("\n🚀 Training XGBoost with default parameters...")
            self.xgb_model = xgb.XGBClassifier(
                max_depth=7,
                learning_rate=0.1,
                n_estimators=200,
                random_state=42,
                n_jobs=-1,
                eval_metric='mlogloss'
            )
            self.xgb_model.fit(self.X_train, self.y_train)
        
        # Cross-validation
        cv_scores = cross_val_score(self.xgb_model, self.X_train, self.y_train, cv=5, scoring='f1_weighted')
        print(f"\n📊 Cross-Validation F1 Scores: {cv_scores}")
        print(f"   Mean: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return self
    
    def evaluate_models(self):
        """Evaluate both models on test set"""
        print("\n" + "=" * 70)
        print("MODEL EVALUATION")
        print("=" * 70)
        
        models = {
            'Random Forest': self.rf_model,
            'XGBoost': self.xgb_model
        }
        
        results = {}
        
        for model_name, model in models.items():
            print(f"\n{'='*70}")
            print(f"{model_name.upper()}")
            print(f"{'='*70}")
            
            # Predictions
            y_pred = model.predict(self.X_test)
            
            # Metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            f1 = f1_score(self.y_test, y_pred, average='weighted')
            
            print(f"\n🎯 Test Accuracy: {accuracy:.4f}")
            print(f"📊 Weighted F1 Score: {f1:.4f}")
            
            # Classification report
            print(f"\n📋 Classification Report:")
            print("-" * 70)
            target_names = self.label_encoder.classes_
            print(classification_report(self.y_test, y_pred, target_names=target_names))
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            self._plot_confusion_matrix(cm, model_name, target_names)
            
            results[model_name] = {
                'accuracy': accuracy,
                'f1_score': f1,
                'predictions': y_pred,
                'confusion_matrix': cm
            }
        
        # Compare models
        self._compare_models(results)
        
        return results
    
    def _plot_confusion_matrix(self, cm, model_name, labels):
        """Plot confusion matrix"""
        plt.figure(figsize=(10, 8))
        
        # Normalize confusion matrix
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        sns.heatmap(
            cm_normalized, 
            annot=True, 
            fmt='.2%', 
            cmap='Blues',
            xticklabels=labels,
            yticklabels=labels,
            square=True,
            linewidths=1,
            cbar_kws={"shrink": 0.8}
        )
        
        plt.title(f'Confusion Matrix - {model_name}\n(Normalized by True Label)', 
                 fontsize=14, fontweight='bold', pad=20)
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        filename = f"tier_a_confusion_matrix_{model_name.lower().replace(' ', '_')}.png"
        plt.savefig(f'D:/precog_task/outputs/{filename}', dpi=300, bbox_inches='tight')
        print(f"✓ Saved confusion matrix to {filename}")
        plt.close()
    
    def _compare_models(self, results):
        """Compare model performance"""
        print("\n" + "=" * 70)
        print("MODEL COMPARISON")
        print("=" * 70)
        
        comparison_df = pd.DataFrame({
            'Model': list(results.keys()),
            'Accuracy': [results[m]['accuracy'] for m in results],
            'F1 Score': [results[m]['f1_score'] for m in results]
        })
        
        print("\n", comparison_df.to_string(index=False))
        
        # Visualize comparison
        fig, ax = plt.subplots(1, 2, figsize=(14, 5))
        
        # Accuracy comparison
        ax[0].bar(comparison_df['Model'], comparison_df['Accuracy'], color=['#2ecc71', '#3498db'])
        ax[0].set_ylabel('Accuracy', fontsize=12)
        ax[0].set_title('Model Accuracy Comparison', fontsize=14, fontweight='bold')
        ax[0].set_ylim([0, 1])
        ax[0].grid(True, alpha=0.3)
        for i, v in enumerate(comparison_df['Accuracy']):
            ax[0].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
        
        # F1 Score comparison
        ax[1].bar(comparison_df['Model'], comparison_df['F1 Score'], color=['#2ecc71', '#3498db'])
        ax[1].set_ylabel('F1 Score', fontsize=12)
        ax[1].set_title('Model F1 Score Comparison', fontsize=14, fontweight='bold')
        ax[1].set_ylim([0, 1])
        ax[1].grid(True, alpha=0.3)
        for i, v in enumerate(comparison_df['F1 Score']):
            ax[1].text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
        
    def analyze_feature_importance(self):
        """Analyze and visualize feature importance"""
        print("\n" + "=" * 70)
        print("FEATURE IMPORTANCE ANALYSIS")
        print("=" * 70)
        
        # Random Forest feature importance
        rf_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.rf_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\n🌲 RANDOM FOREST - Top 10 Important Features:")
        print("-" * 70)
        print(rf_importance.head(10).to_string(index=False))
        
        # XGBoost feature importance
        xgb_importance = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': self.xgb_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        print("\n🚀 XGBOOST - Top 10 Important Features:")
        print("-" * 70)
        print(xgb_importance.head(10).to_string(index=False))
        
        # Visualize
        self._plot_feature_importance(rf_importance, xgb_importance)
        
        return rf_importance, xgb_importance
    
    def _plot_feature_importance(self, rf_importance, xgb_importance):
        """Plot feature importance for both models"""
        fig, axes = plt.subplots(1, 2, figsize=(18, 8))
        
        # Random Forest
        top_n = 10
        rf_top = rf_importance.head(top_n)
        axes[0].barh(rf_top['Feature'], rf_top['Importance'], color='#2ecc71')
        axes[0].set_xlabel('Importance', fontsize=12)
        axes[0].set_title('Random Forest - Feature Importance', fontsize=14, fontweight='bold')
        axes[0].invert_yaxis()
        axes[0].grid(True, alpha=0.3, axis='x')
        
        # XGBoost
        xgb_top = xgb_importance.head(top_n)
        axes[1].barh(xgb_top['Feature'], xgb_top['Importance'], color='#3498db')
        axes[1].set_xlabel('Importance', fontsize=12)
        axes[1].set_title('XGBoost - Feature Importance', fontsize=14, fontweight='bold')
        axes[1].invert_yaxis()
        axes[1].grid(True, alpha=0.3, axis='x')
        
    
    def generate_report(self, results):
        """Generate final analysis report"""
        print("\n" + "=" * 70)
        print("TIER A: FINAL ANALYSIS REPORT")
        print("=" * 70)
        
        best_model = max(results.items(), key=lambda x: x[1]['f1_score'])
        
        print(f"\n🏆 BEST PERFORMING MODEL: {best_model[0]}")
        print(f"   Accuracy: {best_model[1]['accuracy']:.4f}")
        print(f"   F1 Score: {best_model[1]['f1_score']:.4f}")
        
        print("\n📝 KEY FINDINGS:")
        print("-" * 70)
        
        # Analyze if models can distinguish classes
        if best_model[1]['f1_score'] > 0.8:
            print("✓ Models show STRONG ability to distinguish between classes")
            print("  → Statistical features contain significant discriminative power")
        elif best_model[1]['f1_score'] > 0.6:
            print("⚠ Models show MODERATE ability to distinguish between classes")
            print("  → Some overlap exists between class distributions")
        else:
            print("✗ Models show WEAK ability to distinguish between classes")
            print("  → Statistical features may not be sufficient for classification")
            print("  → Consider investigating semantic features (Tier B, C)")
        
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 70)
        if best_model[1]['f1_score'] < 0.7:
            print("• Analyze feature importance to identify weak discriminators")
            print("• Consider feature engineering (interaction terms, ratios)")
            print("• Move to Tier B (semantic features) for better performance")
        else:
            print("• Statistical features are effective for this task")
            print("• Consider ensemble methods combining Tier A with B/C")
            print("• Investigate misclassified samples for insights")
        
        print("\n" + "=" * 70)


def main():
    """Main execution function"""
    # Initialize classifier
    classifier = TierAStatistician('D:\precog_task\paragraph_fingerprint_results1.csv')
    
    # Load and explore data
    classifier.load_and_prepare_data()
    classifier.explore_features()
    
    # Prepare train-test split
    classifier.prepare_train_test_split(test_size=0.2, random_state=42)
    
    # Train models (set use_grid_search=False for faster execution)
    classifier.train_random_forest(use_grid_search=False)
    classifier.train_xgboost(use_grid_search=False)
    
    # Evaluate
    results = classifier.evaluate_models()
    
    # Feature importance
    classifier.analyze_feature_importance()
    
    # Generate final report
    classifier.generate_report(results)
    
    print("\n✅ Tier A analysis complete!")
    print("📁 All visualizations saved to outputs/")


if __name__ == "__main__":
    main()