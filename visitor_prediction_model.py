#!/usr/bin/env python3
"""
Australian Ski Resort Visitor Prediction Model
Uses Random Forest to predict visitor patterns based on resort features, weather, and historical data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.feature_selection import SelectKBest, f_regression
import warnings
warnings.filterwarnings('ignore')

# Set random seed for reproducibility
np.random.seed(42)

class SkiResortVisitorPredictor:
    """
    Random Forest model to predict visitor patterns at Australian ski resorts
    """
    
    def __init__(self):
        self.model = None
        self.scaler = StandardScaler()
        self.feature_selector = None
        self.label_encoders = {}
        self.feature_importance = None
        self.results = {}
        
    def load_data(self):
        """Load and prepare all data sources"""
        print("Loading data sources...")
        
        # Load visitation data
        self.visitation_df = pd.read_csv('data/visitation_data.csv')
        print(f"Loaded visitation data: {self.visitation_df.shape}")
        
        # Load resort features
        self.resort_df = pd.read_csv('data/onthesnow_resorts_data.csv')
        print(f"Loaded resort data: {self.resort_df.shape}")
        
        # Load snowfall data
        self.snowfall_df = pd.read_csv('data/merged_snowfall_data.csv')
        print(f"Loaded snowfall data: {self.snowfall_df.shape}")
        
        # Load climate data
        self.climate_df = pd.read_csv('data/climate.csv')
        print(f"Loaded climate data: {self.climate_df.shape}")
        
        return self
    
    def preprocess_visitation_data(self):
        """Transform visitation data from wide to long format"""
        print("Preprocessing visitation data...")
        
        # Melt the data to get resort names as a column
        resort_columns = ['Mt. Baw Baw', 'Mt. Stirling', 'Mt. Hotham', 'Falls Creek', 
                         'Mt. Buller', 'Selwyn', 'Thredbo', 'Perisher', 'Charlotte Pass']
        
        self.visitation_long = self.visitation_df.melt(
            id_vars=['Year', 'Week'], 
            value_vars=resort_columns,
            var_name='Resort', 
            value_name='Visitors'
        )
        
        # Exclude 2020 data due to COVID-19 impact
        self.visitation_long = self.visitation_long[self.visitation_long['Year'] != 2020]
        print(f"Excluded 2020 data due to COVID-19. Remaining data: {self.visitation_long.shape}")
        
        # Create date features - handle float values properly
        self.visitation_long['Year_clean'] = self.visitation_long['Year'].fillna(2014).astype(int)
        self.visitation_long['Week_clean'] = self.visitation_long['Week'].fillna(1).astype(int)
        
        self.visitation_long['Date'] = pd.to_datetime(
            self.visitation_long['Year_clean'].astype(str) + '-W' + 
            self.visitation_long['Week_clean'].astype(str) + '-1', 
            format='%Y-W%W-%w'
        )
        
        # Add seasonal features
        self.visitation_long['Month'] = self.visitation_long['Date'].dt.month
        self.visitation_long['DayOfYear'] = self.visitation_long['Date'].dt.dayofyear
        self.visitation_long['Season'] = self.visitation_long['Date'].dt.quarter
        
        # Create week of season (1-15 for each ski season)
        self.visitation_long['WeekOfSeason'] = self.visitation_long['Week']
        
        # Add peak season indicator (weeks 5-10 typically have highest visitation)
        self.visitation_long['PeakSeason'] = (
            (self.visitation_long['Week'] >= 5) & 
            (self.visitation_long['Week'] <= 10)
        ).astype(int)
        
        # Add school holiday periods (approximate)
        self.visitation_long['SchoolHolidays'] = (
            (self.visitation_long['Week'] == 1) |  # New Year
            (self.visitation_long['Week'] == 2) |  # New Year
            (self.visitation_long['Week'] == 6) |   # Mid-season
            (self.visitation_long['Week'] == 7) |   # Mid-season
            (self.visitation_long['Week'] == 15)    # End of season
        ).astype(int)
        
        print(f"Processed visitation data: {self.visitation_long.shape}")
        return self
    
    def preprocess_resort_features(self):
        """Extract and process resort features"""
        print("Processing resort features...")
        
        # Clean resort names to match visitation data
        resort_name_mapping = {
            'thredbo-alpine-resort-ski-resort-area-overview': 'Thredbo',
            'perisher-ski-resort-area-overview': 'Perisher',
            'mt.-buller-ski-resort-area-overview': 'Mt. Buller',
            'falls-creek-alpine-resort-ski-resort-area-overview': 'Falls Creek',
            'mt.-hotham-ski-resort-area-overview': 'Mt. Hotham',
            'mt.-baw-baw-alpine-resort-ski-resort-area-overview': 'Mt. Baw Baw',
            'charlotte-pass-ski-resort-area-overview': 'Charlotte Pass',
            'selwyn-snowfields-ski-resort-area-overview': 'Selwyn'
        }
        
        self.resort_df['Resort'] = self.resort_df['resort_key'].map(resort_name_mapping)
        
        # Extract numeric features
        numeric_features = [
            'beginner_runs_percentage', 'intermediate_runs_percentage', 
            'advanced_runs_percentage', 'total_runs', 'longest_run',
            'night_skiing_acres', 'snow_making_acres', 'total_lifts',
            'gondolas_and_trams', 'high_speed_quads', 'quad_chairs',
            'double_chairs', 'surface_lifts', 'days_open_last_year',
            'years_open', 'average_snowfall'
        ]
        
        # Clean numeric features
        for feature in numeric_features:
            if feature in self.resort_df.columns:
                # Remove quotes and convert to numeric
                self.resort_df[feature] = pd.to_numeric(
                    self.resort_df[feature].astype(str).str.replace('"', '').str.replace('%', ''),
                    errors='coerce'
                )
        
        # Extract terrain features
        self.resort_df['terrain_diversity'] = (
            self.resort_df['beginner_runs_percentage'].fillna(0) + 
            self.resort_df['intermediate_runs_percentage'].fillna(0) + 
            self.resort_df['advanced_runs_percentage'].fillna(0)
        )
        
        # Extract lift features
        self.resort_df['modern_lifts'] = (
            self.resort_df['gondolas_and_trams'].fillna(0) + 
            self.resort_df['high_speed_quads'].fillna(0)
        )
        
        # Create resort size indicator
        self.resort_df['resort_size'] = pd.cut(
            self.resort_df['total_runs'].fillna(0), 
            bins=[0, 25, 50, 100, 200], 
            labels=['Small', 'Medium', 'Large', 'Very Large']
        )
        
        print(f"Processed resort features: {self.resort_df.shape}")
        return self
    
    def preprocess_snowfall_data(self):
        """Process snowfall data and create seasonal features"""
        print("Processing snowfall data...")
        
        # Clean snowfall data
        self.snowfall_df['Total_Snowfall_Clean'] = pd.to_numeric(
            self.snowfall_df['Total Snowfall'].str.replace('"', ''), 
            errors='coerce'
        )
        
        self.snowfall_df['Snowfall_Days_Clean'] = pd.to_numeric(
            self.snowfall_df['Snowfall Days'].str.replace(' days', '').str.replace(' Day', ''), 
            errors='coerce'
        )
        
        self.snowfall_df['Average_Base_Depth_Clean'] = pd.to_numeric(
            self.snowfall_df['Average Base Depth'].str.replace('"', ''), 
            errors='coerce'
        )
        
        # Extract year from season
        self.snowfall_df['Year'] = self.snowfall_df['Year'].str.split(' - ').str[0].astype(int)
        
        # Create snowfall quality score
        self.snowfall_df['snowfall_quality'] = (
            self.snowfall_df['Total_Snowfall_Clean'].fillna(0) * 0.4 +
            self.snowfall_df['Snowfall_Days_Clean'].fillna(0) * 0.3 +
            self.snowfall_df['Average_Base_Depth_Clean'].fillna(0) * 0.3
        )
        
        print(f"Processed snowfall data: {self.snowfall_df.shape}")
        return self
    
    def merge_all_data(self):
        """Merge all data sources into a single dataset"""
        print("Merging all data sources...")
        
        # Start with visitation data
        self.merged_df = self.visitation_long.copy()
        
        # Merge with resort features
        resort_features = self.resort_df.copy()
        resort_features = resort_features.drop(['resort_key', 'url', 'scraped_at', 'data_source', 'resort_name'], axis=1, errors='ignore')
        
        self.merged_df = self.merged_df.merge(
            resort_features, 
            on='Resort', 
            how='left'
        )
        
        # Merge with snowfall data
        self.merged_df = self.merged_df.merge(
            self.snowfall_df[['Location', 'Year', 'Total_Snowfall_Clean', 'Snowfall_Days_Clean', 
                            'Average_Base_Depth_Clean', 'snowfall_quality']], 
            left_on=['Resort', 'Year'], 
            right_on=['Location', 'Year'], 
            how='left'
        )
        
        # Drop redundant columns
        self.merged_df = self.merged_df.drop(['Location'], axis=1)
        
        print(f"Merged dataset shape: {self.merged_df.shape}")
        print(f"Columns: {list(self.merged_df.columns)}")
        
        return self
    
    def create_features(self):
        """Create additional features for the model"""
        print("Creating additional features...")
        
        # Resort popularity (average visitors per resort)
        resort_avg = self.merged_df.groupby('Resort')['Visitors'].mean().reset_index()
        resort_avg.columns = ['Resort', 'avg_resort_visitors']
        self.merged_df = self.merged_df.merge(resort_avg, on='Resort', how='left')
        
        # Year trend
        self.merged_df['year_trend'] = self.merged_df['Year'] - 2014
        
        # Week trend within season
        self.merged_df['week_trend'] = self.merged_df['Week'] - 1
        
        # Interaction features
        self.merged_df['peak_snowfall'] = self.merged_df['PeakSeason'] * self.merged_df['snowfall_quality'].fillna(0)
        self.merged_df['holiday_snowfall'] = self.merged_df['SchoolHolidays'] * self.merged_df['snowfall_quality'].fillna(0)
        
        # Resort capacity features
        self.merged_df['resort_capacity'] = (
            self.merged_df['total_runs'].fillna(0) * 
            self.merged_df['total_lifts'].fillna(0)
        )
        
        # Terrain difficulty score
        self.merged_df['terrain_difficulty'] = (
            self.merged_df['beginner_runs_percentage'].fillna(0) * 1 +
            self.merged_df['intermediate_runs_percentage'].fillna(0) * 2 +
            self.merged_df['advanced_runs_percentage'].fillna(0) * 3
        ) / 100
        
        print(f"Created features. Final dataset shape: {self.merged_df.shape}")
        return self
    
    def prepare_model_data(self):
        """Prepare data for modeling"""
        print("Preparing data for modeling...")
        
        # No log transformation - use original scale for better accuracy
        print("Using original visitor numbers scale for better accuracy")
        
        # Select features for modeling
        feature_columns = [
            # Temporal features
            'Year', 'Week', 'Month', 'DayOfYear', 'Season', 'WeekOfSeason',
            'PeakSeason', 'SchoolHolidays', 'year_trend', 'week_trend',
            
            # Resort features
            'beginner_runs_percentage', 'intermediate_runs_percentage', 'advanced_runs_percentage',
            'total_runs', 'longest_run', 'night_skiing_acres', 'snow_making_acres',
            'total_lifts', 'gondolas_and_trams', 'high_speed_quads', 'quad_chairs',
            'double_chairs', 'surface_lifts', 'days_open_last_year', 'years_open',
            'average_snowfall', 'terrain_diversity', 'modern_lifts', 'resort_capacity',
            'terrain_difficulty', 'avg_resort_visitors',
            
            # Snowfall features
            'Total_Snowfall_Clean', 'Snowfall_Days_Clean', 'Average_Base_Depth_Clean',
            'snowfall_quality', 'peak_snowfall', 'holiday_snowfall'
        ]
        
        # Remove rows with missing target variable
        self.model_df = self.merged_df.dropna(subset=['Visitors'])
        
        # Remove rows with too many missing features
        self.model_df = self.model_df.dropna(subset=feature_columns, thresh=len(feature_columns)//2)
        
        # Fill remaining missing values
        numeric_features = self.model_df[feature_columns].select_dtypes(include=[np.number]).columns
        self.model_df[numeric_features] = self.model_df[numeric_features].fillna(0)
        
        # Prepare X and y
        self.X = self.model_df[feature_columns]
        self.y = self.model_df['Visitors']  # Use original target
        
        # Encode categorical variables
        categorical_features = ['resort_size']
        for feature in categorical_features:
            if feature in self.X.columns:
                le = LabelEncoder()
                self.X[feature] = le.fit_transform(self.X[feature].astype(str))
                self.label_encoders[feature] = le
        
        print(f"Model data prepared: X shape {self.X.shape}, y shape {self.y.shape}")
        return self
    
    def train_model(self):
        """Train the Random Forest model"""
        print("Training Random Forest model...")
        
        # Split the data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y, test_size=0.2, random_state=42
        )
        
        # Scale features
        self.X_train_scaled = self.scaler.fit_transform(self.X_train)
        self.X_test_scaled = self.scaler.transform(self.X_test)
        
        # Feature selection
        self.feature_selector = SelectKBest(score_func=f_regression, k=20)
        self.X_train_selected = self.feature_selector.fit_transform(self.X_train_scaled, self.y_train)
        self.X_test_selected = self.feature_selector.transform(self.X_test_scaled)
        
        # Get selected feature names
        selected_features_mask = self.feature_selector.get_support()
        self.selected_features = self.X.columns[selected_features_mask].tolist()
        print(f"Selected features: {self.selected_features}")
        
        # Initialize and train Random Forest
        self.model = RandomForestRegressor(
            n_estimators=100,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=42,
            n_jobs=-1
        )
        
        # Train the model
        self.model.fit(self.X_train_selected, self.y_train)
        
        # Make predictions
        self.y_pred = self.model.predict(self.X_test_selected)
        
        # Calculate metrics on original scale
        self.results = {
            'r2_score': r2_score(self.y_test, self.y_pred),
            'mse': mean_squared_error(self.y_test, self.y_pred),
            'mae': mean_absolute_error(self.y_test, self.y_pred),
            'rmse': np.sqrt(mean_squared_error(self.y_test, self.y_pred))
        }
        
        # Feature importance
        self.feature_importance = pd.DataFrame({
            'feature': self.selected_features,
            'importance': self.model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print("Model training completed!")
        print(f"R² Score: {self.results['r2_score']:.4f}")
        print(f"RMSE: {self.results['rmse']:.2f}")
        print(f"MAE: {self.results['mae']:.2f}")
        
        return self
    
    def cross_validate_model(self):
        """Perform cross-validation"""
        print("Performing cross-validation...")
        
        cv_scores = cross_val_score(
            self.model, 
            self.X_train_selected, 
            self.y_train, 
            cv=5, 
            scoring='r2'
        )
        
        print(f"Cross-validation R² scores: {cv_scores}")
        print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        
        return self
    
    def analyze_predictions(self):
        """Analyze model predictions"""
        print("Analyzing predictions...")
        
        # Create prediction analysis dataframe
        test_indices = self.X_test.index
        self.prediction_analysis = pd.DataFrame({
            'Actual': self.y_test,
            'Predicted': self.y_pred,
            'Resort': self.model_df.loc[test_indices]['Resort'].values,
            'Year': self.model_df.loc[test_indices]['Year'].values,
            'Week': self.model_df.loc[test_indices]['Week'].values
        })
        
        # Calculate prediction errors
        self.prediction_analysis['Error'] = self.prediction_analysis['Actual'] - self.prediction_analysis['Predicted']
        self.prediction_analysis['Error_Percentage'] = (
            self.prediction_analysis['Error'] / self.prediction_analysis['Actual'].replace(0, 1)
        ) * 100
        
        # Resort-wise performance
        resort_performance = self.prediction_analysis.groupby('Resort').agg({
            'Actual': 'mean',
            'Predicted': 'mean',
            'Error': 'mean',
            'Error_Percentage': 'mean'
        }).round(2)
        
        print("\nResort-wise Performance:")
        print(resort_performance)
        
        return self
    
    def create_visualizations(self):
        """Create visualizations for model analysis"""
        print("Creating visualizations...")
        
        # Set up the plotting style
        plt.style.use('seaborn-v0_8')
        
        # 1. Actual vs Predicted scatter plot
        plt.figure(figsize=(10, 8))
        plt.scatter(self.y_test, self.y_pred, alpha=0.6)
        plt.plot([self.y_test.min(), self.y_test.max()], 
                [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        plt.xlabel('Actual Visitors')
        plt.ylabel('Predicted Visitors')
        plt.title('Actual vs Predicted Visitors')
        plt.grid(True, alpha=0.3)
        plt.savefig('visualizations/actual_vs_predicted.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 2. Feature importance
        plt.figure(figsize=(12, 8))
        top_features = self.feature_importance.head(15)
        plt.barh(range(len(top_features)), top_features['importance'])
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Feature Importance')
        plt.title('Top 15 Feature Importance')
        plt.grid(True, alpha=0.3)
        plt.savefig('visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 3. Residuals plot (improved)
        plt.figure(figsize=(10, 8))
        residuals = self.y_test - self.y_pred
        plt.scatter(self.y_pred, residuals, alpha=0.6)
        plt.axhline(y=0, color='r', linestyle='--')
        plt.xlabel('Predicted Visitors')
        plt.ylabel('Residuals')
        plt.title('Residuals Plot')
        plt.grid(True, alpha=0.3)
        plt.savefig('visualizations/residuals_plot.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 4. Resort-wise performance
        plt.figure(figsize=(12, 8))
        resort_avg = self.prediction_analysis.groupby('Resort')['Error_Percentage'].mean().sort_values()
        plt.bar(range(len(resort_avg)), resort_avg.values)
        plt.xticks(range(len(resort_avg)), resort_avg.index, rotation=45)
        plt.ylabel('Mean Error Percentage')
        plt.title('Resort-wise Prediction Error')
        plt.grid(True, alpha=0.3)
        plt.savefig('visualizations/resort_performance.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 5. Week-wise average visitors
        plt.figure(figsize=(10, 8))
        week_avg = self.merged_df.groupby('Week')['Visitors'].mean()
        plt.plot(week_avg.index, week_avg.values, marker='o', linewidth=2, markersize=6)
        plt.xlabel('Week of Season')
        plt.ylabel('Average Visitors')
        plt.title('Average Visitors by Week')
        plt.grid(True, alpha=0.3)
        plt.savefig('visualizations/weekly_trend.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 6. Year-wise trend (excluding 2020)
        plt.figure(figsize=(10, 8))
        year_avg = self.merged_df.groupby('Year')['Visitors'].mean()
        plt.plot(year_avg.index, year_avg.values, marker='o', linewidth=2, markersize=6)
        plt.xlabel('Year')
        plt.ylabel('Average Visitors')
        plt.title('Average Visitors by Year (Excluding 2020)')
        plt.grid(True, alpha=0.3)
        plt.savefig('visualizations/yearly_trend.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        # 7. Combined analysis plot
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Actual vs Predicted
        axes[0, 0].scatter(self.y_test, self.y_pred, alpha=0.6)
        axes[0, 0].plot([self.y_test.min(), self.y_test.max()], 
                       [self.y_test.min(), self.y_test.max()], 'r--', lw=2)
        axes[0, 0].set_xlabel('Actual Visitors')
        axes[0, 0].set_ylabel('Predicted Visitors')
        axes[0, 0].set_title('Actual vs Predicted Visitors')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Feature importance
        top_features = self.feature_importance.head(10)
        axes[0, 1].barh(range(len(top_features)), top_features['importance'])
        axes[0, 1].set_yticks(range(len(top_features)))
        axes[0, 1].set_yticklabels(top_features['feature'])
        axes[0, 1].set_xlabel('Feature Importance')
        axes[0, 1].set_title('Top 10 Feature Importance')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Residuals
        axes[1, 0].scatter(self.y_pred, residuals, alpha=0.6)
        axes[1, 0].axhline(y=0, color='r', linestyle='--')
        axes[1, 0].set_xlabel('Predicted Visitors')
        axes[1, 0].set_ylabel('Residuals')
        axes[1, 0].set_title('Residuals Plot')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Weekly trend
        axes[1, 1].plot(week_avg.index, week_avg.values, marker='o', linewidth=2, markersize=6)
        axes[1, 1].set_xlabel('Week of Season')
        axes[1, 1].set_ylabel('Average Visitors')
        axes[1, 1].set_title('Average Visitors by Week')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig('visualizations/combined_analysis.png', dpi=300, bbox_inches='tight')
        plt.close()
        
        return self
    
    def generate_insights(self):
        """Generate insights from the model"""
        print("\n" + "="*60)
        print("MODEL INSIGHTS AND RECOMMENDATIONS")
        print("="*60)
        
        # Top features affecting visitor numbers
        print("\n1. TOP FACTORS AFFECTING VISITOR NUMBERS:")
        for i, (feature, importance) in enumerate(self.feature_importance.head(5).values, 1):
            print(f"   {i}. {feature}: {importance:.4f}")
        
        # Peak season analysis
        peak_analysis = self.merged_df.groupby('PeakSeason')['Visitors'].mean()
        print(f"\n2. PEAK SEASON IMPACT:")
        print(f"   Peak season (weeks 5-10): {peak_analysis[1]:.0f} average visitors")
        print(f"   Off-peak season: {peak_analysis[0]:.0f} average visitors")
        print(f"   Peak season multiplier: {peak_analysis[1]/peak_analysis[0]:.2f}x")
        
        # Resort performance analysis
        resort_performance = self.merged_df.groupby('Resort')['Visitors'].mean().sort_values(ascending=False)
        print(f"\n3. RESORT POPULARITY RANKING:")
        for i, (resort, visitors) in enumerate(resort_performance.items(), 1):
            print(f"   {i}. {resort}: {visitors:.0f} average visitors")
        
        # Snowfall impact
        if 'snowfall_quality' in self.merged_df.columns:
            snowfall_correlation = self.merged_df['Visitors'].corr(self.merged_df['snowfall_quality'])
            print(f"\n4. SNOWFALL IMPACT:")
            print(f"   Correlation between snowfall quality and visitors: {snowfall_correlation:.3f}")
        
        # Model accuracy by resort
        print(f"\n5. MODEL ACCURACY BY RESORT:")
        resort_accuracy = self.prediction_analysis.groupby('Resort')['Error_Percentage'].apply(lambda x: x.abs().mean()).sort_values()
        for resort, error in resort_accuracy.items():
            print(f"   {resort}: {error:.1f}% average error")
        
        # Recommendations
        print(f"\n6. STRATEGIC RECOMMENDATIONS:")
        print("   • Focus marketing efforts during weeks 5-10 (peak season)")
        print("   • Invest in snowmaking infrastructure to improve snowfall quality")
        print("   • Develop intermediate terrain to attract more visitors")
        print("   • Consider lift modernization to increase resort capacity")
        print("   • Monitor weather patterns to predict visitor demand")
        
        return self
    
    def predict_future_visitors(self, resort, year, week, features_dict=None):
        """Predict visitor numbers for a specific resort, year, and week"""
        print(f"\nPredicting visitors for {resort} in year {year}, week {week}...")
        
        # Create a sample data point
        sample_data = self.X.iloc[0:1].copy()
        
        # Set basic features
        sample_data['Year'] = year
        sample_data['Week'] = week
        sample_data['Month'] = pd.to_datetime(f"{year}-W{week:02d}-1", format='%Y-W%W-%w').month
        sample_data['DayOfYear'] = pd.to_datetime(f"{year}-W{week:02d}-1", format='%Y-W%W-%w').dayofyear
        sample_data['Season'] = pd.to_datetime(f"{year}-W{week:02d}-1", format='%Y-W%W-%w').quarter
        sample_data['WeekOfSeason'] = week
        sample_data['PeakSeason'] = 1 if 5 <= week <= 10 else 0
        sample_data['SchoolHolidays'] = 1 if week in [1, 2, 6, 7, 15] else 0
        sample_data['year_trend'] = year - 2014
        sample_data['week_trend'] = week - 1
        
        # Set resort-specific features
        resort_data = self.resort_df[self.resort_df['Resort'] == resort].iloc[0]
        for feature in self.selected_features:
            if feature in resort_data.index:
                sample_data[feature] = resort_data[feature]
        
        # Override with custom features if provided
        if features_dict:
            for feature, value in features_dict.items():
                if feature in sample_data.columns:
                    sample_data[feature] = value
        
        # Fill NaN values with 0 for prediction
        sample_data = sample_data.fillna(0)
        
        # Scale and select features
        sample_scaled = self.scaler.transform(sample_data)
        sample_selected = self.feature_selector.transform(sample_scaled)
        
        # Make prediction
        prediction = self.model.predict(sample_selected)[0]
        
        print(f"Predicted visitors for {resort}: {prediction:.0f}")
        return prediction
    
    def run_complete_analysis(self):
        """Run the complete analysis pipeline"""
        print("Starting complete visitor prediction analysis...")
        
        (self.load_data()
         .preprocess_visitation_data()
         .preprocess_resort_features()
         .preprocess_snowfall_data()
         .merge_all_data()
         .create_features()
         .prepare_model_data()
         .train_model()
         .cross_validate_model()
         .analyze_predictions()
         .create_visualizations()
         .generate_insights())
        
        print("\nAnalysis completed successfully!")
        return self

def main():
    """Main function to run the analysis"""
    print("Australian Ski Resort Visitor Prediction Model")
    print("="*50)
    
    # Initialize and run the model
    predictor = SkiResortVisitorPredictor()
    predictor.run_complete_analysis()
    
    # Example predictions
    print("\n" + "="*50)
    print("EXAMPLE PREDICTIONS")
    print("="*50)
    
    # Predict for all 8 resorts
    resorts = ['Thredbo', 'Perisher', 'Mt. Buller', 'Falls Creek', 'Mt. Hotham', 'Mt. Baw Baw', 'Charlotte Pass', 'Selwyn']
    
    for resort in resorts:
        # Peak season prediction
        peak_prediction = predictor.predict_future_visitors(resort, 2025, 7)
        
        # Off-peak prediction
        off_peak_prediction = predictor.predict_future_visitors(resort, 2025, 2)
        
        print(f"\n{resort} Predictions for 2025:")
        print(f"  Peak season (week 7): {peak_prediction:.0f} visitors")
        print(f"  Off-peak (week 2): {off_peak_prediction:.0f} visitors")
        print(f"  Peak/Off-peak ratio: {peak_prediction/off_peak_prediction:.2f}x")
    
    # 2026 predictions
    print("\n" + "="*50)
    print("2026 PREDICTIONS")
    print("="*50)
    
    for resort in resorts:
        # Peak season prediction for 2026
        peak_prediction_2026 = predictor.predict_future_visitors(resort, 2026, 7)
        
        # Off-peak prediction for 2026
        off_peak_prediction_2026 = predictor.predict_future_visitors(resort, 2026, 2)
        
        print(f"\n{resort} Predictions for 2026:")
        print(f"  Peak season (week 7): {peak_prediction_2026:.0f} visitors")
        print(f"  Off-peak (week 2): {off_peak_prediction_2026:.0f} visitors")
        print(f"  Peak/Off-peak ratio: {peak_prediction_2026/off_peak_prediction_2026:.2f}x")
    
    print("\nAnalysis saved to visualizations folder:")
    print("- visualizations/actual_vs_predicted.png")
    print("- visualizations/feature_importance.png") 
    print("- visualizations/residuals_plot.png")
    print("- visualizations/resort_performance.png")
    print("- visualizations/weekly_trend.png")
    print("- visualizations/yearly_trend.png")
    print("- visualizations/combined_analysis.png")

if __name__ == "__main__":
    main() 