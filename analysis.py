import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

# ==========================================
# 1. ROBUST CSV PARSING
# ==========================================
print("Loading and cleaning raw CSV data...")

file_name = 'ecommerce_iphone_resale_market_intelligence_usa_2026.csv'

# FIX: Added sep=';' to tell pandas the file uses semicolons, not commas!
df = pd.read_csv(file_name, engine='python', sep=';', on_bad_lines='skip')

print(f"Loaded {len(df)} rows.")
print("Columns found:", df.columns.tolist())

# ==========================================
# 2. ROBUST CLEANING
# ==========================================
print("Cleaning data...")

# CLEANING STEP 1: Fix European comma decimals in the price column (e.g., "329,99" -> 329.99)
df['price'] = df['price'].astype(str).str.replace('.', '', regex=False).str.replace(',', '.', regex=False).astype(float)

# CLEANING STEP 2: Remove obvious scam/fake listings
df = df[df['price'] < 2500]

# CLEANING STEP 3: Drop rows missing key numeric data
df = df.dropna(subset=['storage_gb_numeric', 'generation_number'])

# Ensure correct data types
df['generation_number'] = df['generation_number'].astype(int)
df['storage_gb_numeric'] = df['storage_gb_numeric'].astype(int)

# CLEANING STEP 4: Remove "Lot" listings. 
df = df[~df['title'].str.contains('Lot of|Lot', case=False, na=False)]

# Let's drop the rows where 'sold' is blank for visualizations, but keep them for the ML model 
df['sold'] = pd.to_numeric(df['sold'], errors='coerce').fillna(0)

print(f"Data cleaned. {len(df)} valid rows remaining.")

# ==========================================
# 3. FEATURE ENGINEERING
# ==========================================
print("Engineering features...")

# Calculate Age relative to Gen 17
df['age_years'] = 17 - df['generation_number']

# Convert boolean 'is_pro' to integer (True=1, False=0)
df['is_pro'] = df['is_pro'].astype(int)

# Clean the condition column for plotting
df['condition_clean'] = df['condition'].fillna('Unknown')

# ==========================================
# 4. EXPLORATORY DATA ANALYSIS
# ==========================================
print("Generating visualizations...")

sns.set_theme(style="whitegrid")

# Plot 1: The Aggregate Depreciation Curve
plt.figure(figsize=(10, 6))
avg_price_by_age = df.groupby('age_years')['price'].mean().reset_index()
sns.lineplot(data=avg_price_by_age, x='age_years', y='price', marker='o', linewidth=3, color='blue')
plt.title('iPhone Depreciation Curve (Average Resale Price vs. Age)', fontsize=14)
plt.xlabel('Years Since Release (0 = iPhone 17)', fontsize=12)
plt.ylabel('Average Resale Price ($)', fontsize=12)
plt.xticks(range(0, df['age_years'].max() + 1))
plt.tight_layout()
plt.savefig('depreciation_curve.png', dpi=300)
plt.show()

# Plot 2: Depreciation by Model Type (Pro vs Non-Pro)
plt.figure(figsize=(12, 6))
dep_by_type = df.groupby(['age_years', 'is_pro'])['price'].mean().reset_index()
dep_by_type['Model Type'] = dep_by_type['is_pro'].map({0: 'Standard/Plus/Mini', 1: 'Pro/Pro Max'})

sns.lineplot(data=dep_by_type, x='age_years', y='price', hue='Model Type', marker='o', linewidth=2.5, palette='Set2')
plt.title('Depreciation Trajectories by Model Type', fontsize=14)
plt.xlabel('Years Since Release', fontsize=12)
plt.ylabel('Average Resale Price ($)', fontsize=12)
plt.xticks(range(0, df['age_years'].max() + 1))
plt.legend(title='Model Type')
plt.tight_layout()
plt.savefig('depreciation_by_type.png', dpi=300)
plt.show()

# Plot 3: The Storage Premium 
plt.figure(figsize=(10, 6))
df_14 = df[(df['generation_number'] == 14) & (df['condition_clean'] == 'Used') & (df['is_pro'] == False)]
sns.boxplot(data=df_14, x='storage_gb_numeric', y='price', color='purple')
plt.title('iPhone 14 (Used) - Does Apple\'s Storage Tax hold up?', fontsize=14)
plt.xlabel('Storage (GB)', fontsize=12)
plt.ylabel('Listing Price ($)', fontsize=12)
plt.tight_layout()
plt.savefig('storage_premium.png', dpi=300)
plt.show()

# ==========================================
# 5. PREDICTIVE MODELING
# ==========================================
print("Training Machine Learning Model...")

# Define features (X) and target (y)
features = ['generation_number', 'storage_gb_numeric', 'age_years', 'is_pro']
X = pd.get_dummies(df[features], columns=['generation_number'], drop_first=True)
y = df['price']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest Regressor
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Evaluate
predictions = model.predict(X_test)
mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print(f"\n--- Model Performance ---")
print(f"Mean Absolute Error: ${mae:.2f} (On average, our model is off by this much)")
print(f"R-squared: {r2:.3f} (An R2 of ~0.85+ is excellent for messy marketplace data)")

# ==========================================
# 6. BUSINESS INSIGHTS (Feature Importance)
# ==========================================
print("\n--- Feature Importances ---")
importances = model.feature_importances_
feature_names = X.columns
feature_importance_df = pd.DataFrame({'Feature': feature_names, 'Importance': importances})
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)
print(feature_importance_df.to_string(index=False))

print("\n✅ Analysis Complete! Check your folder for the generated PNG charts.")