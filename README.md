# 📱 iPhone Resale Depreciation Model

## 🎯 Project Objective

How much does an iPhone lose in value the exact day a new generation is announced?

This project builds a predictive machine learning pipeline to forecast the exact resale value of an iPhone based on its age, specifications, and the current market lifecycle. Leveraging real, scraped e-commerce data from the 2026 post-iPhone 17 launch window, this analysis calculates the "Penalty" of not having the newest phone and identifies which features actually hold value on the secondary market.

## 📊 The Dataset

Unlike synthetic or estimated datasets, this data captures **active, real-world iPhone resale listings** from publicly available e-commerce platforms in the USA (2026).

-   **Scope:** Spans six generations (iPhone 12 through iPhone 17) across 23 distinct model variants (Pro, Pro Max, Plus, Mini) and all major storage tiers (64GB–1TB).
-   **Unique Feature:** Includes a rare demand signal column (`sold`), revealing not just what sellers are asking, but what buyers are _actually_ paying for.
-   **Context:** The 2026 collection window captures the post-iPhone 17 launch market, making this one of the first analyses to include Gen 17 resale pricing alongside legacy generations.

## 🛠️ Methodology & Tech Stack

-   **Data Cleaning:** Robust handling of messy raw data, including European decimal formats (`329,99`), unescaped string characters, and the removal of scam/fake listings (e.g., "$5000 TikTok phones").
-   **Feature Engineering:** Calculated relative phone age (`age_years`) against Gen 17, extracted boolean Pro-status, and calculated storage tier deltas.
-   **Visualization:** Matplotlib & Seaborn for aggregate depreciation curves and segmented trajectory analysis.
-   **Modeling:** `RandomForestRegressor` trained to predict secondary market pricing based on hardware specs and lifecycle age.

## 📈 Key Visualizations

1.  **The Aggregate Depreciation Curve:** Average resale price vs. years since release.
2.  **Depreciation by Model Type:** Trajectory comparison between Standard/Plus/Mini vs. Pro/Pro Max models.
3.  **The Storage Premium:** Boxplot distribution analyzing if Apple's "Storage Tax" holds up on the secondary market (Focused on iPhone 14 Used).

## 💡 Key Business Insights

The trained model and feature importance analysis revealed fascinating consumer behaviors:

-   **Time is the ultimate killer:** Phone age (`age_years`) accounts for **~75.5%** of the depreciation variance.
-   **The "Pro" Halo Effect is Real:** Being a Pro/Pro Max model accounts for **~18.2%** of the value retention, proving the Pro badge holds significant weight years after release.
-   **The Storage Tax is a Myth (Resale):** Storage capacity (`storage_gb_numeric`) accounts for less than **4%** of price variance. **Conclusion:** Consumers who buy 256GB/512GB models to "hold resale value" are largely losing money, as the market does not reward higher storage tiers proportionally.

## 📈 Model Performance

-   **Algorithm:** Random Forest Regressor
-   **Mean Absolute Error (MAE):** $99.58 _(On average, the model is off by less than $100 for any given listing)_
-   **R-squared:** 0.642 _(An exceptionally strong score given the irrational pricing inherent in peer-to-peer e-commerce markets)_

## 🚀 How to Run

1.  Clone this repository:
    
    ```bash
    git clone https://github.com/yourusername/iphone_depreciation.git
    ```
    

1.  Navigate to the directory and create a virtual environment:
    
    ```
    cd iphone\_depreciation
    ```
   ```bash
   python \-m venv venv
```
   
2.  Activate the environment (Windows):
    
    ```bash
    
    .\\venv\\Scripts\\activate
    ```
    
3.  Install required dependencies:
    
    ```bash
    
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```
    
4.  Ensure your raw CSV dataset is in the root directory and run the analysis:
    
    ```bash
    
    python analysis.py
    ```

_(Output will print model metrics to the console and save 3 high-res `.png` charts to the folder)._

## 📁 Repository Structure

```text

iphone\_depreciation/

│

├── analysis.py # Main python script (Cleaning, EDA, ML)

├── ecommerce\_iphone\_resale... # Raw CSV data

├── depreciation\_curve.png # Generated Viz 1

├── depreciation\_by\_type.png # Generated Viz 2

├── storage\_premium.png # Generated Viz 3

└── README.md # This file
```
## 📌 Future Improvements

-   **Incorporate `condition`:** Add "New", "Used", "For Parts" to the ML model to further reduce the MAE.
-   **Geographic Heatmaps:** Utilize the `us_state` column to map regional pricing variations (e.g., does a phone sell for more in California vs. Ohio?).
-   **Algorithm Tuning:** Test XGBoost or Polynomial Regression to better capture the non-linear drop in value the exact month a new phone drops.
