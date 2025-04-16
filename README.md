# 🏠 Real Estate Price Prediction – Bangalore

This project predicts house prices in Bangalore using machine learning. It includes end-to-end data preprocessing, feature engineering, model selection, and export for deployment.

---

## 📌 Features

✅ Cleans and preprocesses real estate data  
✅ Handles non-uniform values like `"2100 - 2850"` in square footage  
✅ Removes outliers based on domain knowledge (BHK to sqft ratio, bathroom sanity checks)  
✅ Creates derived features like `price_per_sqft`  
✅ One-hot encodes location data  
✅ Compares multiple ML models using `GridSearchCV`  
✅ Saves the best model and column metadata for deployment  
✅ Includes a custom function to predict price for any input

---

## 🗂 Dataset

- **File**: `bengaluru_house_prices.csv`
- **Source**: Kaggle / scraped real-estate listings
- **Key columns**:  
  - `location`, `size`, `total_sqft`, `bath`, `price`, etc.

---

## 📊 Tech Stack

- Python 🐍
- Pandas, NumPy, Matplotlib 📈
- Scikit-learn (ML) 🔍
- Jupyter Notebook 📓
- Pickle & JSON (model export) 💾

---

 🚀 Getting Started

 1. Clone this repo
```bash
git clone https://github.com/MayankJ03/Real-Estate-Price-Prediction.git
cd Real-Estate-Price-Prediction


2. Install dependencies

pip install pandas numpy matplotlib scikit-learn
3. Open Notebook
Launch RealEstatePricePrediction 2.ipynb in Jupyter Notebook or Colab and follow the code cells.

🧠 Model Used
LinearRegression (Best performing model after GridSearchCV)

Other models tested: Lasso, DecisionTreeRegressor


from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)
📈 Predict House Price
Use this function:


def predict_price(model, x, location, sqft, bath, bhk):
    loc_index = np.where(x.columns == location)[0][0]
    x_input = np.zeros(len(x.columns))
    x_input[0] = sqft
    x_input[1] = bath
    x_input[2] = bhk
    if loc_index >= 0:
        x_input[loc_index] = 1
    return model.predict([x_input])[0]
📍 Example:


predict_price(lr_clf, x, '1st Phase JP Nagar', 1000, 2, 2)
# Output: ~₹83.49 Lakhs
📦 Files
RealEstatePricePrediction 2.ipynb – Core notebook with full pipeline

Banglore_home_prices_model.pickle – Trained model

columns.json – Column metadata for deployment

✍️ Author
Mayank Jain
🎓 Final Year IT Student @ VIT Vellore
📫 LinkedIn

🌐 Future Work
Add Flask/Django Web UI

Deploy as REST API

Use advanced models (XGBoost, Random Forest)

Add CI/CD pipeline for model updates

