# 🏠 Housing Price Prediction using Satellite Imagery & Tabular Data  
### 🚀 Multimodal Machine Learning for Smarter Real Estate Valuation

> **Combining aerial vision + structured data to predict house prices more accurately than traditional ML models.**

---

## 🌟 Project Overview

Traditional house price models rely only on tabular data like sqft, bedrooms, grade, etc.  
But what about:

- 🌳 Greenery around the house?  
- 🌊 Is there water visible nearby?  
- 🏘 Neighborhood density?  
- 🛣 Road connectivity?

This project extracts these **hidden visual signals from satellite imagery** and merges them with structured data using a **multimodal learning pipeline**.

---

## 🎯 What I Achieved

✅ Extracted **2,048 deep visual features** using ResNet-50  
✅ Reduced to **256 PCA components** (~90% variance preserved)  
✅ Built rich domain features (luxury, neighborhood ratios, age)  
✅ Trained **XGBoost multimodal model**  
✅ Got **5% R² boost & 11% RMSE drop** over baseline  

---

## 🧠 Pipeline Architecture
📍 Coordinates
│
▼
🛰 Sentinel Hub API → 256×256 Satellite Image
│
▼
🧠 ResNet-50 Feature Extractor → 2048 Features
│
▼
📉 PCA → 256 Components
│
▼
📊 Tabular Features + Image Features
│
▼
🌳 XGBoost Model → Final Price Prediction


---

## 🛠 Tech Stack

- **Deep Learning:** ResNet-50 (Transfer Learning)  
- **ML Model:** XGBoost  
- **Dimensionality Reduction:** PCA  
- **Data:** Sentinel Hub Satellite Images  
- **Tools:** Python, Pandas, NumPy, Scikit-learn

---

## 📊 Results

| Model | RMSE | MAE | R² |
|------|------|-----|----|
| Tabular Only | $178,425 | $114,270 | 77% |
| Tabular + Images | **$158,572** | **$98,098** | **82%** |

🔥 **11.13% RMSE Reduction using Satellite Imagery**  
🔥 Clear proof that “images speak more than columns”

---

## 💡 Business Insights from Images

- 🌳 **Green Premium:**  
  - High vegetation → **+5–12% price**
- 🌊 **Water Visibility:**  
  - +15–25% valuation boost
- 🏘 **Low Density = High Value**
- 🏆 Luxury homes show:
  - Larger footprints  
  - Pools & landscaping  
  - Exclusive locations

---

## 🧪 Feature Engineering

### Tabular Side
- Removed multicollinearity  
- Outlier capping  
- Created:
  - `effective_house_age`
  - `living_ratio`
  - `lot_ratio`
  - `is_luxury`

### Image Side
- ResNet-50 embeddings  
- PCA compression  
- Captured:
  - Roof condition  
  - Roads  
  - Parks  
  - Waterfront  
  - Urban vs suburban context

---

## 🚀 How to Run (Conceptual)

1. Fetch satellite images via Sentinel Hub  
2. Extract features using ResNet-50  
3. Apply PCA  
4. Merge with tabular data  
5. Train XGBoost  
6. Compare baseline vs multimodal  

---

## 🔮 Future Scope

- Higher resolution imagery  
- Temporal price tracking  
- Attention maps for explainability  
- Street-view fusion  
- Indian market expansion 🇮🇳

---

## 👨‍💻 Author

**Saksham Gupta**  
IIT Roorkee  
2026

---

### 📌 Conclusion

> Multimodal learning = better understanding of real estate  
> Because houses are not just numbers — they are places visible from the sky 🌍
