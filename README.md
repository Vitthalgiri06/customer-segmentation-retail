# Retail Customer Segmentation

A machine learning project that groups online retail customers based on their shopping behavior to help businesses improve customer retention, marketing, and sales.

## Project Overview

This project uses customer purchase data to identify different customer segments based on:

- **Recency:** How recently a customer purchased.
- **Frequency:** How often a customer purchases.
- **Monetary:** How much a customer spends.
- **Product Preferences:** Which product categories a customer prefers.

The goal is to help businesses identify valuable customers, encourage repeat purchases, and target inactive customers.

## Dataset

**Source:** [UCI Online Retail Dataset](https://archive.ics.uci.edu/dataset/352/online+retail)

| Description           |               Value |
| --------------------- | ------------------: |
| Original transactions |             541,909 |
| Clean transactions    |             391,283 |
| Unique customers      |               4,334 |
| Time period           | Dec 2010 – Dec 2011 |

Data cleaning included removing missing customer IDs, duplicates, cancelled orders, and invalid transactions.

## Methodology

1. Data collection and cleaning
2. Customer feature engineering
3. Data preprocessing and scaling
4. Train, validation, and test split (60/20/20)
5. Clustering model training and evaluation
6. Customer segment analysis

## Models Compared

Three clustering algorithms were evaluated:

| Algorithm               | Test Silhouette Score |
| ----------------------- | --------------------: |
| K-Means (K=4)           |            **0.2724** |
| Hierarchical Clustering |                0.2376 |
| DBSCAN                  |                0.1838 |

**Selected Model: K-Means**

K-Means achieved the highest validation and test Silhouette Scores and produced four practical customer segments.

## Customer Segments

| Segment               | Customers | Revenue Share |
| --------------------- | --------: | ------------: |
| Champions (VIP)       |       811 |        67.84% |
| Loyal Regulars        |     1,184 |        20.42% |
| Recent Emerging       |       833 |         6.55% |
| At-Risk / Hibernating |     1,506 |         5.19% |

### Marketing Strategies

- **Champions:** VIP rewards and exclusive offers.
- **Loyal Regulars:** Loyalty programs and personalized recommendations.
- **Recent Emerging:** Encourage second purchases.
- **At-Risk:** Re-engagement campaigns and targeted discounts.

## Model Evaluation

- Test Silhouette Score: **0.2724**
- Borderline customers: **2.42%**
- Model stability (ARI): **0.8582**

## Technologies Used

- Python
- Pandas and NumPy
- Scikit-learn
- Matplotlib and Seaborn
- Streamlit
- Pytest

## Project Structure

```text
customer-segmentation-retail/
├── artifacts/
│   ├── plots/
│   ├── metrics_summary.json
│   └── segmentation_pipeline.pkl
├── data/
├── notebooks/
│   └── customer_segmentation.ipynb
├── src/
│   ├── data_loader.py
│   ├── preprocessing.py
│   └── train_evaluate.py
├── tests/
│   └── test_pipeline.py
├── app.py
├── requirements.txt
└── README.md
```

## How to Run

```bash
# Clone the repository
git clone <repository-url>

# Navigate to the project
cd customer-segmentation-retail

# Create virtual environment
python -m venv .venv

# Install dependencies
pip install -r requirements.txt

# Run tests
pytest -v

# Train and evaluate models
python -m src.train_evaluate

# Launch Streamlit app
streamlit run app.py
```

## Key Findings

- 18.71% of customers generate 67.84% of total revenue.
- Customer behavior can be divided into four meaningful segments.
- K-Means performed better than the other tested clustering models.
- Customer segments can support more effective marketing strategies.

## Limitations

- Approximately 25% of original transactions have missing customer IDs.
- Most customers are based in the UK.
- The dataset covers only one year.
- Wholesale purchases can influence spending patterns.

## Responsible Use

Customer segmentation should support better recommendations and customer loyalty, not unfair pricing or service restrictions. Customer privacy and applicable data protection regulations must be respected.

## Conclusion

This project demonstrates how machine learning can transform retail transaction data into actionable customer insights. The resulting segments help businesses identify VIP customers, improve retention, encourage repeat purchases, and create targeted marketing campaigns.

---

**Author:** Vitthal Giri
**Project:** Retail Customer Segmentation
