# 📊 Data Wrangling & Health Data Analysis: NHANES Body Measurements

A comprehensive data wrangling and statistical analysis project analyzing body measurements from the National Health and Nutrition Examination Survey (NHANES) dataset.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![NumPy](https://img.shields.io/badge/NumPy-Data%20Analysis-orange.svg)
![SciPy](https://img.shields.io/badge/SciPy-Statistics-green.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 📋 Overview

This project demonstrates advanced **data wrangling** techniques applied to real-world health survey data from the CDC's NHANES program. The analysis covers 8,304 adult participants (4,222 females and 4,082 males), exploring relationships between body measurements and computing health indicators like BMI.

### Key Skills Demonstrated:
- 🔧 **Data Loading** - Multi-column CSV parsing with NumPy
- 🧹 **Data Cleaning** - Handling missing values (NaN) across datasets
- 📐 **Feature Engineering** - BMI calculation from height/weight
- 📊 **Statistical Analysis** - Descriptive statistics, correlation analysis
- 🔬 **Comparative Analysis** - Gender-based statistical comparisons
- 📈 **Data Visualization** - Scatter plot matrices, box plots, correlation heatmaps

## 🎯 Project Objectives

1. Load and parse multi-dimensional health survey data
2. Handle missing values across multiple columns
3. Engineer new features (BMI calculation)
4. Perform gender-comparative statistical analysis
5. Analyze correlations between body measurements
6. Apply data standardization (z-score normalization)
7. Visualize relationships using scatter plot matrices

## 📊 Dataset Information

### Source
**National Health and Nutrition Examination Survey (NHANES)**
- Collected by: CDC (Centers for Disease Control)
- Period: 2017-2020
- Published: May 2021

### Data Dimensions

| Dataset | Rows | Columns |
|---------|------|---------|
| Adult Females | 4,222 | 7 |
| Adult Males | 4,082 | 7 |
| **Total** | **8,304** | **7** |

### Variables (Columns)

| Index | Variable | Description | Unit |
|-------|----------|-------------|------|
| 0 | Weight | Body weight | kg |
| 1 | Height | Standing height | cm |
| 2 | Upper Arm Length | Arm measurement | cm |
| 3 | Upper Leg Length | Leg measurement | cm |
| 4 | Arm Circumference | Arm circumference | cm |
| 5 | Hip Circumference | Hip measurement | cm |
| 6 | Waist Circumference | Waist measurement | cm |
| 7* | BMI | Body Mass Index (calculated) | kg/m² |

*BMI is engineered from weight and height

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/YOUR_USERNAME/data-wrangling-health-analysis.git
cd data-wrangling-health-analysis
pip install -r requirements.txt
```

### Basic Usage

```python
import numpy as np

# Load NHANES data
female = np.genfromtxt('nhanes_adult_female_bmx_2020.csv',
                       delimiter=',', skip_header=1)
male = np.genfromtxt('nhanes_adult_male_bmx_2020.csv',
                     delimiter=',', skip_header=1)

print(f"Female dataset: {female.shape}")
print(f"Male dataset: {male.shape}")

# Calculate BMI
weight = female[:, 0]
height = female[:, 1] / 100  # Convert cm to meters
bmi = weight / (height ** 2)
```

## 📁 Project Structure

```
data-wrangling-health-analysis/
├── README.md
├── requirements.txt
├── nhanes_analysis.py           # Main analysis module
├── nhanes_health_analysis.ipynb # Jupyter notebook
├── data/
│   ├── nhanes_adult_female_bmx_2020.csv
│   └── nhanes_adult_male_bmx_2020.csv
└── visualizations/
    ├── bmi_boxplot.png
    ├── scatter_matrix.png
    └── correlation_heatmap.png
```

## 🔬 Analysis Components

### 1. Data Loading & Validation

```python
# Load data with proper handling
data = np.genfromtxt('nhanes_adult_female_bmx_2020.csv',
                     delimiter=',',
                     skip_header=1)

# Check for missing values
nan_count = np.isnan(data).sum()
nan_per_column = np.isnan(data).sum(axis=0)
print(f"Total NaN values: {nan_count}")
```

### 2. Feature Engineering - BMI Calculation

```python
def calculate_bmi(weight_kg, height_cm):
    """
    Calculate Body Mass Index.
    
    BMI = weight (kg) / height (m)²
    """
    height_m = height_cm / 100
    bmi = weight_kg / (height_m ** 2)
    return bmi

# Apply to dataset
bmi = calculate_bmi(data[:, 0], data[:, 1])
```

### 3. Data Standardization (Z-Score)

```python
def standardize(data):
    """
    Apply z-score standardization.
    
    z = (x - mean) / std
    """
    mean = np.nanmean(data, axis=0)
    std = np.nanstd(data, axis=0)
    return (data - mean) / std
```

### 4. Correlation Analysis

```python
# Calculate Pearson correlation
pearson_corr = np.corrcoef(data.T)

# Calculate Spearman correlation
from scipy import stats
spearman_corr = stats.spearmanr(data)[0]
```

## 📈 Key Findings

### BMI Statistics by Gender

| Statistic | Female | Male |
|-----------|--------|------|
| **Mean** | 30.10 | 29.14 |
| **Median** | 28.89 | 28.27 |
| **Std Dev** | 7.76 | 6.31 |
| **Min** | 14.20 | 14.91 |
| **Max** | 67.04 | 66.50 |
| **IQR** | 10.01 | 7.73 |
| **Skewness** | 0.92 | 0.97 |

### BMI Classification Distribution

| Category | BMI Range | Interpretation |
|----------|-----------|----------------|
| Underweight | < 18.5 | Below healthy range |
| Normal | 18.5 - 24.9 | Healthy weight |
| Overweight | 25.0 - 29.9 | Above healthy range |
| Obese | ≥ 30.0 | Health risk |

### Correlation Analysis (Males)

| Variables | Pearson r | Interpretation |
|-----------|-----------|----------------|
| Weight ↔ Hip | 0.942 | Very strong positive |
| Weight ↔ BMI | 0.929 | Very strong positive |
| Waist ↔ BMI | 0.924 | Very strong positive |
| Hip ↔ BMI | 0.925 | Very strong positive |
| Height ↔ BMI | 0.080 | Very weak positive |

### Key Insights

1. **Gender Differences**:
   - Females show higher BMI variability (std: 7.76 vs 6.31)
   - Both genders show right-skewed distributions (positive skewness)

2. **Strong Correlations**:
   - Weight, waist, hip, and BMI are highly correlated (r > 0.89)
   - Height has weak correlation with BMI (r ≈ 0.08)

3. **Health Implications**:
   - Mean BMI > 29 indicates population-level overweight trend
   - High correlation between waist and BMI supports waist-to-height ratio as health indicator

## 📊 Visualizations

### Box Plot - BMI by Gender
- Shows distribution comparison between males and females
- Identifies outliers (extreme BMI values)
- Horizontal layout for easy comparison

### Scatter Plot Matrix
- 5×5 matrix: Height, Weight, Waist, Hip, BMI
- Diagonal shows variable names
- Off-diagonal shows pairwise relationships

### Correlation Heatmap
- Visual representation of correlation matrix
- Color-coded for quick interpretation
- Pearson vs Spearman comparison

## 🔧 Data Wrangling Techniques

| Technique | Implementation |
|-----------|----------------|
| **Multi-column Loading** | `np.genfromtxt()` with column selection |
| **NaN Handling** | `np.isnan()`, boolean indexing |
| **Feature Engineering** | BMI calculation from raw measurements |
| **Standardization** | Z-score normalization |
| **Correlation Analysis** | Pearson and Spearman coefficients |
| **Outlier Detection** | IQR method with box plots |
| **Gender Comparison** | Parallel statistical analysis |

## 🎓 Learning Outcomes

This project demonstrates proficiency in:

1. **NumPy Operations** - Multi-dimensional array manipulation
2. **Data Quality** - Handling missing values in health data
3. **Feature Engineering** - Creating derived health metrics
4. **Statistical Analysis** - Comparative and correlation analysis
5. **Data Visualization** - Multi-panel plots and matrices
6. **Health Data Science** - Understanding body measurement relationships

## 👨‍💼 Author

**Victor Prefa**
- Medical Doctor & Data Scientist
- MSc Data Science & Business Analytics, Deakin University
- Student ID: 225187913

## 📚 References

1. NHANES Dataset - https://www.cdc.gov/nchs/nhanes/
2. CDC Body Measurements - https://wwwn.cdc.gov/nchs/nhanes/
3. BMI Classification - WHO Guidelines
4. NumPy Documentation - https://numpy.org/doc/
5. SciPy Statistics - https://docs.scipy.org/doc/scipy/reference/stats.html

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

*This project was developed as part of the Data Science coursework at Deakin University, demonstrating practical data wrangling skills applied to real-world health survey data.*
