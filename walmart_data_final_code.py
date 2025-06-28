import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.core.interchange.from_dataframe import categorical_column_to_series

# --- Loading and viewing the data set ---
df = pd.read_csv('walmart_data.csv')
# Check the first 5 rows
print(df.head())
# Check the shape of the data set
print("Shape:", df.shape)

# --- Exploring data types, info and Null values ---
# Check for column datatypes and non-null counts
print("\nInfo: ")
print(df.info())
# Check for missing values in each column
print("\nMissing Values:")
print(df.isnull().sum())

# --- Descriptive Statistics and Unique Value Counts---
# Get the summary statistics for numerical columns
print("\nDescribe (Numerical):")
print(df.describe())

# Get the summary statistics for categorical columns
print("\nDescribe (categorical):")
print(df.describe(include='object'))

# Unique Values per column
print("\nUnique values per column:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()}")

# --- Data Visualization and Outlier Detection ---
import matplotlib.pyplot as plt
import seaborn as sns

# Visualize Categorical Variables
categorical_cols = ['Gender', 'Age', 'City_Category', 'Stay_In_Current_City_Years', 'Marital_Status']
for col in categorical_cols:
    plt.figure(figsize = (6,3))
    sns.countplot(x=col, data=df)
    plt.title(f'Distribution of {col}')
    plt.show()

# Visualize Purchase Distribution
plt.figure(figsize=(8,4))
sns.histplot(df['Purchase'], kde = True, bins = 50)
plt.title(f'Purchase Amount Distribution')
plt.show()

# Boxplot for Outliers check
plt.figure(figsize=(6,4))
sns.boxplot(x='Gender', y='Purchase', data=df)
plt.title(f'Purchase Amount by Gender')
plt.show()

plt.figure(figsize=(8,4))
sns.boxplot(x='Age', y='Purchase', data=df)
plt.title(f'Purchase Amount by Age')
plt.show()

plt.figure(figsize=(6,4))
sns.boxplot(x='City_Category', y='Purchase', data=df)
plt.title(f'Purchase Amount by City_Category')
plt.show()

# --- Grouped Summaries (GroupBy Analysis) ---
# Mean Purchase by Gender
print("\nMean Purchase by Gender")
print(df.groupby('Gender')['Purchase']. mean())

# Mean Purchase by Age
print("\nMean Purchase by Age")
print(df.groupby('Age')['Purchase']. mean())

# Mean Purchase by Marital Status
print("\nMean Purchase by marital status")
print(df.groupby('Marital_Status')['Purchase']. mean())

# --- Confidence Interval for mean Purchase (Gender-wise) ---
import numpy as np
from scipy import stats

def conf_int_mean(data, conf=0.95):
    n = len(data)
    mean = np.mean(data)
    sem = stats.sem(data)
    margin =  sem * stats.t.ppf((1 + conf) / 2., n-1)
    return mean, mean - margin, mean + margin

# Confidence interval for Female Gender
f_purchases = df[df['Gender'] == 'F']['Purchase']
mean_f, ci_low_f, ci_high_f = conf_int_mean(f_purchases)
print(f"\nFemale: mean={mean_f:.2f}, 95% CI=({ci_low_f:.2f}, {ci_high_f:.2f})")

# Confidence interval for Male Gender
m_purchases = df[df['Gender'] == 'M']['Purchase']
mean_m, ci_low_m, ci_high_m = conf_int_mean(m_purchases)
print(f"\nMale: mean={mean_m:.2f}, 95% CI=({ci_low_m:.2f}, {ci_high_m:.2f})")

# --- Statistical Testing and Group Comparison ---

import numpy as np
from scipy.stats import ttest_ind, mannwhitneyu, f_oneway
from statsmodels.stats.weightstats import ztest
import matplotlib.pyplot as plt
import seaborn as sns

# Data selection
male_purchase = df[df['Gender'] == 'M']['Purchase']
female_purchase = df[df['Gender'] == 'F']['Purchase']

# 1. T-test (Male vs Female)
t_stat, p_value = ttest_ind(male_purchase, female_purchase, equal_var=False)
print(f"\nT-test: t={t_stat:.2f}, p-value={p_value:.4f}")

# 2. Z-test (Male vs Female)
z_stat, p_val = ztest(male_purchase, female_purchase, alternative='two-sided')
print(f"\nZ-test: z={z_stat:.2f}, p-value={p_val:.4f}")

# 3. Mann-Whitney U test (nonparametric)
u_stat, p_val_u = mannwhitneyu(male_purchase, female_purchase, alternative='two-sided')
print(f"\nMann-Whitney U test: U={u_stat:.2f}, p-value={p_val_u:.4f}")

# 4. Cohen’s d (effect size)
def cohens_d(a, b):
    return (a.mean() - b.mean()) / np.sqrt((a.std()**2 + b.std()**2) / 2)
d = cohens_d(male_purchase, female_purchase)
print(f"\nCohen's d: {d:.3f}")

# 5. ANOVA (Age groups)
groups = [group['Purchase'].values for name, group in df.groupby('Age')]
f_stat, p_value_anova = f_oneway(*groups)
print(f"\nANOVA: F={f_stat:.2f}, p-value={p_value_anova:.4f}")

# 6. Correlation Heatmap (Numeric Variables)
numeric_cols = ['Purchase', 'Product_Category', 'Occupation']
plt.figure(figsize=(6, 4))
sns.heatmap(df[numeric_cols].corr(), annot=True, cmap='coolwarm')
plt.title('Correlation Heatmap (Numeric Variables Only)')
plt.show()

# ---Feature Engineering: Gender and City Interactions---

# Create an interaction feature combining Gender and City_Category
df['Gender_City'] = df['Gender'] + '_' + df['City_Category']

# Mean purchase by Gender_City group
group_means = df.groupby('Gender_City')['Purchase'].mean()
print("\nMean Purchase by Gender_City group:")
print(group_means)

# Visualize purchase by Gender_City
import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(8,5))
sns.boxplot(x='Gender_City', y='Purchase', data=df)
plt.title('Purchase Amount by Gender and City Interactions')
plt.xticks(rotation=30)
plt.tight_layout()
plt.show()

# To Compare Female Vs Male in City A
fa = df[df['Gender_City'] == 'F_A']['Purchase']
ma = df[df['Gender_City'] == 'M_A']['Purchase']

# T-test between F_A and M_A
from scipy.stats import ttest_ind, _mannwhitneyu
t_stat, p_val = ttest_ind(fa, ma, equal_var=False)
print(f"T-test F_A vs M_A: t={t_stat:.2f}, p_value={p_val:.4f}")

# Mann-Whitney U test between F_A and M_A
u_stat, p_val_u = mannwhitneyu(fa, ma, alternative='Two-Sided' )
print(f"Mann-Whitney U F_A vs M_A: U={u_stat:.2f}, p-value={p_val_u:.4f}")








