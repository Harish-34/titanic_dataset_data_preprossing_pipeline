import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy.stats as stats

# Set page config
st.set_page_config(page_title="Titanic Data Preprocessing Pipeline", layout="wide")

# Title
st.title("Titanic Survival Prediction Preprocessing Pipeline")
st.markdown("---")

# Data Description Section
st.header("Dataset Overview")
try:
    data = pd.read_csv('titanic.csv')  # Ensure this file path is correct
except FileNotFoundError:
    st.error("File 'titanic.csv' not found. Please upload the file.")
    st.stop()

col1, col2 = st.columns(2)
with col1:
    st.subheader("First 5 rows")
    st.write(data.head())
with col2:
    st.subheader("Data Dictionary")
    data_dict = {
        'Variable': ['Survived', 'Pclass', 'Name', 'Sex', 'Age', 'SibSp', 'Parch', 
                    'Ticket', 'Fare', 'Cabin', 'Embarked'],
        'Description': ['Survival (0 = No, 1 = Yes)', 'Passenger class', 
                        'Passenger name', 'Gender', 'Age in years', 
                        'Siblings/Spouses aboard', 'Parents/Children aboard',
                        'Ticket number', 'Passenger fare', 'Cabin number',
                        'Embarkation port (C=Cherbourg, Q=Queenstown, S=Southampton)']
    }
    st.table(pd.DataFrame(data_dict))

# Preprocessing Steps
st.header("Data Preprocessing Steps")
titanic = data.copy()

# Step 1: Remove Columns
with st.expander("1. Feature Selection", expanded=True):
    st.write(""" **Columns removed:** - Name: Unique identifiers don't affect survival - Ticket: High cardinality and irrelevant - Fare: Highly correlated with Pclass - Cabin: Too many missing values (>77%) """)
    cols_to_remove = ['Name', 'Ticket', 'Fare', 'Cabin']
    titanic.drop(cols_to_remove, axis=1, inplace=True)
    st.write("Data after removing columns:")
    st.write(titanic.head())

# Step 2: Handle Missing Values
with st.expander("2. Missing Value Treatment", expanded=True):
    st.subheader("Age Imputation")
    st.write("Strategy: Different mean values for survived/non-survived groups")
    
    # Calculate means
    meanS = titanic[titanic.Survived == 1].Age.mean()
    meanNS = titanic[titanic.Survived == 0].Age.mean()
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mean Age (Survived)", f"{meanS:.1f} years")
    with col2:
        st.metric("Mean Age (Not Survived)", f"{meanNS:.1f} years")
    
    # Imputation logic
    titanic['age'] = np.where(pd.isnull(titanic.Age) & (titanic.Survived == 1), meanS, titanic.Age)
    titanic.age.fillna(meanNS, inplace=True)
    titanic.drop('Age', axis=1, inplace=True)
    
    st.write("Data after age imputation:")
    st.write(titanic.head())

# Step 3: Categorical Encoding
with st.expander("3. Categorical Encoding", expanded=True):
    st.write(""" **Encoding Schemes:** - Sex: male=1, female=2 - Embarked: S=1, Q=2, C=3 """)
    
    # Gender encoding
    titanic['Sex'] = titanic['Sex'].apply(lambda x: 1 if x == 'male' else 2)
    
    # Embarked encoding
    embarked_mapping = {'S': 1, 'Q': 2, 'C': 3}
    titanic['Embarked'] = titanic['Embarked'].map(embarked_mapping)
    titanic.dropna(inplace=True)  # Remove remaining nulls in Embarked
    
    st.write("Encoded Data:")
    st.write(titanic.head())

# Visualization Section
st.header("Data Visualization")
col1, col2 = st.columns(2)

with col1:
    st.subheader("Survival Distribution")
    fig, ax = plt.subplots()
    titanic.Survived.value_counts().plot.pie(autopct='%1.1f%%', 
                                            colors=['#ff9999','#66b3ff'],
                                            ax=ax)
    st.pyplot(fig)

with col2:
    st.subheader("Age Distribution by Survival")
    fig, ax = plt.subplots()
    sns.histplot(data=titanic, x='age', hue='Survived', bins=20, 
                kde=True, palette='viridis', ax=ax)
    ax.set_xlabel("Age")
    st.pyplot(fig)

# Statistical Analysis Section
st.header("Statistical Analysis")

# Hypothesis Testing
st.subheader("Hypothesis Testing")
col1, col2 = st.columns(2)

with col1:
    st.write("**Chi-square Test for Categorical Variables**")
    contingency = pd.crosstab(titanic['Sex'], titanic['Survived'])
    chi2, p, _, _ = stats.chi2_contingency(contingency)
    st.write(f"Sex vs Survival: χ²={chi2:.2f}, p={p:.4f}")

with col2:
    st.write("**T-test for Age**")
    survived_age = titanic[titanic.Survived == 1]['age']
    not_survived_age = titanic[titanic.Survived == 0]['age']
    t_stat, p_val = stats.ttest_ind(survived_age, not_survived_age, nan_policy='omit')
    st.write(f"T-test results: t={t_stat:.2f}, p={p_val:.4f}")

# Feature Correlation
st.subheader("Feature Correlation Matrix")
fig, ax = plt.subplots(figsize=(10,6))
sns.heatmap(titanic.corr(), annot=True, cmap='coolwarm', ax=ax)
st.pyplot(fig)

# User Input Section
st.header("Survival Prediction")
st.write("Enter passenger details to see preprocessed features:")

col1, col2 = st.columns(2)
with col1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3])
    age = st.slider("Age", 0.0, 100.0, 30.0)
    sibsp = st.selectbox("Siblings/Spouses", [0, 1, 2, 3, 4, 5])

with col2:
    sex = st.selectbox("Sex", ["male", "female"])
    parch = st.selectbox("Parents/Children", [0, 1, 2, 3, 4, 5])
    embarked = st.selectbox("Embarked", ["S", "Q", "C"])

# Preprocess user input
user_data = pd.DataFrame([[pclass, sex, age, sibsp, parch, embarked]], 
                        columns=['Pclass', 'Sex', 'age', 'SibSp', 'Parch', 'Embarked'])

# Apply same preprocessing
user_data['Sex'] = user_data['Sex'].apply(lambda x: 1 if x == 'male' else 2)
user_data['Embarked'] = user_data['Embarked'].map(embarked_mapping)

st.subheader("Preprocessed Features for ML Model")
st.write(user_data)

st.markdown("---")
st.write("**Preprocessing Pipeline Complete** - This data is now ready for machine learning modeling!")
