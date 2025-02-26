import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import streamlit as st

# Încărcarea datelor
day_df = pd.read_csv("day.csv")
hour_df = pd.read_csv("hour.csv")

# Preprocesare date
st.title("Bike Sharing Data Analysis")
st.subheader("Primele 5 rânduri din setul de date")
st.dataframe(day_df.head())

st.subheader("Statistici descriptive")
st.dataframe(day_df.describe())

# Tratarea valorilor lipsă
missing_values = day_df.isnull().sum()
missing_values = missing_values[missing_values > 0].reset_index()
missing_values.columns = ["Coloană", "Număr valori lipsă"]
st.subheader("Valori lipsă")
st.dataframe(missing_values)

# Eliminarea valorilor lipsă înainte de vizualizare
day_df_cleaned = day_df.dropna()

# Excluderea coloanelor non-numerice
numeric_cols = day_df_cleaned.select_dtypes(include=[np.number])

# Vizualizare corelații între variabile
st.subheader("Matricea de corelație")
fig, ax = plt.subplots(figsize=(10, 6))
corr_matrix = numeric_cols.corr()
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# Detectare și eliminare outlieri
st.subheader("Distribuția variabilelor")
fig, ax = plt.subplots(figsize=(10, 6))
sns.histplot(day_df_cleaned['cnt'], bins=30, kde=True, ax=ax)
ax.set_title("Distribuția numărului de biciclete închiriate")
st.pyplot(fig)

# Model de regresie liniară
X = day_df_cleaned[['temp', 'hum', 'windspeed']]
y = day_df_cleaned['cnt']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

lin_reg = LinearRegression()
lin_reg.fit(X_train, y_train)
y_pred_lin = lin_reg.predict(X_test)

st.subheader("Regresie Liniară")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred_lin)
ax.set_xlabel("Valori reale")
ax.set_ylabel("Predicții")
ax.set_title("Regresie liniară")
st.pyplot(fig)

# Model k-NN
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train, y_train)
y_pred_knn = knn.predict(X_test)

st.subheader("Model K-NN")
fig, ax = plt.subplots()
ax.scatter(y_test, y_pred_knn)
ax.set_xlabel("Valori reale")
ax.set_ylabel("Predicții")
ax.set_title("K-NN Regression")
st.pyplot(fig)

# Compararea performanțelor modelelor
def evaluate_model(y_true, y_pred, model_name):
    return {
        'Model': model_name,
        'MAE': mean_absolute_error(y_true, y_pred),
        'MSE': mean_squared_error(y_true, y_pred),
        'R2 Score': r2_score(y_true, y_pred)
    }

models_performance = [
    evaluate_model(y_test, y_pred_lin, "Regresie Liniară"),
    evaluate_model(y_test, y_pred_knn, "K-NN")
]

performance_df = pd.DataFrame(models_performance)
st.subheader("Compararea performanțelor")
st.dataframe(performance_df)

fig, ax = plt.subplots()
sns.barplot(x='Model', y='R2 Score', data=performance_df, ax=ax)
ax.set_title("Compararea performanțelor modelelor")
st.pyplot(fig)

st.subheader("Predicție concretă bazată pe temperatură, umiditate și viteza vântului")
input_temp = st.slider("Temperatura (scalată între 0 și 1):", 0.0, 1.0, 0.5)
input_hum = st.slider("Umiditatea (scalată între 0 și 1):", 0.0, 1.0, 0.5)
input_wind = st.slider("Viteza vântului (scalată între 0 și 1):", 0.0, 1.0, 0.2)

predicted_cnt = lin_reg.predict([[input_temp, input_hum, input_wind]])[0]
st.write(f"Numărul estimat de biciclete închiriate: {predicted_cnt:.0f}")