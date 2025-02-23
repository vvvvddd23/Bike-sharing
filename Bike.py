import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.metrics import mean_squared_error, r2_score
import streamlit as st

# 1. Încărcarea setului de date
day_data = pd.read_csv('day.csv')
hour_data = pd.read_csv('hour.csv')

# 2. Preprocesarea datelor
numeric_columns = day_data.select_dtypes(include=['int64', 'float64']).columns
for col in numeric_columns:
    Q1 = day_data[col].quantile(0.25)
    Q3 = day_data[col].quantile(0.75)
    IQR = Q3 - Q1
    day_data = day_data[~((day_data[col] < (Q1 - 1.5 * IQR)) | (day_data[col] > (Q3 + 1.5 * IQR)))]

# Pregătirea datelor pentru antrenare
X = day_data[['temp', 'hum', 'windspeed']]
y = day_data['cnt']

# Împărțirea setului în antrenare și testare
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardizarea datelor
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# a) Modelul de regresie liniară
lin_reg = LinearRegression()
lin_reg.fit(X_train_scaled, y_train)

# b) Modelul k-NN
knn = KNeighborsRegressor(n_neighbors=5)
knn.fit(X_train_scaled, y_train)

# 4. Compararea performanțelor modelelor
y_pred_lin = lin_reg.predict(X_test_scaled)
y_pred_knn = knn.predict(X_test_scaled)

mse_lin = mean_squared_error(y_test, y_pred_lin)
r2_lin = r2_score(y_test, y_pred_lin)

mse_knn = mean_squared_error(y_test, y_pred_knn)
r2_knn = r2_score(y_test, y_pred_knn)

# 5. Interfață Streamlit
st.title('Bike Sharing Prediction')
st.write('Comparația modelelor de regresie')
# 6. Afișarea unor exemple din setul de date
st.subheader('Exemple din setul de date')

# Afișarea primelor 10 rânduri din setul de date day_data
st.dataframe(day_data.head(10))  # Aceasta va afișa primele 10 rânduri din 'day.csv'

# Dacă vrei să afișezi și setul de date pentru 'hour.csv'
st.subheader('Exemple din setul de date pe oră')
st.dataframe(hour_data.head(10))  # Aceasta va afișa primele 10 rânduri din 'hour.csv'


# Afișarea rezultatelor în interfață
st.write(f'Regresie Liniară - MSE: {mse_lin:.2f}, R2: {r2_lin:.2f}')
st.write(f'k-NN - MSE: {mse_knn:.2f}, R2: {r2_knn:.2f}')


# Graficul predicțiilor
fig, ax = plt.subplots(1, 2, figsize=(14, 6))
ax[0].scatter(y_test, y_pred_lin, color='blue')
ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax[0].set_title('Regresie Liniară')
ax[0].set_xlabel('Valori reale')
ax[0].set_ylabel('Predicții')

ax[1].scatter(y_test, y_pred_knn, color='green')
ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
ax[1].set_title('k-NN')
ax[1].set_xlabel('Valori reale')
ax[1].set_ylabel('Predicții')

st.pyplot(fig)

# Funcționalitate interactivă pentru predicție în timp real
st.write(' Introduceți valorile pentru a prezice numărul de biciclete închiriate')

# Introducerea valorilor de către utilizator
temp = st.slider('Temperatura (°C)', min_value=0, max_value=40, value=20)
hum = st.slider('Umiditatea (%)', min_value=0, max_value=100, value=50)
windspeed = st.slider('Viteza vântului (m/s)', min_value=0, max_value=20, value=5)

# Crearea unui dataframe pentru predicție
input_data = pd.DataFrame([[temp, hum, windspeed]], columns=['temp', 'hum', 'windspeed'])

# Standardizarea valorilor introduse
input_scaled = scaler.transform(input_data)

# Predicția folosind cele două modele
pred_lin = lin_reg.predict(input_scaled)
pred_knn = knn.predict(input_scaled)

# Afișarea predicției
st.write(f"Predicția modelului de regresie liniară: {pred_lin[0]:.2f} biciclete închiriate")
st.write(f"Predicția modelului k-NN: {pred_knn[0]:.2f} biciclete închiriate")
