import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pandas as pd

df = pd.read_csv(r'C:\Users\Tech\OneDrive\Desktop\co2 emission\CO2 Emissions_Canada.csv')
df.columns
print(df.head(10))
columns_to_drop = ['Make', 'Model', 'Vehicle Class', 'Cylinders', 'Transmission']
X = df.drop(columns=columns_to_drop)
y = df["CO2 Emissions(g/km)"]
print(X.shape)
print(y.shape)
X.columns
np.unique(X['Fuel Type'])
X.isnull().sum()

import plotly.graph_objects as go
import plotly.express as px


def plot_bar_graphs(df, columns):
    for column in columns:
        plt.figure(figsize=(15, 5))
        ax = sns.countplot(x=column, data=df, order=df[column].value_counts().index)
        ax.bar_label(ax.containers[0],rotation=45)
        plt.xlabel(column, fontsize=15)
        plt.ylabel('Count', fontsize=15)
        plt.title(f'Bar Graph of {column}', fontsize=20)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.show()

cat_features = [ 'Vehicle Class', 'Engine Size(L)', 'Cylinders','Transmission', 'Fuel Type']

plot_bar_graphs(df, cat_features)
from sklearn.preprocessing import LabelEncoder

le = LabelEncoder()


X['Fuel Type'] = le.fit_transform(X['Fuel Type'])
X.head(10)
correlation_matrix = X.corr()

plt.figure(figsize=(10,8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix Heatmap')
plt.show()



numerical_df = X.select_dtypes(include=['number'])

plt.figure(figsize=(15, 10))

num_vars = len(numerical_df.columns)

for i, var in enumerate(numerical_df.columns, 1):
    plt.subplot((num_vars // 3) + 1, 3, i)
    sns.histplot(data=df, x=var, kde=True)
    plt.title(f'Distribution of {var}')

plt.tight_layout()
plt.show()


plt.figure(figsize=(8, 5))
sns.histplot(data=X, x='Fuel Consumption Hwy (L/100 km)', kde=True, label = "Fuel Consumption in Highway",color = "orange")
sns.histplot(data=X, x='Fuel Consumption City (L/100 km)', kde=True, label = "Fuel Consumption in City")
plt.xlabel('Consumption (L/100 km)', fontsize=10)
plt.ylabel('Frequency', fontsize=10)
plt.title(f'Histogram of Highway and City', fontsize=10)
plt.legend()
plt.show()

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

r2_sc= {}

from sklearn.linear_model import LinearRegression, Lasso, Ridge

lr_model = LinearRegression()
lr_model.fit(x_train, y_train)


y_pred_lr = lr_model.predict(x_test)


mse_lr = mean_squared_error(y_test, y_pred_lr)
r2_lr = r2_score(y_test, y_pred_lr)

print(f"Linear Regression - Mean Squared Error: {mse_lr}")
print(f"Linear Regression - R^2 Score: {r2_lr}")
r2_sc['Linear Regression'] = r2_lr

lasso_model = Lasso(alpha=0.01)  
lasso_model.fit(x_train, y_train)


y_pred_lasso = lasso_model.predict(x_test)
mse_lasso = mean_squared_error(y_test, y_pred_lasso)
r2_lasso = r2_score(y_test, y_pred_lasso)

print(f"\nLasso Regression - Mean Squared Error: {mse_lasso}")
print(f"Lasso Regression - R^2 Score: {r2_lasso}")
r2_sc['Lasso'] = r2_lasso

ridge_model = Ridge(alpha=0.01)  
ridge_model.fit(x_train, y_train)

y_pred_ridge = ridge_model.predict(x_test)
mse_ridge = mean_squared_error(y_test, y_pred_ridge)
r2_ridge = r2_score(y_test, y_pred_ridge)

print(f"\nRidge Regression - Mean Squared Error: {mse_ridge}")
print(f"Ridge Regression - R^2 Score: {r2_ridge}")
r2_sc['Ridge'] = r2_ridge

from sklearn.metrics import accuracy_score, r2_score
from sklearn.svm import SVC


model = SVC(kernel='linear')

model.fit(x_train, y_train)


y_pred = model.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Accuracy: {accuracy * 100}")
print(f"R^2 Score: {r2}")

r2_sc['SVM'] = r2

import numpy as np
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Step 4: Train the XGBoost Model
model = xgb.XGBRegressor(use_label_encoder=False, eval_metric='rmse')
model.fit(x_train, y_train)

# Step 5: Evaluate the Model
y_pred = model.predict(x_test)

# Calculate performance metrics
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

r2_sc['XGBoost'] = r2

from sklearn.ensemble import RandomForestRegressor

# Initialize and Train Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(x_train, y_train)

# Make Predictions
y_pred_rf = rf_model.predict(x_test)

# Evaluate the Random Forest Model
mse_rf = mean_squared_error(y_test, y_pred_rf)
r2_rf = r2_score(y_test, y_pred_rf)

print(f"Random Forest - Mean Squared Error: {mse_rf}")
print(f"Random Forest - R^2 Score: {r2_rf}")

r2_sc['Random Forest'] = r2_rf

from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

# Initialize and Train Bagging Model
bagging_model = BaggingRegressor(base_estimator=DecisionTreeRegressor(),
                                 n_estimators=100,
                                 random_state=42)
bagging_model.fit(x_train, y_train)

# Make Predictions
y_pred_bagging = bagging_model.predict(x_test)

# Evaluate the Bagging Model
mse_bagging = mean_squared_error(y_test, y_pred_bagging)
r2_bagging = r2_score(y_test, y_pred_bagging)

print(f"Bagging - Mean Squared Error: {mse_bagging}")
print(f"Bagging - R^2 Score: {r2_bagging}")

r2_sc['Bagging'] = r2_bagging
r2_sc
# Get the key with the maximum value
max_key = max(r2_sc, key=r2_sc.get)
print(f"Technique with maximum accuracy: {max_key}")


