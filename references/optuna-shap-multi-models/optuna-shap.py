# pip install dash
# pip install xgboost
# pip install catboost
# pip install lightgbm

import time
# Computational Libs
import numpy as np
import pandas as pd
# Viz Libs
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import holoviews as hv
from holoviews import opts
# Helper and Utility Libs for Pre-processing
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
# Model Libs
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
# Hyper-parameter Libs
import optuna
from sklearn.model_selection import cross_val_score

try:
    df = pd.read_csv(r'2025-4-26-公众号python机器学习ML.csv')
    #df = pd.read_csv('2025-4-26-公众号python机器学习ML.csv')

except:
    print('Error loading the data file')

# Clean the 'Amount' column
df['Amount'] = df['Amount'].str.replace(r'[$,]', '', regex=True).astype(float)
# Convert 'Date' column to datetime objects
df['Date'] = pd.to_datetime(df['Date'], format='%d-%b-%y')
# Create new features from the 'Date' column
df['Month'] = df['Date'].dt.month
df['Year'] = df['Date'].dt.year

# Display the shape
print("\nShape of the DataFrame:", df.shape)
# Display data type information
print("\nData types of each column:\n")
df.info()

df

# Generate descriptive statistics
print("\nDescriptive statistics:\n", df.describe(include='all'))

# Analyze the distribution of key variables
print("\nDistribution of 'Amount' and 'Boxes Shipped':")
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.hist(df['Amount'], bins=20, edgecolor = "black")  # Adjust the number of bins as needed
plt.xlabel('Amount')
plt.ylabel('Frequency')
plt.title('Distribution of Amount')

plt.subplot(1, 2, 2)
plt.hist(df['Boxes Shipped'], bins=20, edgecolor = "black")  # Adjust the number of bins as needed
plt.xlabel('Boxes Shipped')
plt.ylabel('Frequency')
plt.title('Distribution of Boxes Shipped')
plt.tight_layout()
plt.show()

# Calculate and visualize the correlation matrix (with numeric_only=True)
correlation_matrix = df.corr(numeric_only=True)
print("\nCorrelation Matrix:\n", correlation_matrix)
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix of Numerical Features')
plt.show()

# Check for missing values
print(df.isnull().sum())

# Handle outliers using winsorization
def winsorize_outliers(series, limits=(0.05, 0.95)):
    lower_limit = series.quantile(limits[0])
    upper_limit = series.quantile(limits[1])
    winsorized_series = series.clip(lower_limit, upper_limit)
    return winsorized_series

for col in ['Amount', 'Boxes Shipped']:
    df[col] = winsorize_outliers(df[col])
    print(f"Winsorized outliers in '{col}'.")

print(df.describe())

# Calculate the 'Profit' column (assuming a 20% profit margin)
df['Profit'] = df['Amount'] * 0.20

# Display the updated DataFrame
print(df.head())

# Sales Distribution
sales_stats = df['Amount'].describe()
print(sales_stats)

# Identify best and worst-selling products
product_sales = df.groupby('Product')['Amount'].sum()
print("\nProduct Sales:\n", product_sales)
best_selling_product = product_sales.idxmax()
worst_selling_product = product_sales.idxmin()
print(f"\nBest Selling Product: {best_selling_product}")
print(f"Worst Selling Product: {worst_selling_product}")

# Analyze sales trends over time
yearly_sales = df.groupby('Year')['Amount'].sum()
print("\nYearly Sales:\n", yearly_sales)
monthly_sales = df.groupby('Month')['Amount'].sum()
print("\nMonthly Sales:\n", monthly_sales)

# Relationship between sales amount and boxes shipped
correlation = df['Amount'].corr(df['Boxes Shipped'])
print(f"\nCorrelation between Sales Amount and Boxes Shipped: {correlation}")

# Sales contribution of each country
country_sales = df.groupby('Country')['Amount'].sum()
print("\nCountry Sales:\n", country_sales)

# Salesperson performance
salesperson_sales = df.groupby('Sales Person')['Amount'].sum()
print("\nSalesperson Performance:\n", salesperson_sales)

# Combined category analysis - Example: Product performance in each country
product_country_sales = df.groupby(['Product', 'Country'])['Amount'].sum().unstack()
print("\nProduct Performance by Country:\n", product_country_sales)

plt.figure(figsize=(12, 5))

# Histogram
plt.subplot(1, 2, 1)
plt.hist(df['Amount'], bins=20, color='skyblue', edgecolor='black')
plt.xlabel('Sales Amount')
plt.ylabel('Frequency')
plt.title('Distribution of Sales Amount')

# Box plot
plt.subplot(1, 2, 2)
sns.boxplot(y=df['Amount'], color='lightcoral')
plt.ylabel('Sales Amount')
plt.title('Box Plot of Sales Amount')

plt.tight_layout()
plt.show()

# Identify categorical columns
categorical_cols = ['Sales Person', 'Country', 'Product']

# Apply one-hot encoding to categorical features
df_copy = pd.get_dummies(df, columns=categorical_cols, drop_first=True)

# Display the updated DataFrame
print(df.head())

product_columns = [col for col in df_copy.columns if 'Product_' in col]
product_sales = df_copy[product_columns].sum() * df_copy['Amount'].mean() / len(product_columns)

plt.figure(figsize=(15, 6))
ax = product_sales.plot(kind='bar', color='skyblue')
ax.set_xticklabels([col_name.split('_')[1] for col_name in product_columns], rotation=45)
plt.xlabel('Product')
plt.ylabel('Total Sales')
plt.title('Total Sales per Product')
plt.xticks(ha='right')
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 4))

# Monthly Sales Trend
plt.subplot(1, 1, 1)
plt.plot(df.groupby('Month')['Amount'].sum(), marker='o', linestyle='-', color='green')
plt.xlabel('Month')
plt.ylabel('Total Sales')
plt.title('Monthly Sales Trend')

plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
plt.scatter(df['Boxes Shipped'], df['Amount'], color='purple', alpha=0.6)
plt.xlabel('Boxes Shipped')
plt.ylabel('Sales Amount')
plt.title('Sales Amount vs. Boxes Shipped')

# Calculate and plot the trendline
z = np.polyfit(df['Boxes Shipped'], df['Amount'], 1)
p = np.poly1d(z)
plt.plot(df['Boxes Shipped'], p(df['Boxes Shipped']), color='red', linestyle='--', label='Trendline')
plt.legend()

plt.show()

country_columns = [col for col in df_copy.columns if 'Country_' in col]
country_sales = df_copy[country_columns].sum() * df_copy['Amount'].mean() / len(country_columns)

plt.figure(figsize=(10, 6))
ax = country_sales.plot(kind='bar', color='orange')
ax.set_xticklabels([col_name.split('_')[1] for col_name in country_columns], rotation=45)
plt.xlabel('Country')
plt.ylabel('Total Sales')
plt.title('Sales Contribution by Country')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

salesperson_columns = [col for col in df_copy.columns if 'Sales Person_' in col]
salesperson_performance = df_copy[salesperson_columns].sum() * df_copy['Amount'].mean() / len(salesperson_columns)

plt.figure(figsize=(12, 6))
ax = salesperson_performance.plot(kind='bar', color='green')
ax.set_xticklabels([col_name.split('_')[1] for col_name in salesperson_columns], rotation=45)
plt.xlabel('Salesperson')
plt.ylabel('Total Sales')
plt.title('Salesperson Performance')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

hv.extension('bokeh')

unique_labels = list(set(df['Country']).union(set(df['Sales Person'])))
label_map = {label: i for i, label in enumerate(unique_labels)}

links = [(label_map[row['Country']], label_map[row['Sales Person']], row['Amount']) for _, row in df.iterrows()]

nodes = hv.Dataset(pd.DataFrame({'index': list(label_map.values()), 'label': list(label_map.keys())}), 'index')

chord = hv.Chord((links, nodes)).opts(
    opts.Chord(labels='label', cmap='Category20', edge_cmap='viridis', edge_color='Amount', node_color='index', node_size=20, width=800, height=800)
)

chord

# Identify categorical columns
categorical_cols = ['Sales Person', 'Country', 'Product', 'Month']

# Apply one-hot encoding to categorical features
df = pd.get_dummies(df, columns=categorical_cols, drop_first=True, dtype=float)
df

# Define features (X) and target (y)
X = df.drop(['Amount', 'Year', 'Date', 'Profit'], axis=1)
y = df['Amount']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Print the shapes of the resulting datasets to verify the split
print("X_train shape:", X_train.shape)
print("X_test shape:", X_test.shape)
print("y_train shape:", y_train.shape)
print("y_test shape:", y_test.shape)

# Initialize models
dt_model = DecisionTreeRegressor(random_state=42)
rf_model = RandomForestRegressor(random_state=42)
xgb_model = XGBRegressor(random_state=42)
lgb_model = LGBMRegressor(random_state=42)
catb_model = CatBoostRegressor(random_state=42, verbose=0)

# Train Decision Tree model
start_time = time.time()
dt_model.fit(X_train, y_train)
end_time = time.time()
print(f"Decision Tree training time: {end_time - start_time:.2f} seconds")

# Train Random Forest model
start_time = time.time()
rf_model.fit(X_train, y_train)
end_time = time.time()
print(f"Random Forest training time: {end_time - start_time:.2f} seconds")

# Train XGBoost model
start_time = time.time()
xgb_model.fit(X_train, y_train)
end_time = time.time()
print(f"XGBoost training time: {end_time - start_time:.2f} seconds")

# Train LightGB model
start_time = time.time()
lgb_model.fit(X_train, y_train)
end_time = time.time()
print(f"LightGB training time: {end_time - start_time:.2f} seconds")

# Train CatBoost model
start_time = time.time()
catb_model.fit(X_train, y_train)
end_time = time.time()
print(f"CatBoost training time: {end_time - start_time:.2f} seconds")

print("Models trained successfully.")

# Make predictions
dt_predictions = dt_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)
xgb_predictions = xgb_model.predict(X_test)
catb_predictions = catb_model.predict(X_test)
lgbm_predictions = lgb_model.predict(X_test) # 添加 LightGBM 的预测

dt_mse = mean_squared_error(y_test, dt_predictions)
dt_mae = mean_absolute_error(y_test, dt_predictions)
dt_r2 = r2_score(y_test, dt_predictions)
print(f"Decision Tree MSE: {dt_mse}, MAE: {dt_mae}, R2: {dt_r2}")

rf_mse = mean_squared_error(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)
print(f"Random Forest MSE: {rf_mse}, MAE: {rf_mae}, R2: {rf_r2}")

xgb_mse = mean_squared_error(y_test, xgb_predictions)
xgb_mae = mean_absolute_error(y_test, xgb_predictions)
xgb_r2 = r2_score(y_test, xgb_predictions)
print(f"XGBoost MSE: {xgb_mse}, MAE: {xgb_mae}, R2: {xgb_r2}")

lgbm_mse = mean_squared_error(y_test, lgbm_predictions) # 正确：使用 LightGBM 的预测结果
lgbm_mae = mean_absolute_error(y_test, lgbm_predictions) # 正确：使用 LightGBM 的预测结果
lgbm_r2 = r2_score(y_test, lgbm_predictions) # 正确：使用 LightGBM 的预测结果
print(f"LightGBM MSE: {lgbm_mse}, MAE: {lgbm_mae}, R2: {lgbm_r2}")

catb_mse = mean_squared_error(y_test, catb_predictions)
catb_mae = mean_absolute_error(y_test, catb_predictions)
catb_r2 = r2_score(y_test, catb_predictions)
print(f"CatBoost MSE: {catb_mse}, MAE: {catb_mae}, R2: {catb_r2}")

model_results = {
    'Decision Tree': {'MSE': dt_mse, 'MAE': dt_mae, 'R2': dt_r2},
    'Random Forest': {'MSE': rf_mse, 'MAE': rf_mae, 'R2': rf_r2},
    'XG Boost': {'MSE': xgb_mse, 'MAE': xgb_mae, 'R2': xgb_r2},
    'LGBM Boost': {'MSE': lgbm_mse, 'MAE': lgbm_mae, 'R2': lgbm_r2},
    'Cat Boost': {'MSE': catb_mse, 'MAE': catb_mae, 'R2': catb_r2},
}

print(model_results)

model_names = list(model_results.keys())
mse_values = [model_results[model]['MSE'] for model in model_names]

plt.figure(figsize=(8, 6))
plt.bar(model_names, mse_values, color=['skyblue', 'lightgreen', 'lightcoral', 'yellow', 'mediumpurple'])
plt.xlabel("Model")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Model Performance Comparison (MSE)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

print("The bar chart visualizes the Mean Squared Error (MSE) for each model. Lower MSE indicates better performance.")

# Define an objective function for each model
def objective(trial, model_name):
    global X_train, y_train  # Ensure X_train and y_train are accessible

    if model_name == "DecisionTree":
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        }
        model = DecisionTreeRegressor(**params, random_state=42)

    elif model_name == "RandomForest":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        }
        model = RandomForestRegressor(**params, random_state=42)

    elif model_name == "XGBoost":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        }
        model = XGBRegressor(**params, random_state=42)

    elif model_name == "LightGBM":
        params = {
            "n_estimators": trial.suggest_int("n_estimators", 50, 300),
            "max_depth": trial.suggest_int("max_depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
            "num_leaves": trial.suggest_int("num_leaves", 10, 200),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        }
        model = LGBMRegressor(**params, random_state=42, verbose=-1)

    elif model_name == "CatBoost":
        params = {
            "iterations": trial.suggest_int("iterations", 50, 300),
            "depth": trial.suggest_int("depth", 2, 10),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3),
        }
        model = CatBoostRegressor(**params, random_state=42, verbose=0)

    # Evaluate using cross-validation (5 folds)
    score = cross_val_score(model, X_train, y_train, cv=5, scoring="neg_mean_squared_error").mean()
    return -score  # Optuna minimizes the objective function

# Run Optuna optimization
def tune_model(model_name, n_trials=20):
    print(f"Tuning {model_name}...")
    study = optuna.create_study(direction="minimize")
    study.optimize(lambda trial: objective(trial, model_name), n_trials=n_trials)
    print(f"Best parameters for {model_name}: {study.best_params}\n")
    return study.best_params

# Tune each model
dt_params = tune_model("DecisionTree")
rf_params = tune_model("RandomForest")
xgb_params = tune_model("XGBoost")
lgb_params = tune_model("LightGBM")
cat_params = tune_model("CatBoost")

# Train models with best parameters
dt_model = DecisionTreeRegressor(**dt_params, random_state=42)
rf_model = RandomForestRegressor(**rf_params, random_state=42)
xgb_model = XGBRegressor(**xgb_params, random_state=42)
lgb_model = LGBMRegressor(**lgb_params, random_state=42)
cat_model = CatBoostRegressor(**cat_params, random_state=42)

# Train Decision Tree model
start_time = time.time()
dt_model.fit(X_train, y_train)
end_time = time.time()
print(f"Decision Tree training time: {end_time - start_time:.2f} seconds")

# Train Random Forest model
start_time = time.time()
rf_model.fit(X_train, y_train)
end_time = time.time()
print(f"Random Forest training time: {end_time - start_time:.2f} seconds")

# Train XGBoost model
start_time = time.time()
xgb_model.fit(X_train, y_train)
end_time = time.time()
print(f"XGBoost training time: {end_time - start_time:.2f} seconds")

# Train LightGBM model
start_time = time.time()
lgb_model.fit(X_train, y_train)
end_time = time.time()
print(f"LightGBM training time: {end_time - start_time:.2f} seconds")

# Train CatBoost model
start_time = time.time()
cat_model.fit(X_train, y_train)
end_time = time.time()
print(f"CatBoost training time: {end_time - start_time:.2f} seconds")

print("Models trained successfully with optimized hyperparameters.")

# Make predictions
dt_predictions = dt_model.predict(X_test)
rf_predictions = rf_model.predict(X_test)
xgb_predictions = xgb_model.predict(X_test)
# **修正添加：** 需要用优化后的 LightGBM 模型进行预测
lgbm_predictions = lgb_model.predict(X_test) # 添加优化后 LightGBM 的预测
# 使用优化后的 CatBoost 模型进行预测 (使用 cat_model 变量)
# **修正：** 原代码下面评估 CatBoost 时仍然使用了旧的 catb_predictions，这里应使用 cat_model
catb_predictions = cat_model.predict(X_test) # 使用优化后的 CatBoost 模型预测
# Evaluate models
dt_mse = mean_squared_error(y_test, dt_predictions)
dt_mae = mean_absolute_error(y_test, dt_predictions)
dt_r2 = r2_score(y_test, dt_predictions)
print(f"Decision Tree MSE: {dt_mse}, MAE: {dt_mae}, R2: {dt_r2}")

rf_mse = mean_squared_error(y_test, rf_predictions)
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)
print(f"Random Forest MSE: {rf_mse}, MAE: {rf_mae}, R2: {rf_r2}")

xgb_mse = mean_squared_error(y_test, xgb_predictions)
xgb_mae = mean_absolute_error(y_test, xgb_predictions)
xgb_r2 = r2_score(y_test, xgb_predictions)
print(f"XGBoost MSE: {xgb_mse}, MAE: {xgb_mae}, R2: {xgb_r2}")

lgbm_mse = mean_squared_error(y_test, lgbm_predictions)
lgbm_mae = mean_absolute_error(y_test, lgbm_predictions)
lgbm_r2 = r2_score(y_test, lgbm_predictions)
print(f"LightGBM MSE: {lgbm_mse}, MAE: {lgbm_mae}, R2: {lgbm_r2}")

catb_mse = mean_squared_error(y_test, catb_predictions)
catb_mae = mean_absolute_error(y_test, catb_predictions)
catb_r2 = r2_score(y_test, catb_predictions)
print(f"CatBoost MSE: {catb_mse}, MAE: {catb_mae}, R2: {catb_r2}")

opti_model_results = {
    'Decision Tree': {'MSE': dt_mse, 'MAE': dt_mae, 'R2': dt_r2},
    'Random Forest': {'MSE': rf_mse, 'MAE': rf_mae, 'R2': rf_r2},
    'XG Boost': {'MSE': xgb_mse, 'MAE': xgb_mae, 'R2': xgb_r2},
    'LGBM Boost': {'MSE': lgbm_mse, 'MAE': lgbm_mae, 'R2': lgbm_r2},
    'Cat Boost': {'MSE': catb_mse, 'MAE': catb_mae, 'R2': catb_r2},
}

print(opti_model_results)

model_names = list(model_results.keys())
mse_values = [model_results[model]['MSE'] for model in model_names]
opti_mse_values = [opti_model_results[model]['MSE'] for model in model_names]

plt.figure(figsize=(8, 6))
plt.bar(model_names, mse_values, color=['skyblue', 'lightgreen', 'lightcoral', 'yellow', 'mediumpurple'])
plt.bar(model_names, opti_mse_values, color=['cyan', 'green', 'red', 'orange', 'indigo'])
plt.xlabel("Model")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Model Performance Comparison (MSE)")
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
bar_width = 0.35 # 定义条形宽度
index = np.arange(len(model_names)) # 创建模型名称对应的数字索引

plt.bar(index - bar_width/2, mse_values, bar_width, label='Initial MSE', color='skyblue')
plt.bar(index + bar_width/2, opti_mse_values, bar_width, label='Optimized MSE', color='lightgreen')

plt.xticks(index, model_names, rotation=45, ha="right") # 设置刻度在组的中间
plt.xlabel("Model")
plt.ylabel("Mean Squared Error (MSE)")
plt.title("Model Performance Comparison (MSE)")
plt.tight_layout()
plt.show()

#shap

# 导入SHAP和其他必要的库
import shap
import matplotlib.pyplot as plt
import numpy as np

# 为XGBoost模型创建SHAP解释器
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test)


# . 摘要图 - 显示特征重要性和影响方向
shap.summary_plot(shap_values, X_test, plot_type="bar")
plt.tight_layout()
plt.savefig("shap_feature_importance.png", dpi=300, bbox_inches='tight')
plt.show()

# . 摘要点图 - 显示每个特征的SHAP值分布
shap.summary_plot(shap_values, X_test, plot_type="dot")
plt.tight_layout()
plt.savefig("shap_feature_importance.png", dpi=300, bbox_inches='tight')
plt.show()

print(X_test.columns)

# . 决策图 - 显示样本的预测路径

shap.decision_plot(explainer.expected_value, shap_values.values[:50],
                   feature_names=np.array(X_test.columns),
                   show=False)
plt.tight_layout()
plt.savefig("shap_decision_plot.png", dpi=300, bbox_inches='tight')
plt.show()

# . 瀑布图 - 详细分析多个预测样本
sample_indices = [0, 5, 10]  # 选择多个样本进行对比分析
for i, idx in enumerate(sample_indices):
    plt.figure(figsize=(12, 8))

    # 修改此处，确保负号正常显示
    waterfall = shap.plots.waterfall(shap.Explanation(values=shap_values.values[i],
                                                      base_values=explainer.expected_value,
                                                      data=X_test.iloc[idx],
                                                      feature_names=np.array(X_test.columns)),
                                     show=False)  # 使用新的API并设置show=False
    plt.tight_layout()
    plt.savefig(f"shap_waterfall_sample_{idx}.png", dpi=300, bbox_inches='tight')
    plt.show()

#  力图 - 单个样本预测的特征贡献
# 为多个不同样本创建力图以比较分析
sample_indices = [7, 15, 25]  # 选择多个不同特征的样本
print("力图解释：")
print("- 力图直观显示每个特征如何推动预测值远离或接近基准值")
print("- 红色表示增加风险的特征值，蓝色表示降低风险的特征值")
for sample_idx in sample_indices:
    plt.figure(figsize=(14, 3))
    # 创建样本数据的副本并将其舍入到2位小数
    sample_data = X_test.iloc[sample_idx].copy()  # 使用.iloc进行基于位置的索引
    for i in range(len(sample_data)):
        sample_data[i] = round(sample_data[i], 2)
    # 使用处理后的数据绘制force plot
    shap.force_plot(explainer.expected_value,
                    shap_values.values[sample_idx],
                    sample_data,
                    matplotlib=True,
                    feature_names=np.array(X_test.columns),
                    show=False)
    plt.title(f" #{sample_idx} force_plot", fontsize=15)
    plt.tight_layout()
    plt.savefig(f"shap_force_plot_sample_{sample_idx}.png", dpi=300, bbox_inches='tight')
    plt.show()
    # 计算模型预测概率
    pred_prob = 1 / (1 + np.exp(-explainer.expected_value - np.sum(shap_values.values[sample_idx])))
    print(f"样本 #{sample_idx} 分析:")
    print(f"- 模型预测概率: {pred_prob:.2%}")
    feature_names=np.array(X_test.columns)
    print(f"- 主要风险因素: {[feature_names[i] for i in np.argsort(-shap_values.values[sample_idx])[:3]]}")
    print(f"- 主要保护因素: {[feature_names[i] for i in np.argsort(shap_values.values[sample_idx])[:3]]}")
    print()

#  SHAP值的热图
explainer = shap.Explainer(xgb_model)
shap_values = explainer(X_test)

# 绘制 SHAP 热图
shap.plots.heatmap(shap_values)

# 保存或显示图形
plt.savefig("shap_heatmap.png")
plt.show()









