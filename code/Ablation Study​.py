#!/usr/bin/env python
# coding: utf-8

# In[25]:


import pandas as pd
import xgboost as xgb
import numpy as np
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split, KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
import lightgbm as lgb


# In[2]:


from sklearn.decomposition import NMF
import numpy as np
df = pd.read_excel(r"")


# In[3]:


y = df['Temperature_peak(℃)'].values

# 只选择需要的两列作为特征
feature_columns = ["Catalysts_Mass_Fraction(wt%)", "heating_rate(℃/min)"]
X = df[feature_columns].values  # 直接提取为NumPy数组

# 划分训练集和测试集（一九分）
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

# 检查维度
print("训练集特征维度:", X_train.shape)
print("测试集特征维度:", X_test.shape)
print("训练集目标维度:", y_train.shape)
print("测试集目标维度:", y_test.shape)

# 特征名称（用于后续分析）
print("使用的特征:", feature_columns)


# In[17]:


xgbt = xgb.XGBRegressor(objective='reg:squarederror', learning_rate=0.1, gamma=0.90,
                        reg_alpha=0.9, reg_lambda=0.9, max_depth=3, 
                        min_child_weight=20, subsample=0.99, colsample_bytree=1,n_estimators=900,
                        random_state=42)
xgbt.fit(X_train, y_train)

# 训练集评估
train_pred = xgbt.predict(X_train)
print('Training R2 = %.3f' % r2_score(y_train, train_pred))
print('Training RMSE = %.3f' % np.sqrt(mean_squared_error(y_train, train_pred)))
print('Training MAE = %.3f' % mean_absolute_error(y_train, train_pred))

# 测试集评估
test_pred = xgbt.predict(X_test)
print('\nTesting R2 = %.3f' % r2_score(y_test, test_pred))
print('Testing RMSE = %.3f' % np.sqrt(mean_squared_error(y_test, test_pred)))
print('Testing MAE = %.3f' % mean_absolute_error(y_test, test_pred))

crossvalidation = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(xgbt, X_train, y_train, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=-1)
rmse_scores = [np.sqrt(abs(s)) for s in scores]
r2_scores = cross_val_score(xgbt, X_train, y_train, scoring='r2', cv=crossvalidation, n_jobs=-1)
mae_scores = cross_val_score(xgbt, X_train, y_train, scoring='neg_mean_absolute_error', cv=crossvalidation, n_jobs=-1)

print('\nCross-validation results:')
print('Folds: %i, mean R2: %.3f' % (len(r2_scores), np.mean(r2_scores)))
print('Folds: %i, mean RMSE: %.3f' % (len(rmse_scores), np.mean(rmse_scores)))
print('Folds: %i, mean MAE: %.3f' % (len(mae_scores), np.mean(np.abs(mae_scores))))


# In[23]:


rf = RandomForestRegressor(n_estimators=100, random_state=42,max_depth=27,min_samples_split=5,min_samples_leaf=1,criterion='absolute_error',
                           min_weight_fraction_leaf=0.0, max_features='log2', max_leaf_nodes=None, min_impurity_decrease=0, bootstrap=True, 
                           oob_score=False, n_jobs=-1,verbose=0, warm_start=False, ccp_alpha=0.0, max_samples=None)

rf.fit(X_train, y_train)

# 训练集评估
train_pred = rf.predict(X_train)
print('Training R2 = %.3f' % r2_score(y_train, train_pred))
print('Training RMSE = %.3f' % np.sqrt(mean_squared_error(y_train, train_pred)))
print('Training MAE = %.3f' % mean_absolute_error(y_train, train_pred))

# 测试集评估
test_pred = rf.predict(X_test)
print('\nTesting R2 = %.3f' % r2_score(y_test, test_pred))
print('Testing RMSE = %.3f' % np.sqrt(mean_squared_error(y_test, test_pred)))
print('Testing MAE = %.3f' % mean_absolute_error(y_test, test_pred))
# 交叉验证
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(rf, X_train, y_train, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=-1)
rmse_scores = [np.sqrt(abs(s)) for s in scores]
r2_scores = cross_val_score(rf, X_train, y_train, scoring='r2', cv=crossvalidation, n_jobs=-1)
mae_scores = cross_val_score(rf, X_train, y_train, scoring='neg_mean_absolute_error', cv=crossvalidation, n_jobs=-1)

print('\nCross-validation results:')
print('Folds: %i, mean R2: %.3f' % (len(r2_scores), np.mean(r2_scores)))
print('Folds: %i, mean RMSE: %.3f' % (len(rmse_scores), np.mean(rmse_scores)))
print('Folds: %i, mean MAE: %.3f' % (len(mae_scores), np.mean(np.abs(mae_scores))))


# In[26]:


lgb_reg = lgb.LGBMRegressor(
    boosting_type='gbdt',
    num_leaves=25,             
    max_depth=20,                # 限制最大深度，防止过拟合
    learning_rate=0.05,         # 更小的学习率，配合早停
    n_estimators=1000,          # 增加迭代次数，但配合早停
    objective='regression',
    min_child_samples=40,       # 防止叶子节点过拟合
    min_child_weight=11,
    subsample=1,              # 引入更多样本随机性
    colsample_bytree=1,       # 引入特征随机性
    reg_alpha=1,              # L1 正则增强
    reg_lambda=1.0,             # L2 正则增强
    subsample_freq=1,
    random_state=42,
    n_jobs=-1
)

# 训练模型
lgb_reg.fit(X_train, y_train)

# 训练集评估
train_pred = lgb_reg.predict(X_train)
print('Training R2 = %.3f' % r2_score(y_train, train_pred))
print('Training RMSE = %.3f' % np.sqrt(mean_squared_error(y_train, train_pred)))
print('Training MAE = %.3f' % mean_absolute_error(y_train, train_pred))

# 测试集评估
test_pred = lgb_reg.predict(X_test)
print('\nTesting R2 = %.3f' % r2_score(y_test, test_pred))
print('Testing RMSE = %.3f' % np.sqrt(mean_squared_error(y_test, test_pred)))
print('Testing MAE = %.3f' % mean_absolute_error(y_test, test_pred))

# 交叉验证
crossvalidation = KFold(n_splits=10, shuffle=True, random_state=42)
scores = cross_val_score(lgb_reg, X_train, y_train, scoring='neg_mean_squared_error', cv=crossvalidation, n_jobs=-1)
rmse_scores = [np.sqrt(abs(s)) for s in scores]
r2_scores = cross_val_score(lgb_reg, X_train, y_train, scoring='r2', cv=crossvalidation, n_jobs=-1)
mae_scores = cross_val_score(lgb_reg, X_train, y_train, scoring='neg_mean_absolute_error', cv=crossvalidation, n_jobs=-1)

print('\nCross-validation results:')
print('Folds: %i, mean R2: %.3f' % (len(r2_scores), np.mean(r2_scores)))
print('Folds: %i, mean RMSE: %.3f' % (len(rmse_scores), np.mean(rmse_scores)))
print('Folds: %i, mean MAE: %.3f' % (len(mae_scores), np.mean(np.abs(mae_scores))))


# In[ ]:




