#!/usr/bin/env python
# coding: utf-8

# In[103]:


import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
import matplotlib.pyplot as plt


# In[104]:


df = pd.read_csv("Telco-Customer-Churn.csv")


# In[105]:


df.head()


# In[106]:


df.columns = df.columns.str.lower().str.replace(' ', '_')

categorical_columns = list(df.dtypes[df.dtypes == 'str'].index)

for c in categorical_columns:
    df[c] = df[c].str.lower().str.replace(' ', '_')


# In[107]:


df.head().T


# In[108]:


df.totalcharges = pd.to_numeric(df.totalcharges, errors='coerce')


# In[109]:


df.totalcharges = df.totalcharges.fillna(df.totalcharges.median())


# In[110]:


df.churn.head()


# In[111]:


df.churn = (df.churn == 'yes').astype(int)


# In[112]:


from sklearn.model_selection import train_test_split


# In[113]:


df_full_train, df_test = train_test_split(df, test_size = 0.2, random_state = 1)


# In[114]:


df_train , df_val = train_test_split(df_full_train, test_size = 0.25, random_state = 1)


# In[115]:


len(df_test), len(df_val), len(df_train)


# In[116]:


df_train = df_train.reset_index(drop = True)
df_val = df_val.reset_index(drop = True)
df_test = df_test.reset_index(drop = True)


# In[117]:


y_train = df_train.churn.values
y_val = df_val.churn.values
y_test = df_test.churn.values


# In[118]:


del df_train['churn']
del df_val['churn']
del df_test['churn']


# In[119]:


df_full_train = df_full_train.reset_index(drop = True)


# In[120]:


df_full_train.churn.value_counts(normalize = True)


# In[121]:


global_churn_rate = df_full_train.churn.mean()
print(round(global_churn_rate, 2))


# In[122]:


numerical = ['tenure', 'monthlycharges', 'totalcharges']


# In[123]:


df_full_train.columns


# In[124]:


categorical = ['gender', 'seniorcitizen', 'partner', 'dependents', 'phoneservice', 'multiplelines', 'internetservice',
       'onlinesecurity', 'onlinebackup', 'deviceprotection', 'techsupport',
       'streamingtv', 'streamingmovies', 'contract', 'paperlessbilling',
       'paymentmethod']


# In[125]:


df_full_train[categorical].nunique()


# In[126]:


churn_female = df_full_train[df_full_train.gender == 'female'].churn.mean()
print(churn_female)


# In[127]:


churn_male = df_full_train[df_full_train.gender == 'male'].churn.mean()
print(churn_male)


# In[128]:


churn_partner = df_full_train[df_full_train.partner == 'yes'].churn.mean()
print(churn_partner)


# In[129]:


churn_no_partner = df_full_train[df_full_train.partner == 'no'].churn.mean()
print(churn_no_partner)


# In[130]:


churn_senior_citizen = df_full_train[df_full_train.seniorcitizen == 1].churn.mean()
print(churn_senior_citizen)


# In[131]:


print(global_churn_rate - churn_senior_citizen)


# In[132]:


churn_not_senior_citizen = df_full_train[df_full_train.seniorcitizen == 0].churn.mean()
print(churn_not_senior_citizen)


# In[133]:


print(churn_partner / global_churn_rate)


# In[134]:


print(churn_no_partner / global_churn_rate)


# In[135]:


print(churn_senior_citizen / global_churn_rate)


# In[136]:


print(churn_not_senior_citizen / global_churn_rate)


# In[137]:


from IPython.display import display


# In[138]:


for c in categorical:
    df_group = df_full_train.groupby(c).churn.agg(['mean', 'count'])
    df_group['diff'] = df_group['mean'] - global_churn_rate
    df_group['risk'] = df_group['mean'] / global_churn_rate
    display(df_group)
    print()
    print()


# In[139]:


from sklearn.metrics import mutual_info_score


# In[140]:


mutual_info_score(df_full_train.churn, df_full_train.partner)


# In[141]:


mutual_info_score(df_full_train.churn, df_full_train.gender)


# In[142]:


mutual_info_score(df_full_train.churn, df_full_train.contract)


# In[143]:


def mutual_info_score_churn(series):
    return mutual_info_score(df_full_train.churn, series)


# In[144]:


mi = df_full_train[categorical].apply(mutual_info_score_churn)
mi.sort_values(ascending = False)


# In[145]:


df_full_train[numerical].corrwith(df_full_train.churn)


# In[146]:


print(df_full_train[df_full_train.tenure <= 2].churn.mean())


# In[147]:


print(df_full_train[(df_full_train.tenure > 2) & (df_full_train.tenure <= 12)].churn.mean())


# In[148]:


print(df_full_train[df_full_train.tenure > 12].churn.mean())


# In[149]:


print(df_full_train[df_full_train.monthlycharges <= 20].churn.mean())


# In[150]:


print(df_full_train[(df_full_train.monthlycharges > 20) & (df_full_train.monthlycharges <= 50)].churn.mean())


# In[151]:


print(df_full_train[df_full_train.monthlycharges > 50].churn.mean())


# In[152]:


from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder


# In[153]:


X_train_num = df_train[numerical].values
scaler = StandardScaler()

X_train_num = scaler.fit_transform(X_train_num)


# In[154]:


X_train_num[:, 0]


# In[155]:


ohe = OneHotEncoder(sparse_output = False, handle_unknown = 'ignore')


# In[156]:


X_train_cat = ohe.fit_transform(df_train[categorical].values)


# In[157]:


print(ohe.get_feature_names_out())


# In[158]:


X_train = np.column_stack([X_train_num, X_train_cat])


# In[159]:


from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score


# In[160]:


model = LogisticRegression(solver='lbfgs', C=1.0, random_state=42)
model.fit(X_train, y_train)


# In[161]:


print(model.intercept_[0])


# In[162]:


model.coef_[0].round(3)


# In[163]:


model.predict_proba(X_train)


# In[164]:


X_val_num = df_val[numerical].values
X_val_num = scaler.transform(X_val_num)

X_val_cat = ohe.transform(df_val[categorical].values)

X_val = np.column_stack([X_val_num, X_val_cat])


# In[198]:


y_pred = model.predict_proba(X_val)[:,1]
roc_auc_score(y_val, y_pred)


# In[166]:


churn_decision = (y_pred >= 0.5)


# In[167]:


df_val[churn_decision].customerid


# In[168]:


X_full_train_num = df_full_train[numerical].values
scaler = StandardScaler()

X_full_train_num = scaler.fit_transform(X_full_train_num)


# In[169]:


X_full_train_num[:, 0]


# In[170]:


X_full_train_cat = ohe.fit_transform(df_full_train[categorical].values)


# In[171]:


print(ohe.get_feature_names_out())


# In[172]:


X_full_train = np.column_stack([X_full_train_num, X_full_train_cat])


# In[173]:


y_full_train = df_full_train.churn.values


# In[174]:


model = LogisticRegression(solver='lbfgs', C=1.0, random_state=42) 
model.fit(X_full_train, y_full_train)


# In[175]:


X_test_num = df_test[numerical].values
X_test_num = scaler.transform(X_test_num)

X_test_cat = ohe.transform(df_test[categorical].values)

X_test = np.column_stack([X_test_num, X_test_cat])


# In[176]:


y_pred = model.predict_proba(X_test)[:, 1]
roc_auc_score(y_test, y_pred)


# In[177]:


y_train_pred = model.predict(X_full_train)

print(
    "Training accuracy:",
    accuracy_score(y_full_train, y_train_pred)
)


# In[178]:


y_pred = model.predict_proba(X_val)[:, 1]
churn_decision = (y_pred >= 0.5)
print((y_val == churn_decision).mean())


# In[179]:


thresholds = np.linspace(0, 1, 21)
scores = []

for t in thresholds:
    score = accuracy_score(y_val, y_pred >= t)
    print('%.2f %.3f' % (t, score))
    scores.append(score)


# In[180]:


plt.plot(thresholds, scores)


# In[181]:


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

y_pred = model.predict_proba(X_val)[:, 1]
y_pred_class = (y_pred >= 0.5).astype(int)

cm = confusion_matrix(y_val, y_pred_class)

disp = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=['No Churn', 'Churn']
)

disp.plot(cmap='Blues', values_format='d')
plt.title('Confusion Matrix - Churn Prediction')
plt.show()


# In[182]:


from sklearn.metrics import classification_report

print(classification_report(
    y_val,
    y_pred_class,
    target_names=['No Churn', 'Churn'],
    zero_division=0
))


# In[183]:


import pickle


# In[184]:


C = 1.0
output_file = f'model_C={C}.bin'
output_file


# In[185]:


with open(output_file, 'wb') as f_out:
    pickle.dump(model, f_out)


# In[186]:


import pickle


# In[187]:


model_file = 'model_C=1.0.bin'


# In[188]:


with open(model_file, 'rb') as f_in:
    model = pickle.load(f_in)


# In[191]:


y_pred = model.predict_proba(X_val)


# In[193]:


y_pred[:,1]

