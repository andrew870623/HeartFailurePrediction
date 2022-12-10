#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
data = pd.read_csv("heart_failure_clinical_records_dataset.csv", header= 0, encoding='gbk')


# In[2]:


data.head()


# In[3]:


data.isnull().sum()


# In[4]:


data.describe().T


# In[5]:


#數據整理
import numpy as np
import pandas as pd

#視覺化工具
import matplotlib.pyplot as plt
import seaborn as sns
import plotly as py 
import plotly.graph_objs as go
import plotly.express as px
import plotly.figure_factory as ff

#建立模型
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier , RandomForestClassifier
import lightgbm

from sklearn.ensemble import ExtraTreesClassifier

#前處理
from sklearn.preprocessing import StandardScaler

#模型評估
from sklearn.model_selection import train_test_split , GridSearchCV
from sklearn.metrics import plot_confusion_matrix , confusion_matrix , f1_score, accuracy_score, precision_score,recall_score,classification_report

import tensorflow as tf


# In[6]:


death_num = data['DEATH_EVENT'].value_counts()                  
death_num = death_num.reset_index()

#pie plot
fig = px.pie(death_num , names = 'index', values = 'DEATH_EVENT')                 
fig.update_layout( title_text = '目標變量DEATH_EVENT的分佈')                   
# py.offline.plot( fig , filename = '目標變量DEATH_EVENT的分佈')
fig.show()


# In[7]:


# 挑選要訓練的特徵 Feature Selection

plt.rcParams['figure.figsize']=15,6 
sns.set_style("darkgrid")

x = data.iloc[:, :-1]
y = data.iloc[:,-1]

model = ExtraTreesClassifier()
model.fit(x,y)
print(model.feature_importances_) 
feat_importances = pd.Series(model.feature_importances_, index=x.columns)
feat_importances.nlargest(12).plot(kind='barh')
plt.show()


# In[8]:


import plotly.express as px
fig = px.histogram(data, x="serum_creatinine", color="DEATH_EVENT", marginal="violin", hover_data=data.columns,
                   title ="Distribution of serum_creatinine to DEATH_EVENT", 
                   labels={"serum_creatinine": "serum_creatinine"},
                   color_discrete_map={"0": "purple", "1": "green"})
fig.show()


# In[9]:


import plotly.express as px
fig = px.histogram(data, x="time", color="DEATH_EVENT", marginal="violin", hover_data=data.columns,
                   title ="Distribution of time to DEATH_EVENT", 
                   labels={"time": "time"},
                   color_discrete_map={"0": "purple", "1": "green"})
fig.show()


# In[10]:


import plotly.express as px
fig = px.histogram(data, x="ejection_fraction", color="DEATH_EVENT", marginal="violin", hover_data=data.columns,
                   title ="Distribution of ejection_fraction to DEATH_EVENT", 
                   labels={"ejection_fraction": "ejection_fraction"},
                   color_discrete_map={"0": "purple", "1": "green"})
fig.show()


# In[11]:


# serum_creatinine
fig = px.box(data, x="DEATH_EVENT", y="serum_creatinine", points="all", color="DEATH_EVENT")
fig.show()


# In[12]:


# 取四分位數
q3, q1 = np.percentile(data.serum_creatinine[data.DEATH_EVENT==0], [75,25])
iqr = q3 - q1
upper = q3 + 1.5*iqr
lower = q1 - 1.5*iqr

q3_1, q1_1 = np.percentile(data.serum_creatinine[data.DEATH_EVENT==1], [75,25])
iqr_1 = q3_1 - q1_1
upper_1 = q3_1 + 1.5*iqr_1
lower_1 = q1_1 - 1.5*iqr_1

print("Outliers Upper with no death")
print(sorted(data.serum_creatinine[(data.DEATH_EVENT==0) & (data.serum_creatinine>upper)]))
print("Outliers Lower with no death")
print(sorted(data.serum_creatinine[(data.DEATH_EVENT==0) & (data.serum_creatinine<lower)]))
print("Outliers Upper with death")
print(sorted(data.serum_creatinine[(data.DEATH_EVENT==1) & (data.serum_creatinine>upper_1)]))
print("Outliers Lower with death")
print(sorted(data.serum_creatinine[(data.DEATH_EVENT==1) & (data.serum_creatinine<lower_1)]))


# In[13]:


# 將異常值以平均值去做取代
median = data.serum_creatinine[data.DEATH_EVENT==0].median()
median1 = data.serum_creatinine[data.DEATH_EVENT==1].median()

data.serum_creatinine[(data.DEATH_EVENT==0) & (data.serum_creatinine==1.7)]=median
data.serum_creatinine[(data.DEATH_EVENT==0) & (data.serum_creatinine==1.8)]=median
data.serum_creatinine[(data.DEATH_EVENT==0) & (data.serum_creatinine==2.1)]=median
data.serum_creatinine[(data.DEATH_EVENT==0) & (data.serum_creatinine==2.3)]=median
data.serum_creatinine[(data.DEATH_EVENT==0) & (data.serum_creatinine==2.4)]=median
data.serum_creatinine[(data.DEATH_EVENT==0) & (data.serum_creatinine==2.7)]=median
data.serum_creatinine[(data.DEATH_EVENT==0) & (data.serum_creatinine==3.2)]=median
data.serum_creatinine[(data.DEATH_EVENT==0) & (data.serum_creatinine==3.4)]=median
data.serum_creatinine[(data.DEATH_EVENT==0) & (data.serum_creatinine==3.5)]=median
data.serum_creatinine[(data.DEATH_EVENT==0) & (data.serum_creatinine==3.8)]=median
data.serum_creatinine[(data.DEATH_EVENT==0) & (data.serum_creatinine==5)]=median
data.serum_creatinine[(data.DEATH_EVENT==0) & (data.serum_creatinine==6.1)]=median


data.serum_creatinine[(data.DEATH_EVENT==1) & (data.serum_creatinine==3.5)]=median1
data.serum_creatinine[(data.DEATH_EVENT==1) & (data.serum_creatinine==3.7)]=median1
data.serum_creatinine[(data.DEATH_EVENT==1) & (data.serum_creatinine==4.0)]=median1
data.serum_creatinine[(data.DEATH_EVENT==1) & (data.serum_creatinine==4.4)]=median1
data.serum_creatinine[(data.DEATH_EVENT==1) & (data.serum_creatinine==5.8)]=median1
data.serum_creatinine[(data.DEATH_EVENT==1) & (data.serum_creatinine==6.8)]=median1
data.serum_creatinine[(data.DEATH_EVENT==1) & (data.serum_creatinine==9.0)]=median1
data.serum_creatinine[(data.DEATH_EVENT==1) & (data.serum_creatinine==9.4)]=median1


# In[14]:


# serum_creatinine
fig = px.box(data, x="DEATH_EVENT", y="serum_creatinine", points="all", color="DEATH_EVENT")
fig.show()


# In[15]:


# ejection_fraction
fig = px.box(data, x="DEATH_EVENT", y="ejection_fraction", points="all", color="DEATH_EVENT")
fig.show()


# In[16]:


# 取四分位數
q3, q1 = np.percentile(data.ejection_fraction[data.DEATH_EVENT==0], [75,25])
iqr = q3 - q1
upper = q3 + 1.5*iqr
lower = q1 - 1.5*iqr

q3_1, q1_1 = np.percentile(data.ejection_fraction[data.DEATH_EVENT==1], [75,25])
iqr_1 = q3_1 - q1_1
upper_1 = q3_1 + 1.5*iqr_1
lower_1 = q1_1 - 1.5*iqr_1

print("Outliers Upper with no death")
print(sorted(data.ejection_fraction[(data.DEATH_EVENT==0) & (data.ejection_fraction>upper)]))
print("Outliers Lower with no death")
print(sorted(data.ejection_fraction[(data.DEATH_EVENT==0) & (data.ejection_fraction<lower)]))
print("Outliers Upper with death")
print(sorted(data.ejection_fraction[(data.DEATH_EVENT==1) & (data.ejection_fraction>upper_1)]))
print("Outliers Lower with death")
print(sorted(data.ejection_fraction[(data.DEATH_EVENT==1) & (data.ejection_fraction<lower_1)]))


# In[17]:


median = data.ejection_fraction[data.DEATH_EVENT==0].median()
median1 = data.ejection_fraction[data.DEATH_EVENT==1].median()

data.ejection_fraction[(data.DEATH_EVENT==0) & (data.ejection_fraction==62)]=median
data.ejection_fraction[(data.DEATH_EVENT==0) & (data.ejection_fraction==80)]=median
data.ejection_fraction[(data.DEATH_EVENT==0) & (data.ejection_fraction==17)]=median


data.ejection_fraction[(data.DEATH_EVENT==1) & (data.ejection_fraction==60)]=median1
data.ejection_fraction[(data.DEATH_EVENT==1) & (data.ejection_fraction==62)]=median1
data.ejection_fraction[(data.DEATH_EVENT==1) & (data.ejection_fraction==65)]=median1
data.ejection_fraction[(data.DEATH_EVENT==1) & (data.ejection_fraction==70)]=median1


# In[18]:


# ejection_fraction
fig = px.box(data, x="DEATH_EVENT", y="ejection_fraction", points="all", color="DEATH_EVENT")
fig.show()


# In[19]:


# time(因為離群值數量較少，選擇不用median取代，效果不大)
fig = px.box(data, x="DEATH_EVENT", y="time", points="all", color="DEATH_EVENT")
fig.show()


# In[20]:


#劃分X和y
# 以下參數為特徵選擇樹依序對目標變量最高的三個特徵
list = ['serum_creatinine',  'ejection_fraction', 'time']
X = data[list]  

# 非特徵工程使用下面這個X
# X = data.drop(columns = ['DEATH_EVENT'])

y = data['DEATH_EVENT']
# X.head()


# In[21]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=2020)


# In[22]:


# 正規化
scaler = StandardScaler()
scaler_Xtrain = scaler.fit_transform(X_train) 
scaler_Xtest = scaler.fit_transform(X_test) 

lr = LogisticRegression()
lr.fit(scaler_Xtrain, y_train)
test_pred = lr.predict(scaler_Xtest)

print("Accuracy")
print(accuracy_score(y_test, test_pred))
print("F1 score")
print(f1_score(y_test, test_pred))
print("Precision")
print(precision_score(y_test, test_pred))
print("Recall")
print(recall_score(y_test, test_pred))


# draw a diagram
plt.figure(figsize=(10, 7))
plot_confusion_matrix(lr, X_test, y_test, cmap='Greens') 
plt.title("DecisionTreeClassifier - Confusion Matrix", fontsize=15)
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=12)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=12)
plt.show() 


# In[23]:


# DecisionTreeClassifier
clf = DecisionTreeClassifier(criterion='gini', max_depth=5, random_state=1)
clf.fit(X_train, y_train)
test_pred = clf.predict(X_test)  



print("Accuracy")
print(accuracy_score(y_test, test_pred))
print("F1 score")
print(f1_score(y_test, test_pred))
print("Precision")
print(precision_score(y_test, test_pred))
print("Recall")
print(recall_score(y_test, test_pred))

# draw a diagram
plt.figure(figsize=(10, 7))
plot_confusion_matrix(clf, X_test, y_test, cmap='Greens') 
plt.title("DecisionTreeClassifier - Confusion Matrix", fontsize=15)
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=12)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=12)
plt.show()  


# In[24]:


# RandomForestClassifier
rfc = RandomForestClassifier(n_estimators=1000, random_state=1)

parameters = {'max_depth': np.arange(2, 20, 1) }
GS = GridSearchCV(rfc, param_grid=parameters, cv=10, scoring='f1', n_jobs=-1)  
GS.fit(X_train, y_train)  

test_pred = GS.best_estimator_.predict(X_test)


print("Accuracy")
print(accuracy_score(y_test, test_pred))
print("F1 score")
print(f1_score(y_test, test_pred))
print("Precision")
print(precision_score(y_test, test_pred))
print("Recall")
print(recall_score(y_test, test_pred))

# draw a diagram
plt.figure(figsize=(10, 7))
plot_confusion_matrix(GS, X_test, y_test, cmap='Greens') 
plt.title("DecisionTreeClassifier - Confusion Matrix", fontsize=15)
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=12)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=12)
plt.show() 


# In[25]:


#GradientBoostingClassifier
gbl = GradientBoostingClassifier(n_estimators=1000, random_state=1)

parameters = {'max_depth': np.arange(2, 20, 1) }
GS = GridSearchCV(gbl, param_grid=parameters, cv=10, scoring='f1', n_jobs=-1)  
GS.fit(X_train, y_train)  

test_pred = GS.best_estimator_.predict(X_test)


print("Accuracy")
print(accuracy_score(y_test, test_pred))
print("F1 score")
print(f1_score(y_test, test_pred))
print("Precision")
print(precision_score(y_test, test_pred))
print("Recall")
print(recall_score(y_test, test_pred))

# draw a diagram
plt.figure(figsize=(10, 7))
plot_confusion_matrix(GS, X_test, y_test, cmap='Greens') 
plt.title("DecisionTreeClassifier - Confusion Matrix", fontsize=15)
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=12)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=12)
plt.show() 


# In[26]:


#LGBMClassifier
lgb_clf = lightgbm.LGBMClassifier(boosting_type='gbdt', random_state=1)

parameters = {'max_depth': np.arange(2, 20, 1) }
GS = GridSearchCV(lgb_clf, param_grid=parameters, cv=10, scoring='f1', n_jobs=-1)  
GS.fit(X_train, y_train)  

test_pred = GS.best_estimator_.predict(X_test)

print("Accuracy")
print(accuracy_score(y_test, test_pred))
print("F1 score")
print(f1_score(y_test, test_pred))
print("Precision")
print(precision_score(y_test, test_pred))
print("Recall")
print(recall_score(y_test, test_pred))

# draw a diagram
plt.figure(figsize=(10, 7))
plot_confusion_matrix(GS, X_test, y_test, cmap='Greens') 
plt.title("DecisionTreeClassifier - Confusion Matrix", fontsize=15)
plt.xticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=12)
plt.yticks(range(2), ["Heart Not Failed","Heart Fail"], fontsize=12)
plt.show() 


# In[ ]:




