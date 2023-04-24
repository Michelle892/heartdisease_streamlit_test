#!/usr/bin/env python
# coding: utf-8

# # Heart Disease Prediction System

# ## Importing the libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier


# ## Data Preprocessing

# #Reading CSV files

# In[2]:


dataset = pd.read_csv('Heart_disease_dataset.csv')


# In[3]:


heart_data = dataset[~dataset.isin(['?'])]


# In[4]:


heart = heart_data.fillna(0.0)


# In[5]:


heart["result"] = np.where(heart["result"] > 0, 1, heart["result"])


# In[6]:


info = ["        age","        1: male, 0: female","        chest pain type, 1: typical angina, 2: atypical angina, 3: non-anginal pain, 4: asymptomatic","resting blood pressure","        serum cholestoral in mg/dl","        fasting blood sugar > 120 mg/dl","        resting electrocardiographic results (values 0,1,2)","maximum heart rate achieved","        exercise induced angina","ST depression induced by exercise relative to rest","        the slope of the peak exercise ST segment","        number of major vessels (0-3) colored by flourosopy","        thal: 3 = normal; 6 = fixed defect; 7 = reversable defect"]
for i in range(len(info)):
    print(heart.columns[i]+":\t"+info[i])


# In[7]:


dataset.info()


# #Top 10 data

# In[8]:


heart.head(10)


# Tail 10 

# In[9]:


heart.tail(10)


# Checking rows and columns

# In[10]:


print("(Rows, columns): " + str(heart.shape))
heart.columns


# ##Checking data size

# In[11]:


heart.size


# ##Getting the statistical measures of the data

# ##Summarizes the count, mean, standard deviation, min, and max for numeric variables.

# In[12]:


heart.describe()


# Testing Missing Values

# In[13]:


heart.isnull().sum()


# ##Checking the result

# 0 ==> heart is okay, and 1,2,3,4 ==> heart disease.

# In[14]:


heart['result'].value_counts()


# ## Data Visualization

# In[15]:


heart.hist(figsize=(14,14))
plt.show()


# In[16]:


countNoDisease = len(heart[heart.result == 0])
countHaveDisease = len(heart[heart.result > 0])
print("Percentage of Patients Haven't Heart Disease: {:.2f}%".format((countNoDisease / (len(heart.result))*100)))
print("Percentage of Patients Have Heart Disease: {:.2f}%".format((countHaveDisease / (len(heart.result))*100)))


# In[17]:


plt.figure(figsize=(10, 5))
sns.countplot(x="result", data=heart, palette="mako_r")
plt.xlabel("0=no Heart Disease,     1=Heart Disease")
plt.show()


# In[18]:


countFemale = len(heart[heart.sex == 0])
countMale = len(heart[heart.sex == 1])
print("Percentage of Female Patients: {:.2f}%".format((countFemale / (len(heart.sex))*100)))
print("Percentage of Male Patients: {:.2f}%".format((countMale / (len(heart.sex))*100)))


# In[19]:


plt.figure(figsize=(50, 50))
sns.catplot(x= 'sex', y='age', data= heart,hue= 'result', palette= 'husl')
plt.title('Sex Vs Age')
plt.xlabel('Sex')
plt.ylabel('Age')
plt.show()


# In[20]:


plt.figure(figsize=(10, 5))
plt.subplot(121)
sns.violinplot(x="result", y="thalach", data=heart, inner=None)
sns.swarmplot(x="result", y="thalach", data=heart, color='w', alpha=0.5)

plt.subplot(122)
sns.swarmplot(x="result", y="thalach", data=heart)
plt.show()


# In[21]:


# create four distplots
plt.figure(figsize=(12,10))
plt.subplot(221)
sns.distplot(heart[heart['result']==0].age)
plt.title('Age of patients without heart disease')
plt.subplot(222)
sns.distplot(heart[heart['result']==1].age)
plt.title('Age of patients with heart disease')
plt.subplot(223)
sns.distplot(heart[heart['result']==0].thalach )
plt.title('Max heart rate of patients without heart disease')
plt.subplot(224)
sns.distplot(heart[heart['result']==1].thalach )
plt.title('Max heart rate of patients with heart disease')
plt.show()


# In[22]:



sns.boxplot(x = 'sex', y='chol', data= heart,hue= 'result', palette= 'mako')
plt.title('Sex Vs Cholesterol')
plt.xlabel('Sex')
plt.ylabel('Cholesterol')
plt.show()


# In[23]:


plt.figure(figsize=(20,10))
sns.barplot(x= 'fbs',y = 'chol', data= heart, hue= 'result', palette='plasma')
plt.xlabel('Fasting blood sugar')
plt.ylabel('Cholesterol')
plt.show()


# ## Separating Feature and Label

# In[24]:


#Separate Feature and Label 
feature = heart.drop('result',axis = 1) 
label = heart.result


# In[25]:


feature


# In[26]:


label


# ## Data Standardization

# In[27]:


scaler = StandardScaler()
X_scale = scaler.fit_transform(feature)


# In[28]:


X_scale


# ## Spliting Training data and Testing data (80%,20%)

# In[29]:


X_train, X_test, Y_train, Y_test = train_test_split(X_scale, label, test_size=0.2, random_state=0)


# In[30]:


print ("train_set_x shape: " + str(X_train.shape))
print ("train_set_y shape: " + str(Y_train.shape))
print ("test_set_x shape: " + str(X_test.shape))
print ("test_set_y shape: " + str(Y_test.shape))


# In[31]:


scores_dict = {}


# ## SVM (Support Vector Machine)

# In[32]:


#Create a svm Classifier
clf = svm.SVC(kernel='linear') # Linear Kernel


# In[33]:


#Train the model using the training sets
clf.fit(X_train, Y_train)


# In[34]:


#Predict the response for test dataset
y_pred = clf.predict(X_test)


# Checking accuracy, precision, recall, f1_score and confusion matrix

# In[35]:


from sklearn.metrics import confusion_matrix
confusion_matrix(Y_test, y_pred)


# In[36]:


print('Accuracy',accuracy_score(Y_test,y_pred)*100)
print('Precision',precision_score(Y_test,y_pred)*100)
print('Recall',precision_score(Y_test,y_pred)*100)
print('F1',f1_score(Y_test,y_pred)*100)


# ## Tuning SVM

# In[37]:


from sklearn.model_selection import GridSearchCV


# In[38]:


param_grid = {'C':[0,1,1,10,100], 'gamma':[1,0.1,0.01,0.001], 'kernel':['rbf','poly','sigmoid','linear']}


# In[39]:


grid = GridSearchCV(svm.SVC(),param_grid,refit=True,verbose=2)


# In[40]:


grid.fit(X_train,Y_train)


# In[41]:


print(grid.best_estimator_)


# In[42]:


grid_predictions = grid.predict(X_test)


# In[43]:


grid_predictions


# In[44]:


print('Accuracy:',accuracy_score(Y_test,grid_predictions)*100)
print('Precision:',precision_score(Y_test,grid_predictions)*100)
print('Recall:',precision_score(Y_test,grid_predictions)*100)
print('F1:',f1_score(Y_test,grid_predictions)*100)


# In[45]:


param_grid1 = {'C':[0.01,0.1,1,10,100], 'gamma':[0.1,0.5,0.01,0.001], 'kernel':['rbf','poly','sigmoid','linear']}


# In[46]:


grid1 = GridSearchCV(svm.SVC(),param_grid1,refit=True,verbose=2)


# In[47]:


grid1.fit(X_train,Y_train)


# In[48]:


print(grid1.best_estimator_)


# In[49]:


grid1_predictions = grid1.predict(X_test)


# In[50]:


print('Accuracy :',accuracy_score(Y_test,grid1_predictions)*100)
print('Precision:',precision_score(Y_test,grid1_predictions)*100)
print('Recall   :',precision_score(Y_test,grid1_predictions)*100)
print('F1       :',f1_score(Y_test,grid1_predictions)*100)


# In[51]:


scores_dict['SVM'] = accuracy_score(Y_test,grid1_predictions)*100


# In[52]:


import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.inspection import permutation_importance

svc =  svm.SVC(kernel='rbf', C=1)
svc.fit(X_train, Y_train)

perm_importance = permutation_importance(svc, X_test, Y_test)

feature_names = ['age','sex','cp','trestbps','chol','fbs','restcg','thalach','exang','oldpeak','slope','ca','thal']
features = np.array(feature_names)

sorted_idx = perm_importance.importances_mean.argsort()
plt.barh(features[sorted_idx], perm_importance.importances_mean[sorted_idx])
plt.xlabel("Permutation Importance")


# ## Decision Tree

# In[53]:


tree = DecisionTreeClassifier(random_state=0)


# In[54]:


tree.fit(X_train,Y_train)


# In[55]:


tree_test_pred = tree.predict(X_test)


# In[56]:


print("Accuracy :",accuracy_score(Y_test,tree_test_pred)*100)
print("Precision:",precision_score(Y_test,tree_test_pred)*100)
print("Recall   :",precision_score(Y_test,tree_test_pred)*100)
print("F1       :",f1_score(Y_test,tree_test_pred)*100)


# In[57]:


scores_dict['DecisionTree'] = accuracy_score(Y_test,tree_test_pred)*100


# ## Logistic Regression

# In[58]:


logistic = LogisticRegression()
logistic.fit(X_train,Y_train)


# In[59]:


lo_test_pred = logistic.predict(X_test) 


# In[60]:


print("Accuracy :",accuracy_score(Y_test,lo_test_pred)*100)
print("Precision:",precision_score(Y_test,lo_test_pred)*100)
print("Recall   :",precision_score(Y_test,lo_test_pred)*100)
print("F1       :",f1_score(Y_test,lo_test_pred)*100)


# In[61]:


scores_dict['LogisticRegression'] = accuracy_score(Y_test,lo_test_pred)*100


# KNN model

# In[62]:


Knn = KNeighborsClassifier()


# In[63]:


Knn.fit(X_train,Y_train)


# In[64]:


Knn_test_pred = Knn.predict(X_test)


# In[65]:


print("Accuracy :",accuracy_score(Y_test,Knn_test_pred)*100)
print("Precision:",precision_score(Y_test,Knn_test_pred)*100)
print("Recall   :",precision_score(Y_test,Knn_test_pred)*100)
print("F1       :",f1_score(Y_test,Knn_test_pred)*100)


# In[66]:


scores_dict['KNN'] = accuracy_score(Y_test,Knn_test_pred)*100


# In[67]:


with sns.color_palette('muted'):
  algo_name = list(scores_dict.keys())
  scoress = list(scores_dict.values())

  sns.set(rc={'figure.figsize':(15,7)})
  plt.xlabel("\nAlgorithms")
  plt.ylabel("Accuracy score")

  sns.barplot(algo_name,scoress)


# ## Input Data Using SVM

# In[68]:


data1 = (63 ,1, 3,145,233,1,0,150,0,2.3,0,0,1)


# In[69]:


data = (58.0,1.0,3.0,132.0,224.0,0.0,2.0,173.0,0.0,3.2,1.0,2.0,7.0)


# In[70]:


input_data =np.asarray(data)


# In[71]:


input_data_reshaped = input_data.reshape(1,-1)


# Standardize the input data

# In[72]:


std_data = scaler.transform(input_data_reshaped)


# In[73]:


prediction = grid1.predict(std_data)


# In[74]:


if (prediction[0]==0):
    print('No heart disease! Yay!!!')
else:
    print('The person has heart disease!!')    

