#!/usr/bin/env python
# coding: utf-8

# ## Importing Libraries

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import skew
from warnings import filterwarnings
filterwarnings(action='ignore')


# ## Loading Dataset

# In[2]:


df=pd.read_excel(r'C:\Users\User\Documents\Projects\ML Projects\NHL Games Prediction\game_teams_stats.xlsx')
print("Successfully Imported Data!")
df


# ## Data Profiling

# ### Shape of the dataset

# In[3]:


print(df.shape)


# ## Description

# ### Statistical summary of data

# In[4]:


df.describe()


# ### Feature information

# In[5]:


df.info()


# ## Structure of the Dataset

# In[6]:


df_shape = df.shape
print("The credit rating dataset has", df_shape[0], "records, each with", df_shape[1],"attributes")


# ## Finding Null Values

# In[7]:


print(df.isna().sum())


# In[8]:


df.corr()


# In[9]:


df.groupby('goals').mean()


# ## Treating Null Values

# In[10]:


df['head_coach']=df['head_coach'].ffill()
df['goals']=df['goals'].fillna(df['goals'].mode()[0])
df['shots']=df['shots'].fillna(df['shots'].mode()[0])
df['hits']=df['hits'].fillna(df['hits'].mode()[0])
df['pim']=df['pim'].fillna(df['pim'].mode()[0])
df['powerPlayOpportunities']=df['powerPlayOpportunities'].fillna(df['powerPlayOpportunities'].mode()[0])
df['powerPlayGoals']=df['powerPlayGoals'].fillna(df['powerPlayGoals'].mode()[0])
df['faceOffWinPercentage']=df['faceOffWinPercentage'].fillna(df['faceOffWinPercentage'].mode()[0])
df['giveaways']=df['giveaways'].fillna(df['giveaways'].mode()[0])
df['takeaways']=df['takeaways'].fillna(df['takeaways'].mode()[0])
df['blocked']=df['blocked'].fillna(df['blocked'].mode()[0])
df['startRinkSide']=df['startRinkSide'].fillna(df['startRinkSide'].mode()[0])


# ## Checking Null Values

# In[11]:


print(df.isna().sum())


# ## Performing Exploratory Data Analysis

# ### Checking For Outlier in dataset

# In[12]:


df.describe(percentiles=[0.05,0.5,0.997])


# In[13]:


df.columns


# ## Treating the Outliers

# In[14]:


Q1_goals = df.goals.quantile(0.25)
Q3_goals = df.goals.quantile(0.75)
IQR_goals = Q3_goals-Q1_goals

lower_limit_goals = Q1_goals-1.5*IQR_goals
upper_limit_goals = Q3_goals+1.5*IQR_goals

a_goals = df['goals'].median()

for x in ['goals']:
    q75,q25 = np.percentile(df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    df.loc[df[x] < min,x] = a_goals
    df.loc[df[x] > max,x] = a_goals


# In[15]:


Q1_shots = df.shots.quantile(0.25)
Q3_shots = df.shots.quantile(0.75)
IQR_shots = Q3_shots-Q1_shots

lower_limit_shots = Q1_shots-1.5*IQR_shots
upper_limit_shots = Q3_shots+1.5*IQR_shots

a_shots = df['shots'].median()

for x in ['shots']:
    q75,q25 = np.percentile(df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    df.loc[df[x] < min,x] = a_shots
    df.loc[df[x] > max,x] = a_shots


# In[16]:


Q1_hits = df.hits.quantile(0.25)
Q3_hits = df.hits.quantile(0.75)
IQR_hits = Q3_hits-Q1_hits

lower_limit_hits = Q1_hits-1.5*IQR_hits
upper_limit_hits = Q3_hits+1.5*IQR_hits

a_hits = df['hits'].median()

for x in ['hits']:
    q75,q25 = np.percentile(df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    df.loc[df[x] < min,x] = a_hits
    df.loc[df[x] > max,x] = a_hits


# In[17]:


Q1_pim = df.pim.quantile(0.25)
Q3_pim = df.pim.quantile(0.75)
IQR_pim = Q3_pim-Q1_pim

lower_limit_pim = Q1_pim-1.5*IQR_pim
upper_limit_pim = Q3_pim+1.5*IQR_pim

a_pim = df['pim'].median()

for x in ['pim']:
    q75,q25 = np.percentile(df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    df.loc[df[x] < min,x] = a_pim
    df.loc[df[x] > max,x] = a_pim  


# In[18]:


Q1_powerPlayOpportunities = df.powerPlayOpportunities.quantile(0.25)
Q3_powerPlayOpportunities = df.powerPlayOpportunities.quantile(0.75)
IQR_powerPlayOpportunities = Q3_powerPlayOpportunities-Q1_powerPlayOpportunities

lower_limit_powerPlayOpportunities = Q1_powerPlayOpportunities-1.5*IQR_powerPlayOpportunities
upper_limit_powerPlayOpportunities = Q3_powerPlayOpportunities+1.5*IQR_powerPlayOpportunities

a_powerPlayOpportunities = df['powerPlayOpportunities'].median()

for x in ['powerPlayOpportunities']:
    q75,q25 = np.percentile(df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    df.loc[df[x] < min,x] = a_powerPlayOpportunities
    df.loc[df[x] > max,x] = a_powerPlayOpportunities


# In[19]:


Q1_powerPlayGoals = df.powerPlayGoals.quantile(0.25)
Q3_powerPlayGoals = df.powerPlayGoals.quantile(0.75)
IQR_powerPlayGoals = Q3_powerPlayGoals-Q1_powerPlayGoals

lower_limit_powerPlayGoals = Q1_powerPlayGoals-1.5*IQR_powerPlayGoals
upper_limit_powerPlayGoals = Q3_powerPlayGoals+1.5*IQR_powerPlayGoals

a_powerPlayGoals = df['powerPlayGoals'].median()

for x in ['powerPlayGoals']:
    q75,q25 = np.percentile(df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    df.loc[df[x] < min,x] = a_powerPlayGoals
    df.loc[df[x] > max,x] = a_powerPlayGoals


# In[20]:


Q1_faceOffWinPercentage = df.faceOffWinPercentage.quantile(0.25)
Q3_faceOffWinPercentage = df.faceOffWinPercentage.quantile(0.75)
IQR_faceOffWinPercentage = Q3_faceOffWinPercentage-Q1_faceOffWinPercentage

lower_limit_faceOffWinPercentage = Q1_faceOffWinPercentage-1.5*IQR_faceOffWinPercentage
upper_limit_faceOffWinPercentage = Q3_faceOffWinPercentage+1.5*IQR_faceOffWinPercentage

a_faceOffWinPercentage = df['faceOffWinPercentage'].median()

for x in ['faceOffWinPercentage']:
    q75,q25 = np.percentile(df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    df.loc[df[x] < min,x] = a_faceOffWinPercentage
    df.loc[df[x] > max,x] = a_faceOffWinPercentage


# In[21]:


Q1_giveaways = df.giveaways.quantile(0.25)
Q3_giveaways = df.giveaways.quantile(0.75)
IQR_giveaways = Q3_giveaways-Q1_giveaways

lower_limit_giveaways = Q1_giveaways-1.5*IQR_giveaways
upper_limit_giveaways = Q3_giveaways+1.5*IQR_giveaways

a_giveaways = df['giveaways'].median()

for x in ['giveaways']:
    q75,q25 = np.percentile(df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    df.loc[df[x] < min,x] = a_giveaways
    df.loc[df[x] > max,x] = a_giveaways


# In[22]:


Q1_takeaways = df.takeaways.quantile(0.25)
Q3_takeaways = df.takeaways.quantile(0.75)
IQR_takeaways = Q3_takeaways-Q1_takeaways

lower_limit_takeaways = Q1_takeaways-1.5*IQR_takeaways
upper_limit_takeaways = Q3_takeaways+1.5*IQR_takeaways

a_takeaways = df['takeaways'].median()

for x in ['takeaways']:
    q75,q25 = np.percentile(df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    df.loc[df[x] < min,x] = a_takeaways
    df.loc[df[x] > max,x] = a_takeaways


# In[23]:


Q1_blocked = df.blocked.quantile(0.25)
Q3_blocked = df.blocked.quantile(0.75)
IQR_blocked = Q3_blocked-Q1_blocked

lower_limit_blocked = Q1_blocked-1.5*IQR_blocked
upper_limit_blocked = Q3_blocked+1.5*IQR_blocked

a_blocked = df['blocked'].median()

for x in ['blocked']:
    q75,q25 = np.percentile(df.loc[:,x],[75,25])
    intr_qr = q75-q25

    max = q75+(1.5*intr_qr)
    min = q25-(1.5*intr_qr)

    df.loc[df[x] < min,x] = a_blocked
    df.loc[df[x] > max,x] = a_blocked


# ## Data Analysis

# ### Countplot:

# In[24]:


sns.countplot(df['goals'])
plt.show()


# In[25]:


sns.countplot(df['shots'])
plt.show()


# In[26]:


sns.countplot(df['hits'])
plt.show()


# In[27]:


sns.countplot(df['powerPlayOpportunities'])
plt.show()


# In[28]:


sns.countplot(df['powerPlayGoals'])
plt.show()


# In[29]:


sns.countplot(df['faceOffWinPercentage'])
plt.show()


# In[30]:


sns.countplot(df['blocked'])
plt.show()


# In[31]:


sns.countplot(df['giveaways'])
plt.show()


# In[32]:


sns.countplot(df['takeaways'])
plt.show()


# In[33]:


sns.countplot(df['pim'])
plt.show()


# ## KDE plot:

# In[34]:


sns.kdeplot(df.query('goals > 2').goals)


# ## Distplot:

# In[35]:


sns.distplot(df['goals'])


# In[36]:


df.plot(kind ='box',subplots = True, layout =(4,4),sharex = False)


# In[37]:


df.plot(kind ='density',subplots = True, layout =(4,4),sharex = False)


# ## Histogram

# In[38]:


df.hist(figsize=(10,10),bins=50)
plt.show()


# ## Checking for Multicollinearity

# ## Heatmap for expressing correlation

# In[39]:


plt.figure(figsize =(15,10))
sns.heatmap(df.corr(),robust=True,fmt='.1g',linewidths=1.3,linecolor='gold',annot=True);


# ### Dropping columns that doesn't affect the output variable

# In[40]:


df.drop(df.columns[[0,1,4]], axis = 1, inplace = True)


# In[41]:


df


# ### One-Hot Encoding categorical variables in columns that affect the output variable

# In[42]:


df = pd.get_dummies(df, columns = ['HoA','settled_in','startRinkSide','won'], drop_first = True)


# In[43]:


df


# ## Model selection and building

# ### Splitting the X and Y values

# In[44]:


X = df.iloc[:,:-1].values
y = df.iloc[:,-1].values


# ### Splitting training and testing data

# In[45]:


from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)


# ## Feature Scaling (It is done to bring values of all the colums to same scale)

# In[46]:


from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
print(X_train)
print(X_test)


# ## Logistic Regression:

# In[47]:


from sklearn.linear_model import LogisticRegression
model = LogisticRegression()
model.fit(X_train,y_train)
Y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score,confusion_matrix
print("Accuracy Score:",accuracy_score(y_test,Y_pred))


# In[48]:


confusion_mat = confusion_matrix(y_test,Y_pred)
print('Confusion Matrix is\n',confusion_matrix(y_test,Y_pred))


# ## K-Nearest Neighbors:

# In[49]:


from sklearn.neighbors import KNeighborsClassifier
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,y_pred))


# ## Support Vector Classifier:

# In[50]:


from sklearn.svm import SVC
model = SVC()
model.fit(X_train,y_train)
pred_y = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,pred_y))


# ## Decision Tree:

# In[51]:


from sklearn.tree import DecisionTreeClassifier
model = DecisionTreeClassifier(criterion='entropy',random_state=7)
model.fit(X_train,y_train)
y_pred = model.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,y_pred))


# ## GaussianNB:

# In[52]:


from sklearn.naive_bayes import GaussianNB
model3 = GaussianNB()
model3.fit(X_train,y_train)
y_pred3 = model3.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,y_pred3))


# ## Random Forest Classifier:

# In[53]:


from sklearn.ensemble import RandomForestClassifier
model2 = RandomForestClassifier(random_state=1)
model2.fit(X_train, y_train)
y_pred2 = model2.predict(X_test)

from sklearn.metrics import accuracy_score
print("Accuracy Score:",accuracy_score(y_test,y_pred2))


# In[54]:


from sklearn.metrics import classification_report
report_model2=classification_report(y_test,y_pred2)
print(report_model2)


# In[55]:


results = pd.DataFrame({
    'Model': ['Logistic Regression','K-Nearest Neighbours', 'Support Vector Classifier','Decision Tree' ,'GaussianNB','Random Forest Classifier'],
    'Score': [0.777,0.729,0.778,0.757,0.754,0.807]})

result_df = results.sort_values(by='Score', ascending=False)
result_df = result_df.set_index('Score')
result_df


# In[56]:


#Observation:

#Random Forest Classifier performs well than other models

#Hence I will use Random Forest Classifier for training my model.


# In[57]:


conf_mat2=confusion_matrix(y_test,y_pred2)
plt.figure(figsize=(6,6))
sns.heatmap(conf_mat2,annot=True,fmt=".0f")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.title("Confusion Matrix for Random Forest Classifier")
plt.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




