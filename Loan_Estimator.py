import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn import metrics
from sklearn.svm import SVC
from imblearn.over_sampling import RandomOverSampler
import csv

df = pd.read_csv('Dataset.csv')

# allows you pandas to prind ALL columns of a table
pd.options.display.max_columns = None
pd.options.display.max_rows = None
df = df.drop('Dependents', axis='columns')

# Replace any NAN values with the mean for that column

print(df.isnull().sum())  # Check for remaining NaNs
numeric_cols = df.select_dtypes(include=['float64', 'int64']).columns  # Select numeric columns
df.replace('', pd.NA, inplace=True)
for col in numeric_cols:
    mean_value = df[col].mean() 
    df[col].fillna(mean_value, inplace=True)

#print(df.head() ,'\n')
#print(df.describe()['LoanAmount']['count']) this is used to extract specific data can be used later on
# loan_count = df['Loan_Status'].value_counts()
# print(loan_count)
# plt.pie(loan_count.values,
#         labels=loan_count.index,
#         autopct='%.3f%%')
# plt.show()
# max_gender_count = len(df[df['Gender'] == 'Male']['Gender'])

# fig, axes =  plt.subplots(figsize=(15, 7))
# plt.tick_params(left = False, right = False , labelleft = False , 
#                 labelbottom = False, bottom = False) 
# for i, col in enumerate(['Gender', 'Married']):
#     plt.subplot(1, 2, i+1)
#     sb.countplot(data=df, x=col, hue='Loan_Status')
# plt.tight_layout()
# plt.show()


# plt.subplots(figsize=(15, 5))
# plt.tick_params(left = False, right = False , labelleft = False , 
#                 labelbottom = False, bottom = False) 
# for i, col in enumerate(['ApplicantIncome', 'LoanAmount']):
#     plt.subplot(1, 2, i+1)
#     sb.distplot(df[col])
# plt.tight_layout()
# plt.show()
# loan amount is the number of £1000 so 70 meaning £70k



# plt.subplots(figsize=(15, 5))
# plt.tick_params(left = False, right = False , labelleft = False , 
#                 labelbottom = False, bottom = False) 
# for i, col in enumerate(['ApplicantIncome', 'LoanAmount']):
#     plt.subplot(1, 2, i+1)
#     sb.boxplot(df[col])
# plt.tight_layout()
# plt.show()


# adjusted to remove the strongest outliers from the set
df = df[df['ApplicantIncome'] < 30000]
df = df[df['LoanAmount'] < 550]

#males on average requested a higher loan amount than females
#print(df.groupby('Gender').mean(numeric_only=True)['LoanAmount'])
# Function to apply label encoding
def encode_labels(data):
    # for each coplumn in the data passed
    for col in data.columns:
        if data[col].dtype == 'object': # if the data type is an object
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col])

    return data

# Applying function in whole column
df = encode_labels(df)

features = df.drop('Loan_Status', axis=1)  # remove loan status since we dont want to include in training features
target = df['Loan_Status'].values   # the outcome due to the features

X_train, X_val, Y_train, Y_val = train_test_split(features, target, 
                                    test_size=0.2, 
                                    random_state=10) # random state here is a seed so data will always be split in the same way


ros = RandomOverSampler(sampling_strategy='minority', 
                        random_state=0) 

X, Y = ros.fit_resample(X_train, Y_train) 
scaler = StandardScaler() # from maybe [0,5,10,15,25] to [-10, -5, 0 , 5, 10] just means centered around 0 and the larger numbers dont over contribute
# standardizing the data here by fitting it (calc mean and other stats) then transform it (standardizethe data)
X = scaler.fit_transform(X)
X_val = scaler.transform(X_val)

from sklearn.metrics import roc_auc_score
model = SVC(kernel='linear')
model.fit(X, Y)

print('Training Accuracy : ', metrics.roc_auc_score(Y, model.predict(X)))
print('Validation Accuracy : ', metrics.roc_auc_score(Y_val, model.predict(X_val)))