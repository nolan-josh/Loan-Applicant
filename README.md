THIS IS A ML MODEL MADE BY @nolan-josh 
Using python testing various ML algorithms in order to determine the most suitable to create
a model that will intake applicant's data to then estimate if they are suitable for a loan or not

IMPORTANT NOTES FOR THE DATASET:
- The income values are monthly income not yearly
- The LoanAmount has been divided by 1000


Code that was used in order to produce the graphs: 

<!-- 
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
# plt.show() -->
