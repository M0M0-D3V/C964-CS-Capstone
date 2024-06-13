import pandas as pd

# Load the ddataset
df = pd.read_csv('fake-real-job/fake_job_postings.csv')

# Identify problem columns
print(df.dtypes)



# fill missing values with mean values
df.fillna(df.mean(), inplace=True)

# Calculate the IQR
Q1 =df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 - Q1

# Remove outliers
df - df[~((df < (Q1 - 1.5 * IQR)) | (df > (Q3 + 1.5 * IQR))).any(axis=1)]

# Save the cleaned data
df.to_csv('cleaned_fake_job_postings.csv', index=False)