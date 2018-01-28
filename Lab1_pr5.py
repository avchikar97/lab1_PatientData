# In[5]:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

### Questions ###
# a) There are 452 patients and 280 features.
# b) Column 0 is probably age because of the concentration around the middle, the low number of patients in the beginning bins, and the sudden dropoff around the 67 or 70 year mark.
#    Column 1 is probably gender since the data given only has two choices and the distribution is approaching even (0 = 252, 1 = 250)
#    Column 2 is probably height in centimeters because it's so concentrated between 155 (~5'2") and 175 (~5'10") range. The average male height in the U.S. is around 5'10" while the average female height is around 5'4".
#    Column 3: the majority of patients appear to have col3 reading between 50 and 80 with one notable outlier at 175. It could be a a mistake, like the two outliers from col2, but it's best to assume they are true readings.
#                since we are assuming the metric system for column 2, if we conclude the third column refers to weight in kg, it makes sense. Even the 175 kg outlier, while overweight provides a plausible weight for column 3 as the weight column
# c) Yes there are missing values. This can be seen by printing df_na_columns or by running and printing df.isnull().any().any().
# d) One way to test which features influence one another would be to assign any two features, plot it, and check correlation. This is likely to be a very slow method and it might not work if the data is not in the form of strictly numbers
#    if it's something checking whether or not gender affects certain features. I think the best way to do it would be to groupby() a certain column and see if there are any trends that stick out as possibly related then manipulate the data
#    in other ways such as create histogram or plot to see if it's possible to see a further possible correlation. Linear regression and checking the error value of each data point is also an option.
# I would say the three most important features are (1) Condition (2) Gender and (3) Age since the last can make certain conditions more likely to occur and they affect the treatment regimen of the patient.
#     Condition is the most important feature because it is the absolute deciding factor in the treatment of the patient and 

df = pd.read_csv('PatientData.csv', header=None, na_values='?') # header=None so that top row isn't seen as column indices

## Column 0 analysis
df[0].hist(bins=20)
plt.xlabel('Column 0 distribution')
plt.show()

## Column 1 analysis
plt.figure()
df[1].hist(bins=10)
plt.xlabel('Column 1 distribution')
plt.show()

##Column 2 analysis 
df[2].value_counts()

## Column 3 analysis
plt.figure()
df[3].hist(bins=30)
plt.xlabel('Column 3 distribution')
plt.show()

## Replacing missing value (NaN)
rows, columns = df.shape
df_na = df.isnull() # to tell if the entry is NaN
df_na_columns = df.isnull().any() # to tell if the column contains NaN
df_mean = np.zeros(columns) # used to store column averages
df_sum = df.sum(axis=1) # mean calculation (total in column)
df_dummy = df.notnull().sum() # how many entries are non-NaN for averaging
df_defined = np.array(df_dummy, dtype=int) # put it into numpy array
for column in range(columns):
    df_mean[column] = df_sum[column]
df_mean = np.divide(df_mean, df_defined) # final mean calculation

for column in range(columns):
    if(df_na_columns[column]):
        for row in range(rows):
            if(df_na[column][row]):
                df[column][row] = df_mean[column] # replacing NaN values
