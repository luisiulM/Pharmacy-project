import preprocessor as pp
import pandas as pd
import numpy as np
import myplot as mpf


filenames = [r'data_files/kreftregisteret_menn_norge.xlsx', r'data_files/reseptregisteret_menn_norge.xls']

files = []
for i in range(len(filenames)):
    files.append(pp.excel_lines(filenames[i]))

### Making it more orderly
#----------------------------------------------------------------------------------
dataframe = files[1][8:293]
pd_registry = dataframe

# adding columns with desired 
pd_registry['Year'] = dataframe['Unnamed: 1']
pd_registry['Age'] = dataframe['Unnamed: 2']
pd_registry['Gender'] = dataframe['Unnamed: 3']
pd_registry['Cases'] = dataframe['Unnamed: 6']

for i in range(1,9):
    # Dropping old columns
    pd_registry.drop(columns = ['Unnamed: '+str(i)+''], inplace = True)

# the date is different for every file
pd_registry.drop(columns = ['Report date: 17/06/2019 14:43'], inplace = True)

### processing data
#----------------------------------------------------------------------------------
# just some labels
pd_year_label = []
for i in range(4,10):
    pd_year_label.append( 'Year: 200'+str(i)+'' )
for i in range(10,19):
    pd_year_label.append( 'Year: 20'+str(i)+'' )

pd_age = np.array(pd_registry['Age'][0:19])

## extracting data for each year, where each list corresponds to an age
pd_data = []
for i in range(len(pd_year_label)):
    start = i*len(pd_age)
    end = (1 + i)*len(pd_age)
    
    pd_data.append( np.array(pd_registry['Cases'][start:end]) )

## changing "under 5" labels data to 2
pd_data2 = []
for i in range(len(pd_data)):
    for j in range(len(pd_data[i])):
        if pd_data[i][j] == 'under 5':
            pd_data[i][j] = 2
            
    pd_data2.append( pd_data[i] )

## adding ages together, so that they match the cancer age periods.
pd_cases = []
for i in range(len(pd_data2)):
    pd_data3 = []
    for j in range(0,len(pd_data2[i])-3,2):
        pd_data3.append( pd_data2[i][j] + pd_data2[i][j+1])
    
    # last three 
    pd_data3.append( pd_data2[i][j+2] + pd_data2[i][j+3] + pd_data2[i][j+4])
    
    pd_cases.append( pd_data3 )