import numpy as np
import scipy as scp
import pandas as pd


### Loading files
#--------------------------------------------------------------------------------------------------
def file_lines(filename):
    '''
    This function merely opens the file, reads it and returns the contant as a column vector [n,1].
    Where each line in the file is a row.
    
    filename: Name of the file, must be in string format.
    '''    
    infile = open(filename, "r")
    lines = infile.readlines()
    
    return lines

def excel_lines(filename):
    '''
    This function reads excel files and returns it's 'pandas' data frame.
    
    filename: Name of the file, must be in string format. f.eg r'directory/file_name'
    '''
    data_frame = pd.read_excel(filename)
    
    return data_frame
#--------------------------------------------------------------------------------------------------


def data_extract(filename):
    '''
    This marvelous function extract the data from our recent nemesis, log files from lammps.
    
    filename: File name. Must be a string text and the file must lie in the same directory.
    str_line: The line the data starts at.
    end_line: The line the data ends at.
    '''
    lines = file_lines(filename) # extracting lines
    
    n_data_types = len(lines[str_line].split()) # The number of data-types/columns
    n_elements = end_line - str_line            # The number of elements in each column
    
    data = np.zeros([n_elements,n_data_types])
    
    ## Extracting data
    # The first column will always refer to the timesteps
    for i in range(str_line, end_line):
        words = lines[i].split()
        
        for j in range(n_data_types):
            data[(i - str_line),j] = float(words[j])
        
    return data

def excel_extract(filename, keyword = ['category','all'], num_parameter = 100, case = 'Tilfeller', cancer_type = 'Anus'):
    dataframe = excel_lines(filename)
    
    category = keyword[0]
    parameter = keyword[1]
    data_category = case
    
    ### checking if the number of parameters exceeds the amount of data
    if num_parameter > len(dataframe[category]):
        print('The number of parameters exceeds the data length by: ', len(dataframe[category])/num_parameter)
    
    ### data extracting 
    if parameter == 'all':
        data_parameter = []
        for j in range(num_parameter):
            data_parameter_per_case = []
            
            ## "if" cases handle the "age" data
            if j+1 < num_parameter and dataframe[category][j] != dataframe[category][j+1]:
                #print('first if case: ' , j)
                for i in range(len(np.array( dataframe[category] ))):
                
                    # checking if it is the right parameter and cancer
                    if dataframe[category][i] == dataframe[category][j] and dataframe['Kreftform'][i] == cancer_type:
                        data_parameter_per_case.append( dataframe[data_category][i] )
    
                data_parameter.append( np.array(data_parameter_per_case) )
        
            elif j+1 == num_parameter and dataframe[category][j] != dataframe[category][j-1]:
                #print('second if case: ' , j)
                for i in range(len(np.array( dataframe[category] ))):
                
                    # checking if it is the right parameter and cancer
                    if dataframe[category][i] == dataframe[category][j] and dataframe['Kreftform'][i] == cancer_type:
                        data_parameter_per_case.append( dataframe[data_category][i] )
    
                data_parameter.append( np.array(data_parameter_per_case) )
            
            ## The "else" case handles the "Region" data
            else:
                #print('third if case: ' , [j,dataframe[category][j]])
                # Run throught the entire data set
                for i in range(len(np.array( dataframe[category] ))):
                
                    # checking if it is the right parameter and cancer, then save the value
                    if dataframe[category][i] == dataframe[category][j*num_parameter] and dataframe['Kreftform'][i] == cancer_type:
                        data_parameter_per_case.append( dataframe[data_category][i] )
    
                data_parameter.append( np.array(data_parameter_per_case) )
        
    else:
        data_parameter = []
        for i in range(len(np.array( dataframe[category] ))):
            if dataframe[category][i] == parameter and dataframe['Kreftform'][i] == cancer_type:
                data_parameter.append( np.array(dataframe[data_category][i]) )
        
    return data_parameter

def Prescription_data(processed_data, case = 'age'):
    '''
    This code was specifically implemented to handle Prescription Database excel files. (Not perfect)
    '''
    pd_registry = processed_data
    
    ## just some labels
    #-------------------
    pd_year_label = []
    for i in range(4,10):
        pd_year_label.append( 'Year: 200'+str(i)+'' )
    for i in range(10,19):
        pd_year_label.append( 'Year: 20'+str(i)+'' )
    pd_age = np.array(pd_registry['Age'][0:19])
    #-------------------
    
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
    
        pd_cases.append( np.array(pd_data3) )
        
    return pd_cases