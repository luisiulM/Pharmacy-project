import numpy as np
import scipy as scp
import pandas as pd
import os


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

def excel_extract(filename, keyword = ['category','all'], num_parameter = 100, case = 'Tilfeller', cancer_type = 'Anus', num_regions = 20):
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
                #print('third if case: ' , [j,dataframe[category][j*num_regions]])
                # Run throught the entire data set
                for i in range(len(np.array( dataframe[category] ))):
                
                    # checking if it is the right parameter and cancer, then save the value
                    if dataframe[category][i] == dataframe[category][j*num_regions] and dataframe['Kreftform'][i] == cancer_type:
                        data_parameter_per_case.append( dataframe[data_category][i] )
    
                data_parameter.append( np.array(data_parameter_per_case) )
        
    else:
        data_parameter = []
        for i in range(len(np.array( dataframe[category] ))):
            if dataframe[category][i] == parameter and dataframe['Kreftform'][i] == cancer_type:
                data_parameter.append( np.array(dataframe[data_category][i]) )
        
    return data_parameter

def Prescription_data(filename, case = 'Age', grouping = 10):
    '''
    This code was specifically implemented to handle Prescription Database excel files.
    '''
    processed_data = excel_lines(filename)
    
    if case == 'Age':
        dataframe = processed_data[8:293]
        pd_registry = dataframe

        pd_registry['Year'] = dataframe['Unnamed: 1']
        pd_registry['Age'] = dataframe['Unnamed: 2']
        pd_registry['Gender'] = dataframe['Unnamed: 3']
        pd_registry['Cases'] = dataframe['Unnamed: 6']
        
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
            
            pd_data2.append( np.array(pd_data[i]) )
        
        if grouping == 5:
            pd_cases = pd_data2
            
            xlabel = ['00–04', '05–09', '10–14', '15–19', '20–24', '25–29', '30–34', '35–39', '40–44', '45–49', '50–54', '55–59', '60–64', '65–69', '70–74', '75–79', '80–84', '85–89', '90+']
    
        if grouping == 10:
            ## adding ages together, so that they match the cancer age periods.
            pd_cases = []
            for i in range(len(pd_data2)):
                pd_data3 = []
                for j in range(0,len(pd_data2[i])-3,2):
                    pd_data3.append( pd_data2[i][j] + pd_data2[i][j+1])
    
                # last three 
                pd_data3.append( pd_data2[i][j+2] + pd_data2[i][j+3] + pd_data2[i][j+4])
    
                pd_cases.append( np.array(pd_data3) )
        
            xlabel = ['00–09', '10–19', '20–29', '30–39', '40–49', '50–59', '60–69', '70–79', '80+']
    
    elif case == 'Region':
        dataframe = processed_data[8:353] 
        pd_registry = dataframe

        pd_registry['Year'] = dataframe['Unnamed: 1']
        pd_registry['Age'] = dataframe['Unnamed: 2']
        pd_registry['Gender'] = dataframe['Unnamed: 3']
        pd_registry['Region'] = dataframe['Unnamed: 4']
        pd_registry['Cases'] = dataframe['Unnamed: 6']
        
        ## labels
        pd_year_label = []
        for i in range(4,10):
            pd_year_label.append( 'Year: 200'+str(i)+'' )
        for i in range(10,19):
            pd_year_label.append( 'Year: 20'+str(i)+'' )

        pd_divition = np.array(pd_registry['Region'][0:5])
        pd_regions = np.array(pd_registry['Region'][5:23])

        ### extracting data
        pd_data = []
        for i in range(len(pd_year_label)):
            start = i*(len(pd_regions)+len(pd_divition))
            end = (1 + i)*(len(pd_regions)+len(pd_divition))
    
            pd_data.append( np.array(pd_registry['Cases'][start:end]) )

        ## changing 'under 5' to 2
        pd_data2 = []
        for i in range(len(pd_data)):
            for j in range(len(pd_data[i])):
                if pd_data[i][j] == 'under 5':
                    pd_data[i][j] = 2
                
            pd_data2.append( pd_data[i] )

        ## separating divitions and regions
        pd_casesD = []
        pd_casesR = []
        for i in range(len(pd_year_label)):
            pd_casesD.append( pd_data[i][0:5] )
            pd_casesR.append( pd_data[i][5:] )
            
        xlabel = pd_regions
        pd_cases = pd_casesR
        
    return pd_cases , xlabel

def grouping(data, group = 10):
    '''
    This function groups data by the given numbers.
    '''
    data = np.array(data)
    g = group
    
    #print('K: ', K)
    #print('data shape: ', np.shape(data))
    if len(np.shape(data)) > 1:
        K = int(np.floor(len(data[0])/g))
        x = [(np.floor(g/2) + i*g) for i in range(K) ]
        
        data_new = np.zeros((len(data),K))
        for i in range(len(data)):
            for k in range(0, K):
                start = int(g*k)     
                end = int(g*k + g)
        
                data_new[i,k] = sum( np.array(data[i,start:end]) ) 
        
        if int(K*g) < len(data):
            data_new[i,k] = sum(data[i,int(K*g):])
        
        data_new = data_new.tolist()
    
    else:
        K = int(np.floor(len(data)/g))
        x = [(np.floor(g/2) + i*g) for i in range(K) ]
        
        data_new = np.zeros(K)
        for k in range(0, K):
            start = int(g*k)     
            end = int(g*k + g)
        
            data_new[k] = sum( np.array(data[start:end]) ) 
        
        if K*g < len(data):
            data_new[k] = sum(data[start:])
         
    return data_new , x

def create_folder(directory):
    try:
        os.mkdir(directory)

    except FileExistsError:
        print('The directory named, '+directory+', already exists.' )