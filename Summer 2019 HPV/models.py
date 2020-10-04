import numpy as np
import myplot as mpf
import scipy.io as sio
import preprocessor as pp
import Regression_functions as rf

import warnings
#Comment this to turn on warnings
warnings.filterwarnings('ignore')


def HPV_model(age_start = 13, age_end = 75, period_start = 2004, period_end = 2017, gender = 'menn', 
              fast_plot = False, data_grouping = 0):
    '''
    This function gives the predicted number of people that are vaccinated at a particular "year" at a give "age". 
    To get a single value set the "start" and "end" parameters equal to each other.
    The models where trained by using reseptregisteret data.
    
    age_...:       These parameters determine the start and end of the different ages we want to consider.
    period_...:    These parameters determine the start and end of the different years we want to consider.
    gender:        This parameter specifies which model, one wants to use. The one fitted using mens vaccination data 
                   (gender = 'menn'), or the one fitted using womans vaccination data (gender = 'kvinne'). Must be a string.
    fast_plot:     To plot the result, one only needs to change from False to True. -> fast_plot = True. Saving the figure is 
                   not an option.
    data_grouping: This parameter decides how many point are grouped together. Meaning that if you have an age period of 
                   10 to 100 and a data_grouping = 3, then the data from age 10, 11 and 12 are added together, 
                   then 13, 14 and 15 etc ...
                   Nothing happens if data_grouping is equal to 0 or 1.
    
    
    Note: The number of vaccinations for ages under 9 and over 75 (since the number is small) are automatically set to zero. 
          The same applies to the years before 2006 for women and 2007 for men.
    
    Return: This function returns three values; prediction, age and year in it given order (pred, ages , years). 
            Where the prediction can either be a single value, a vector or a list of vectors in which each list represents 
            a year.
    '''
    
    filename_load = 'beta_parameters/HPV_parameters_age.mat'
    loaded_data = sio.loadmat(filename_load)
    
    beta_male = loaded_data['beta_male_age']
    beta_female = loaded_data['beta_female_age']
    
    C = 1 # constant used to tackle zeros
    ### x and y axis
    x = np.arange(age_start, age_end+1,1)
    y = np.arange(period_start, period_end+1,1)
    
    if len(x) > 1 and len(y) > 1:

        ## making x and y the same length
        if len(x) > len(y):
            diff = len(x) - len(y)
            rows = np.arange(period_start, period_end+diff+1,1)
            cols = x
            [X,Y] = np.meshgrid(cols,rows)
            
        elif len(x) < len(y):
            diff = len(y) - len(x)
            cols = np.arange(age_start, age_end+diff+1,1)
            rows = y
            [X,Y] = np.meshgrid(cols,rows)
            
        elif len(x) == len(y):
            rows = y
            cols = x
            [X,Y] = np.meshgrid(cols,rows)
                
    else:
        rows = y
        cols = x
        [X,Y] = np.meshgrid(cols,rows)
    
    #### Womens model
    # --------------------------------------------------------------------------------------------------------------------
    if gender == 'kvinne':
        order = 9   # polynomial order
        Xmatrix = rf.gen_def_matrix(X, Y, k=order)
        
        temp = np.exp( np.dot(Xmatrix,beta_female.T) ) - C # converting logarithmic prediction to exponential prediction
        
        ### case: both x and y are single values
        if len(temp) == 1:

            ## conditions
            # ------------------------------------------------------------------------
            temp = 0 if temp < 0 else temp         # converting negative to zero
            temp = 0 if x < 9 or x > 75 else temp # turning number of vaccinations to zero, for the age of < 13 and > 75
            temp = 0 if y < 2006 else temp
            # ------------------------------------------------------------------------
            
            ## single value prediction
            pred = temp*10
        
        ### case: either x or y is a vector 
        elif len(y) > 1 or len(x) > 1:
            
            temp[np.where(temp < 0)] = 0    # converting negatives to zeros
            if len(x) == 1:
            
                ## conditions
                # ------------------------------------------------------------------------
                temp[:] = 0 if x < 9 or x > 75 else temp # vaccinations set to zero, for the age of < 13 and > 75
                temp[np.where(y < 2006)] = 0              # turning past prediction (before 2006) to zero
                # ------------------------------------------------------------------------
            
                ## vector prediction
                pred = (temp.T*10).tolist()
                
            elif len(y) == 1:
                ## conditions
                # ------------------------------------------------------------------------
                temp[:] = 0 if y < 2006 else temp # turning past prediction (before 2006) to zero
                temp[np.where(x > 75)] = 0        # turning number of vaccinations to zero, after the age of 75 
                temp[np.where(x < 13)] = 0        # turning number of vaccinations to zero, before the age of 13
                # ------------------------------------------------------------------------
            
                ## vector prediction
                pred = (temp.T*10).tolist()
            
            else:
                ## reshaping results and keeping only the desired ones
                temp = temp.reshape((len(cols), len(rows)))
                temp = temp[0:len(y),0:len(x)]
            
                ## conditions
                # ------------------------------------------------------------------------
                temp[np.where(y < 2006), :] = 0  # turning past prediction (before 2006) to zero
                temp[:, np.where(x > 75)] = 0    # turning number of vaccinations to zero, after the age og 75 
                temp[:, np.where(x < 9)] = 0    # turning number of vaccinations to zero, before the age og 13 
                # ------------------------------------------------------------------------
                
                ## turning prediction to a series of lists, where each list represents a year
                pred = (temp*10).tolist()
    # --------------------------------------------------------------------------------------------------------------------
    
    #### Mens model
    # --------------------------------------------------------------------------------------------------------------------
    elif gender == 'menn':
        order = 11  # polynomial order
        Xmatrix = rf.gen_def_matrix(X, Y, k=order)
        
        temp = np.exp( np.dot(Xmatrix,beta_male.T) ) - C # converting logarithmic prediction to exponential prediction
            
        ### case: both x and y are single values
        if len(temp) == 1:
            
            ## conditions
            # ------------------------------------------------------------------------
            temp = 0 if temp < 0 else temp         # converting negative to zero
            temp = 0 if x < 13 or x > 75 else temp # turning number of vaccinations to zero, for the age of < 13 and > 75
            temp = 0 if y < 2007 else temp
            # ------------------------------------------------------------------------
            
            ## single value prediction
            pred = temp*10
        
        ### case: either x or y is a vector 
        elif len(y) > 1 or len(x) > 1:
            
            temp[np.where(temp < 0)] = 0    # converting negatives to zeros
            if len(x) == 1:
                
                ## conditions
                # ------------------------------------------------------------------------
                temp[:] = 0 if x < 13 or x > 75 else temp # vaccinations set to zero, for the age of < 13 and > 75
                temp[np.where(y < 2007)] = 0              # turning past prediction (before 2006) to zero
                # ------------------------------------------------------------------------
            
                ## vector prediction
                pred = (temp.T*10).tolist()
                
            elif len(y) == 1:
                
                ## conditions
                # ------------------------------------------------------------------------
                temp[:] = 0 if y < 2007 else temp # turning past prediction (before 2006) to zero
                temp[np.where(x > 75)] = 0        # turning number of vaccinations to zero, after the age of 75 
                temp[np.where(x < 13)] = 0        # turning number of vaccinations to zero, before the age of 13
                # ------------------------------------------------------------------------
            
                ## vector prediction
                pred = (temp.T*10).tolist()
            
            else:
                
                ## reshaping results and keeping only the desired ones
                temp = temp.reshape((len(cols), len(rows)))
                temp = temp[0:len(y),0:len(x)]
            
                ## conditions
                # ------------------------------------------------------------------------
                temp[np.where(y < 2007), :] = 0  # turning past prediction (before 2006) to zero
                temp[:, np.where(x > 75)] = 0    # turning number of vaccinations to zero, after the age og 75 
                temp[:, np.where(x < 13)] = 0    # turning number of vaccinations to zero, before the age og 13 
                # ------------------------------------------------------------------------
                
                ## turning prediction to a series of lists, where each list represents a year
                pred = (temp*10).tolist()
    # --------------------------------------------------------------------------------------------------------------------
    
    if data_grouping > 1:
        g = data_grouping
        K = int(np.floor(len(pred[0])/g))

        pred, new_axis = pp.grouping(pred, group = data_grouping)
        
        if len(x) == 1:
            y = np.array([(y[0] + i*g) for i in range(K) ])
        else:
            x = np.array(new_axis)
        
    ### Plotting the results
    if fast_plot:
        if len(y) > 1 or len(x) > 1:
            if len(x) == 1:
                mpf.plot(y, pred, titl=""+str(gender)+": Alder "+str(x[0])+"", Xlabel='År', Ylabel='Vaksiner', ltype = 'o-')
                
            elif len(y) == 1:
                mpf.plot(x, pred, titl=""+str(gender)+": År "+str(y[0])+"", Xlabel='Alder', Ylabel='Vaksiner', ltype = 'o-')
            
            else:
                mpf.surface_plot(Z = np.array(pred).T, Y=x, X=y, Xlabel='År', Ylabel='Alder', Zlabel=""+str(gender)+" Vaksiner")
                
    ## correcting axis
    years = y
    ages = x
    
    return pred, ages, years



def Cancer_model(age_start = 13, age_end = 75, period_start = 2004, period_end = 2017, cancer_type = '' , 
                 gender = 'menn', fast_plot = False, data_grouping = 0):
    '''
    This function gives the predicted incidence rate for a specific type of cancer at a particular "year" at a give "age". 
    To get a single value set the "start" and "end" parameters equal to each other. The models were fitted using data 
    before vaccination.
    
    age_...:     These parameters determine the start and end of the different ages we want to consider.
    period_...:  These parameters determine the start and end of the different years we want to consider.
    cancer_type: This parameter specifies the cancer type. Women's cancer Types; 'Livmorhals', 'Anus', 'Munn, andre',
                 'Livmorlegeme', 'Livmor, usesifisert'. Men's cancer Types; 'Anus', 'Munn, andre'. 
    gender:      This parameter specifies which model, one wants to use. The one fitted using mens vaccination data 
                 (gender = 'menn'), or the one fitted using womans vaccination data (gender = 'kvinne'). Must be a string.
    fast_plot:   To plot the result, one only needs to change from False to True. -> fast_plot = True
    data_grouping: This parameter decides how many point are grouped together. Meaning that if you have an age period of 
                   10 to 100 and a data_grouping = 3, then the data from age 10, 11 and 12 are added together, 
                   then 13, 14 and 15 etc ...
                   Nothing happens if data_grouping is equal to 0 or 1.
    
    Note: The incidence rate for ages under 10 and over 100 (since the number is small) is automatically set to zero.
    
    Return: This function returns three values; prediction, age and year in it given order (pred, ages , years). 
            Where the prediction can either be a single value, a vector or a list of vectors in which each list represents 
            a year.
    '''
    
    C = 1 # constant used to tackle zeros
    pred = 0
    ### x and y axis
    x = np.arange(age_start, age_end+1,1)
    y = np.arange(period_start, period_end+1,1)
    
    if len(x) > 1 and len(y) > 1:

        ## making x and y the same length
        if len(x) > len(y):
            diff = len(x) - len(y)
            rows = np.arange(period_start, period_end+diff+1,1)
            cols = x
            [X,Y] = np.meshgrid(cols,rows)
            
        elif len(x) < len(y):
            diff = len(y) - len(x)
            cols = np.arange(age_start, age_end+diff+1,1)
            rows = y
            [X,Y] = np.meshgrid(cols,rows)
            
        elif len(x) == len(y):
            rows = y
            cols = x
            [X,Y] = np.meshgrid(cols,rows)
                
    else:
        rows = y
        cols = x
        [X,Y] = np.meshgrid(cols,rows)
    
    #### Womens models
    # --------------------------------------------------------------------------------------------------------------------
    if gender == 'kvinne':
        filename_load = 'beta_parameters/Kcancer_parameters_before.mat'
        loaded_data = sio.loadmat(filename_load)
            
        # extracting beta parameters
        beta_b_LH = loaded_data['beta_before_Livsmorhalse']
        beta_b_A = loaded_data['beta_before_Anus']
        beta_b_MO = loaded_data['beta_before_Mouth']
        beta_b_LL = loaded_data['beta_before_Livmorlegeme']
        beta_b_L = loaded_data['beta_before_Livmor']
            
        if cancer_type == 'Livmorhals':
            order = 9 # polynomial order
            Xmatrix = rf.gen_def_matrix(X, Y, k=order)
                
            # exponential model
            temp = np.exp(np.dot(Xmatrix,beta_b_LH.T)) - C # converting logarithmic prediction to exponential prediction
            
            ### case: both x and y are single values
            if len(temp) == 1:
                ## conditions
                # ------------------------------------------------------------------------
                temp = 0 if temp < 0 else temp         # converting negative to zero
                temp = 0 if x < 10 or x > 100 else temp # turning number of vaccinations to zero, for the age of < 13 and > 94
                temp = 0 if y > 2110 else temp
                # ------------------------------------------------------------------------
            
                ## single value prediction
                pred = temp*0.9
        
            ### case: either x or y is a vector 
            elif len(y) > 1 or len(x) > 1:
                    
                temp[np.where(temp < 0)] = 0    # converting negatives to zeros
                if len(x) == 1:
                    
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[:] = 0 if x < 10 or x > 100 else temp # vaccinations set to zero, for the age of < 13 and > 94
                    temp[np.where(y > 2110)] = 0              # turning past prediction (before 2007) to zero
                    # ------------------------------------------------------------------------
            
                    ## vector prediction
                    pred = (temp.T*0.9).tolist()
                
                elif len(y) == 1:
                        
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[:] = 0 if y > 2110 else temp # turning past prediction (before 2007) to zero
                    temp[np.where(x > 100)] = 0        # turning number of vaccinations to zero, after the age of 94 
                    temp[np.where(x < 10)] = 0        # turning number of vaccinations to zero, before the age of 13
                    # ------------------------------------------------------------------------
            
                    ## vector prediction
                    pred = (temp.T*0.9).tolist()
            
                else:
                    ## reshaping results and keeping only the desired ones
                    temp = temp.reshape((len(cols), len(rows)))
                    temp = temp[0:len(y),0:len(x)]
            
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[np.where(y > 2110), :] = 0  # turning past prediction (before 2007) to zero
                    temp[:, np.where(x > 100)] = 0    # turning number of vaccinations to zero, after the age og 94 
                    temp[:, np.where(x < 10)] = 0    # turning number of vaccinations to zero, before the age og 13 
                    # ------------------------------------------------------------------------
                
                    ## turning prediction to a series of lists, where each list represents an age
                    pred = (temp*0.9).tolist()
            
            
        elif cancer_type == 'Anus':
            order = 9 # polynomial order
            Xmatrix = rf.gen_def_matrix(X, Y, k=order)
                
            # exponential model
            temp = np.exp(np.dot(Xmatrix,beta_b_A.T)) - C # converting logarithmic prediction to exponential prediction
            
            ### case: both x and y are single values
            if len(temp) == 1:
                ## conditions
                # ------------------------------------------------------------------------
                temp = 0 if temp < 0 else temp         # converting negative to zero
                temp = 0 if x < 10 or x > 100 else temp # turning number of vaccinations to zero, for the age of < 10 and > 100
                # ------------------------------------------------------------------------
            
                ## single value prediction
                pred = temp*0.5
        
            ### case: either x or y is a vector 
            elif len(y) > 1 or len(x) > 1:
                    
                temp[np.where(temp < 0)] = 0    # converting negatives to zeros
                if len(x) == 1:
                    
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[:] = 0 if x < 10 or x > 100 else temp # vaccinations set to zero, for the age of < 10 and > 100
                    # ------------------------------------------------------------------------
            
                    ## vector prediction
                    pred = (temp.T*0.5).tolist()
                
                elif len(y) == 1:
                        
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[np.where(x > 100)] = 0        # turning number of vaccinations to zero, after the age of 100 
                    temp[np.where(x < 10)] = 0        # turning number of vaccinations to zero, before the age of 10
                    # ------------------------------------------------------------------------
            
                    ## vector prediction
                    pred = (temp.T*0.5).tolist()
            
                else:
                    ## reshaping results and keeping only the desired ones
                    temp = temp.reshape((len(cols), len(rows)))
                    temp = temp[0:len(y),0:len(x)]
            
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[:, np.where(x > 100)] = 0    # turning number of vaccinations to zero, after the age og 100 
                    temp[:, np.where(x < 10)] = 0    # turning number of vaccinations to zero, before the age og 10
                    # ------------------------------------------------------------------------
                
                    ## turning prediction to a series of lists, where each list represents a year
                    pred = (temp*0.5).tolist()
        
        elif cancer_type == 'Munn, andre':
            order = 7 # polynomial order
            Xmatrix = rf.gen_def_matrix(X, Y, k=order)
                
            # exponential model
            temp = np.exp(np.dot(Xmatrix,beta_b_MO.T)) - C # converting logarithmic prediction to exponential prediction
            
            ### case: both x and y are single values
            if len(temp) == 1:
                ## conditions
                # ------------------------------------------------------------------------
                temp = 0 if temp < 0 else temp         # converting negative to zero
                temp = 0 if x < 10 or x > 100 else temp # turning number of vaccinations to zero, for the age of < 10 and > 100
                # ------------------------------------------------------------------------
            
                ## single value prediction
                pred = temp*0.6
        
            ### case: either x or y is a vector 
            elif len(y) > 1 or len(x) > 1:
                    
                temp[np.where(temp < 0)] = 0    # converting negatives to zeros
                if len(x) == 1:
                    
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[:] = 0 if x < 10 or x > 100 else temp # vaccinations set to zero, for the age of < 10 and > 100
                    # ------------------------------------------------------------------------
            
                    ## vector prediction
                    pred = (temp.T*0.6).tolist()
                
                elif len(y) == 1:
                        
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[np.where(x > 100)] = 0        # turning number of vaccinations to zero, after the age of 100 
                    temp[np.where(x < 10)] = 0        # turning number of vaccinations to zero, before the age of 10
                    # ------------------------------------------------------------------------
            
                    ## vector prediction
                    pred = (temp.T*0.6).tolist()
            
                else:
                    ## reshaping results and keeping only the desired ones
                    temp = temp.reshape((len(cols), len(rows)))
                    temp = temp[0:len(y),0:len(x)]
            
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[:, np.where(x > 100)] = 0    # turning number of vaccinations to zero, after the age og 100 
                    temp[:, np.where(x < 10)] = 0    # turning number of vaccinations to zero, before the age og 10
                    # ------------------------------------------------------------------------
                
                    ## turning prediction to a series of lists, where each list represents an age
                    pred = (temp*0.6).tolist()
            
        elif cancer_type == 'Livmorlegeme':
            order = 9 # polynomial order
            Xmatrix = rf.gen_def_matrix(X, Y, k=order)
                
            # exponential model
            temp = np.exp(np.dot(Xmatrix,beta_b_LL.T)) - C # converting logarithmic prediction to exponential prediction
            
            ### case: both x and y are single values
            if len(temp) == 1:
                ## conditions
                # ------------------------------------------------------------------------
                temp = 0 if temp < 0 else temp         # converting negative to zero
                temp = 0 if x < 10 or x > 100 else temp # turning number of vaccinations to zero, for the age of < 10 and > 100
                # ------------------------------------------------------------------------
            
                ## single value prediction
                pred = temp*1.5
        
            ### case: either x or y is a vector 
            elif len(y) > 1 or len(x) > 1:
                    
                temp[np.where(temp < 0)] = 0    # converting negatives to zeros
                if len(x) == 1:
                    
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[:] = 0 if x < 10 or x > 100 else temp # vaccinations set to zero, for the age of < 10 and > 100
                    # ------------------------------------------------------------------------
            
                    ## vector prediction
                    pred = (temp.T*1.5).tolist()
                
                elif len(y) == 1:
                        
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[np.where(x > 100)] = 0        # turning number of vaccinations to zero, after the age of 100 
                    temp[np.where(x < 10)] = 0        # turning number of vaccinations to zero, before the age of 10
                    # ------------------------------------------------------------------------
            
                    ## vector prediction
                    pred = (temp.T*1.5).tolist()
            
                else:
                    ## reshaping results and keeping only the desired ones
                    temp = temp.reshape((len(cols), len(rows)))
                    temp = temp[0:len(y),0:len(x)]
            
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[:, np.where(x > 100)] = 0    # turning number of vaccinations to zero, after the age og 100 
                    temp[:, np.where(x < 10)] = 0    # turning number of vaccinations to zero, before the age og 10
                    # ------------------------------------------------------------------------
                
                    ## turning prediction to a series of lists, where each list represents a year
                    pred = (temp*1.5).tolist()
            
        elif cancer_type == 'Livmor, usesifisert':
            order = 9 # polynomial order
            Xmatrix = rf.gen_def_matrix(X, Y, k=order)
                
            # exponential model
            temp = np.exp(np.dot(Xmatrix,beta_b_L.T)) - C # converting logarithmic prediction to exponential prediction
            
            ### case: both x and y are single values
            if len(temp) == 1:
                ## conditions
                # ------------------------------------------------------------------------
                temp = 0 if temp < 0 else temp         # converting negative to zero
                temp = 0 if x < 10 or x > 100 else temp # turning number of vaccinations to zero, for the age of < 10 and > 100
                # ------------------------------------------------------------------------
            
                ## single value prediction
                pred = temp*0.5
        
            ### case: either x or y is a vector 
            elif len(y) > 1 or len(x) > 1:
                    
                temp[np.where(temp < 0)] = 0    # converting negatives to zeros
                if len(x) == 1:
                    
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[:] = 0 if x < 10 or x > 100 else temp # vaccinations set to zero, for the age of < 10 and > 100
                    # ------------------------------------------------------------------------
            
                    ## vector prediction
                    pred = (temp.T*0.5).tolist()
                
                elif len(y) == 1:
                        
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[np.where(x > 100)] = 0        # turning number of vaccinations to zero, after the age of 100 
                    temp[np.where(x < 10)] = 0        # turning number of vaccinations to zero, before the age of 10
                    # ------------------------------------------------------------------------
            
                    ## vector prediction
                    pred = (temp.T*0.5).tolist()
            
                else:
                    ## reshaping results and keeping only the desired ones
                    temp = temp.reshape((len(cols), len(rows)))
                    temp = temp[0:len(y),0:len(x)]
            
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[:, np.where(x > 100)] = 0    # turning number of vaccinations to zero, after the age og 100 
                    temp[:, np.where(x < 10)] = 0    # turning number of vaccinations to zero, before the age og 10
                    # ------------------------------------------------------------------------
                
                    ## turning prediction to a series of lists, where each list represents a year
                    pred = (temp*0.5).tolist()
        
        else:
            print("Krefttype må spesifiseres, du kan velge mellom; 'Livmorhals', 'Anus', 'Munn, andre', 'Livmorlegeme,' 'Livmor, usesifisert'")
            pred = 0
            fast_plot = False
    
    # --------------------------------------------------------------------------------------------------------------------
    
    #### Mens models
    # --------------------------------------------------------------------------------------------------------------------
    elif gender == 'menn':
        filename_load = 'beta_parameters/Mcancer_parameters_before.mat'
        loaded_data = sio.loadmat(filename_load)
            
        # extracting beta parameters
        beta_b_A = loaded_data['beta_before_Anus']
        beta_b_MO = loaded_data['beta_before_Mouth']
        
        if cancer_type == 'Anus':
            order = 9  # polynomial order
            Xmatrix = rf.gen_def_matrix(X, Y, k=order)
                
            # exponential model
            temp = ( np.exp(np.dot(Xmatrix,beta_b_A.T)) - C )/10 # converting logarithmic prediction to exponential prediction
                
            ### case: both x and y are single values
            if len(temp) == 1:
                ## conditions
                # ------------------------------------------------------------------------
                temp = 0 if temp < 0 else temp         # converting negative to zero
                temp = 0 if x < 13 or x > 100 else temp # turning number of vaccinations to zero, for the age of < 13 and > 75
                # ------------------------------------------------------------------------
            
                ## single value prediction
                pred = temp
        
            ### case: either x or y is a vector 
            elif len(y) > 1 or len(x) > 1:
                    
                temp[np.where(temp < 0)] = 0    # converting negatives to zeros
                if len(x) == 1:
                    
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[:] = 0 if x < 13 or x > 100 else temp # vaccinations set to zero, for the age of < 13 and > 75
                    # ------------------------------------------------------------------------
            
                    ## vector prediction
                    pred = (temp.T).tolist()
                
                elif len(y) == 1:
                        
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[np.where(x > 100)] = 0        # turning number of vaccinations to zero, after the age of 75 
                    temp[np.where(x < 13)] = 0        # turning number of vaccinations to zero, before the age of 13
                    # ------------------------------------------------------------------------
            
                    ## vector prediction
                    pred = (temp.T).tolist()
            
                else:
                    ## reshaping results and keeping only the desired ones
                    temp = temp.reshape((len(cols), len(rows)))
                    temp = temp[0:len(y),0:len(x)]
            
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[:, np.where(x > 100)] = 0    # turning number of vaccinations to zero, after the age og 75 
                    temp[:, np.where(x < 13)] = 0    # turning number of vaccinations to zero, before the age og 13 
                    # ------------------------------------------------------------------------
                
                    ## turning prediction to a series of lists, where each list represents a year
                    pred = temp.tolist()
               
        elif cancer_type == 'Munn, andre':
            order = 7  # polynomial order
            Xmatrix = rf.gen_def_matrix(X, Y, k=order)
                
            # exponential model
            temp = ( np.exp(np.dot(Xmatrix,beta_b_MO.T)) - C ) # converting logarithmic prediction to exponential prediction
                
            ### case: both x and y are single values
            if len(temp) == 1:
                
                ## conditions
                # ------------------------------------------------------------------------
                temp = 0 if temp < 0 else temp         # converting negative to zero
                temp = 0 if x < 10 or x > 100 else temp # turning number of vaccinations to zero, for the age of < 13 and > 94
                # ------------------------------------------------------------------------
            
                ## single value prediction
                pred = temp
        
            ### case: either x or y is a vector 
            elif len(y) > 1 or len(x) > 1:
                    
                temp[np.where(temp < 0)] = 0    # converting negatives to zeros
                if len(x) == 1:
                        
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[:] = 0 if x < 10 or x > 100 else temp # vaccinations set to zero, for the age of < 13 and > 94            
                    # ------------------------------------------------------------------------
            
                    ## vector prediction
                    pred = (temp.T).tolist()
                
                elif len(y) == 1:
                        
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[np.where(x > 100)] = 0                    # turning number of vaccinations to zero, after the age of 94 
                    temp[np.where(x < 10)] = 0                    # turning number of vaccinations to zero, before the age of 13
                    # ------------------------------------------------------------------------
            
                    ## vector prediction
                    pred = (temp.T).tolist()
            
                else:
                    ## reshaping results and keeping only the desired ones
                    temp = temp.reshape((len(cols), len(rows)))
                    temp = temp[0:len(y),0:len(x)]
            
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[:, np.where(x > 100)] = 0    # turning number of vaccinations to zero, after the age og 94 
                    temp[:, np.where(x < 10)] = 0    # turning number of vaccinations to zero, before the age og 13 
                    # ------------------------------------------------------------------------
                
                    ## turning prediction to a series of lists, where each list represents a year
                    pred = temp.tolist()
        
        else:
            print("Krefttype må spesifiseres, du kan velge mellom; 'Anus', 'Munn, andre'")
            pred = 0
            fast_plot = False
            
    # --------------------------------------------------------------------------------------------------------------------
    
    if data_grouping > 1:
        g = data_grouping
        K = int(np.floor(len(pred[0])/g))

        pred, new_axis = pp.grouping(pred, group = data_grouping)
        
        if len(x) == 1:
            y = np.array([(y[0] + i*g) for i in range(K) ])
        else:
            x = np.array(new_axis)
    
    ### Plotting the results
    if fast_plot:
        if len(y) > 1 or len(x) > 1:
            if len(x) == 1:
                mpf.plot(y, pred, titl=""+str(gender)+" "+str(cancer_type)+": Alder "+str(x[0])+"", Xlabel='År',
                         Ylabel='Insidensrate', ltype = 'o-')
                
            elif len(y) == 1:
                mpf.plot(x, pred, titl=""+str(gender)+" "+str(cancer_type)+": År "+str(y[0])+"", Xlabel='Alder',
                         Ylabel='Insidensrate', ltype = 'o-')
            
            else:
                mpf.surface_plot(Z = np.array(pred).T, Y=x, X=y, Xlabel='År', Ylabel='Alder', 
                                 Zlabel=""+str(gender)+" Insidensrate")
    
    ## correcting axis
    years = y
    ages = x
    
    return pred, ages, years



def Future_model(age_start = 13, age_end = 75, period_start = 2004, period_end = 2017, cancer_type = '' , 
                 gender = 'menn', fast_plot = False, data_grouping = 0):
    '''
    This function gives the predicted incidence rate for a specific type of cancer at a particular "year" at a give "age". 
    To get a single value set the "start" and "end" parameters equal to each other. The models were fitted using data 
    during vaccination.
    
    age_...:       These parameters determine the start and end of the different ages we want to consider.
    period_...:    These parameters determine the start and end of the different years we want to consider.
    cancer_type:   This parameter specifies the cancer type. Women's cancer Types; 'Livmorhals', 'Anus', 'Munn, andre',
                   'Livmorlegeme', 'Livmor, usesifisert'. Men's cancer Types; 'Anus', 'Munn, andre'. 
    gender:        This parameter specifies which model, one wants to use. The one fitted using mens vaccination data 
                   (gender = 'menn'), or the one fitted using womans vaccination data (gender = 'kvinne'). Must be a string.
    fast_plot:     To plot the result, one only needs to change from False to True. -> fast_plot = True
    data_grouping: This parameter decides how many point are grouped together. Meaning that if you have an age period of 
                   10 to 100 and a data_grouping = 3, then the data from age 10, 11 and 12 are added together, 
                   then 13, 14 and 15 etc ...
                   Nothing happens if data_grouping is equal to 0 or 1.
    
    Note: The incidence rate for ages under 10 and over 100 (since the number is small) are automatically set to zero. 
          The same applies to the years before 2006 for women and 2007 for men.

    Return: This function returns three values; prediction, age and year in it given order (pred, ages , years). 
            Where the prediction can either be a single value, a vector or a list of vectors in which each list represents 
            a year.
    '''
    
    C = 1 # constant used to tackle zeros
    pred = 0       
    ### x and y axis
    x = np.arange(age_start, age_end+1,1)
    y = np.arange(period_start, period_end+1,1)
    
    if len(x) > 1 and len(y) > 1:

        ## making x and y the same length
        if len(x) > len(y):
            diff = len(x) - len(y)
            rows = np.arange(period_start, period_end+diff+1,1)
            cols = x
            [X,Y] = np.meshgrid(cols,rows)
            
        elif len(x) < len(y):
            diff = len(y) - len(x)
            cols = np.arange(age_start, age_end+diff+1,1)
            rows = y
            [X,Y] = np.meshgrid(cols,rows)
            
        elif len(x) == len(y):
            rows = y
            cols = x
            [X,Y] = np.meshgrid(cols,rows)
                
    else:
        rows = y
        cols = x
        [X,Y] = np.meshgrid(cols,rows)
    
    #### Womens models
    # --------------------------------------------------------------------------------------------------------------------    
    if gender == 'kvinne':
        filename_load = 'beta_parameters/Kcancer_parameters_during.mat'
        loaded_data = sio.loadmat(filename_load)
            
        # extracting beta parameters
        beta_d_LH = loaded_data['beta_during_Livsmorhalse']
        beta_d_A = loaded_data['beta_during_Anus']
        beta_d_MO = loaded_data['beta_during_Mouth']
        beta_d_LL = loaded_data['beta_during_Livmorlegeme']
        beta_d_L = loaded_data['beta_during_Livmor']
            
        if cancer_type == 'Livmorhals':
            order = 9 # polynomial order
            Xmatrix = rf.gen_def_matrix(X, Y, k=order)
                
            # exponential model
            temp = np.exp(np.dot(Xmatrix,beta_d_LH.T)) - C # converting logarithmic prediction to exponential prediction
                
            ### case: both x and y are single values
            if len(temp) == 1:
                ## conditions
                # ------------------------------------------------------------------------
                temp = 0 if temp < 0 else temp         # converting negative to zero
                temp = 0 if x < 10 or x > 100 else temp # turning number of vaccinations to zero, for the age of < 13 and > 94
                temp = 0 if y < 2006 else temp
                # ------------------------------------------------------------------------
            
                ## single value prediction
                pred = temp*0.9
        
            ### case: either x or y is a vector 
            elif len(y) > 1 or len(x) > 1:
                    
                temp[np.where(temp < 0)] = 0    # converting negatives to zeros
                if len(x) == 1:
                    
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[:] = 0 if x < 10 or x > 100 else temp # vaccinations set to zero, for the age of < 13 and > 94
                    temp[np.where(y < 2006)] = 0              # turning past prediction (before 2007) to zero
                    # ------------------------------------------------------------------------
            
                    ## vector prediction
                    pred = (temp.T*0.9).tolist()
                
                elif len(y) == 1:
                        
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[:] = 0 if y < 2006 else temp # turning past prediction (before 2007) to zero
                    temp[np.where(x > 100)] = 0        # turning number of vaccinations to zero, after the age of 94 
                    temp[np.where(x < 10)] = 0        # turning number of vaccinations to zero, before the age of 13
                    # ------------------------------------------------------------------------
            
                    ## vector prediction
                    pred = (temp.T*0.9).tolist()
            
                else:
                    ## reshaping results and keeping only the desired ones
                    temp = temp.reshape((len(cols), len(rows)))
                    temp = temp[0:len(y),0:len(x)]
            
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[np.where(y < 2006), :] = 0  # turning past prediction (before 2007) to zero
                    temp[:, np.where(x > 100)] = 0    # turning number of vaccinations to zero, after the age og 94 
                    temp[:, np.where(x < 10)] = 0    # turning number of vaccinations to zero, before the age og 13 
                    # ------------------------------------------------------------------------
                
                    ## turning prediction to a series of lists, where each list represents a year
                    pred = (temp*0.9).tolist()
                    
                
        elif cancer_type == 'Anus':
            order = 9 # polynomial order
            Xmatrix = rf.gen_def_matrix(X, Y, k=order)
                
            # exponential model
            temp = np.exp(np.dot(Xmatrix,beta_d_A.T)) - C # converting logarithmic prediction to exponential prediction
                
            ### case: both x and y are single values
            if len(temp) == 1:
                ## conditions
                # ------------------------------------------------------------------------
                temp = 0 if temp < 0 else temp         # converting negative to zero
                temp = 0 if x < 10 or x > 100 else temp # turning number of vaccinations to zero, for the age of < 13 and > 94
                temp = 0 if y < 2006 else temp
                # ------------------------------------------------------------------------
            
                ## single value prediction
                pred = temp*0.6
        
            ### case: either x or y is a vector 
            elif len(y) > 1 or len(x) > 1:
                    
                temp[np.where(temp < 0)] = 0    # converting negatives to zeros
                if len(x) == 1:
                    
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[:] = 0 if x < 10 or x > 100 else temp # vaccinations set to zero, for the age of < 13 and > 94
                    temp[np.where(y < 2006)] = 0              # turning past prediction (before 2007) to zero
                    # ------------------------------------------------------------------------
            
                    ## vector prediction
                    pred = (temp.T*0.6).tolist()
                
                elif len(y) == 1:
                        
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[:] = 0 if y < 2006 else temp # turning past prediction (before 2007) to zero
                    temp[np.where(x > 100)] = 0        # turning number of vaccinations to zero, after the age of 94 
                    temp[np.where(x < 10)] = 0        # turning number of vaccinations to zero, before the age of 13
                    # ------------------------------------------------------------------------
            
                    ## vector prediction
                    pred = (temp.T*0.6).tolist()
            
                else:
                    ## reshaping results and keeping only the desired ones
                    temp = temp.reshape((len(cols), len(rows)))
                    temp = temp[0:len(y),0:len(x)]
            
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[np.where(y < 2006), :] = 0  # turning past prediction (before 2007) to zero
                    temp[:, np.where(x > 100)] = 0    # turning number of vaccinations to zero, after the age og 94 
                    temp[:, np.where(x < 10)] = 0    # turning number of vaccinations to zero, before the age og 13 
                    # ------------------------------------------------------------------------
                
                    ## turning prediction to a series of lists, where each list represents a year
                    pred = (temp*0.6).tolist()
                
        elif cancer_type == 'Munn, andre':
            order = 9 # polynomial order
            Xmatrix = rf.gen_def_matrix(X, Y, k=order)
                
            # exponential model
            temp = np.exp(np.dot(Xmatrix,beta_d_MO.T)) - C # converting logarithmic prediction to exponential prediction
                
            ### case: both x and y are single values
            if len(temp) == 1:
                ## conditions
                # ------------------------------------------------------------------------
                temp = 0 if temp < 0 else temp         # converting negative to zero
                temp = 0 if x < 10 or x > 100 else temp # turning number of vaccinations to zero, for the age of < 13 and > 94
                temp = 0 if y < 2006 else temp
                # ------------------------------------------------------------------------
            
                ## single value prediction
                pred = temp*0.5
        
            ### case: either x or y is a vector 
            elif len(y) > 1 or len(x) > 1:
                    
                temp[np.where(temp < 0)] = 0    # converting negatives to zeros
                if len(x) == 1:
                    
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[:] = 0 if x < 10 or x > 100 else temp # vaccinations set to zero, for the age of < 13 and > 94
                    temp[np.where(y < 2006)] = 0              # turning past prediction (before 2007) to zero
                    # ------------------------------------------------------------------------
            
                    ## vector prediction
                    pred = (temp.T*0.5).tolist()
                
                elif len(y) == 1:
                        
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[:] = 0 if y < 2006 else temp # turning past prediction (before 2007) to zero
                    temp[np.where(x > 100)] = 0        # turning number of vaccinations to zero, after the age of 94 
                    temp[np.where(x < 10)] = 0        # turning number of vaccinations to zero, before the age of 13
                    # ------------------------------------------------------------------------
            
                    ## vector prediction
                    pred = (temp.T*0.5).tolist()
            
                else:
                    ## reshaping results and keeping only the desired ones
                    temp = temp.reshape((len(cols), len(rows)))
                    temp = temp[0:len(y),0:len(x)]
            
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[np.where(y < 2006), :] = 0  # turning past prediction (before 2007) to zero
                    temp[:, np.where(x > 100)] = 0    # turning number of vaccinations to zero, after the age og 94 
                    temp[:, np.where(x < 10)] = 0    # turning number of vaccinations to zero, before the age og 13 
                    # ------------------------------------------------------------------------
                
                    ## turning prediction to a series of lists, where each list represents a year
                    pred = (temp*0.5).tolist()
                
        elif cancer_type == 'Livmorlegeme':
            order = 5 # polynomial order
            Xmatrix = rf.gen_def_matrix(X, Y, k=order)
                
            # exponential model
            temp = np.exp(np.dot(Xmatrix,beta_d_LL.T)) - C # converting logarithmic prediction to exponential prediction
                
            ### case: both x and y are single values
            if len(temp) == 1:
                ## conditions
                # ------------------------------------------------------------------------
                temp = 0 if temp < 0 else temp         # converting negative to zero
                temp = 0 if x < 10 or x > 100 else temp # turning number of vaccinations to zero, for the age of < 13 and > 94
                temp = 0 if y < 2006 else temp
                # ------------------------------------------------------------------------
            
                ## single value prediction
                pred = temp*1.5
        
            ### case: either x or y is a vector 
            elif len(y) > 1 or len(x) > 1:
                    
                temp[np.where(temp < 0)] = 0    # converting negatives to zeros
                if len(x) == 1:
                    
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[:] = 0 if x < 10 or x > 100 else temp # vaccinations set to zero, for the age of < 13 and > 94
                    temp[np.where(y < 2006)] = 0              # turning past prediction (before 2007) to zero
                    # ------------------------------------------------------------------------
            
                    ## vector prediction
                    pred = (temp.T*1.5).tolist()
                
                elif len(y) == 1:
                        
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[:] = 0 if y < 2006 else temp # turning past prediction (before 2007) to zero
                    temp[np.where(x > 100)] = 0        # turning number of vaccinations to zero, after the age of 94 
                    temp[np.where(x < 10)] = 0        # turning number of vaccinations to zero, before the age of 13
                    # ------------------------------------------------------------------------
            
                    ## vector prediction
                    pred = (temp.T*1.5).tolist()
            
                else:
                    ## reshaping results and keeping only the desired ones
                    temp = temp.reshape((len(cols), len(rows)))
                    temp = temp[0:len(y),0:len(x)]
            
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[np.where(y < 2006), :] = 0  # turning past prediction (before 2007) to zero
                    temp[:, np.where(x > 100)] = 0    # turning number of vaccinations to zero, after the age og 94 
                    temp[:, np.where(x < 10)] = 0    # turning number of vaccinations to zero, before the age og 13 
                    # ------------------------------------------------------------------------
                
                    ## turning prediction to a series of lists, where each list represents a year
                    pred = (temp*1.5).tolist()
                
        elif cancer_type == 'Livmor, usesifisert':
            order = 9 # polynomial order
            Xmatrix = rf.gen_def_matrix(X, Y, k=order)
                
            # exponential model
            temp = np.exp(np.dot(Xmatrix,beta_d_L.T)) - C # converting logarithmic prediction to exponential prediction
                
            ### case: both x and y are single values
            if len(temp) == 1:
                ## conditions
                # ------------------------------------------------------------------------
                temp = 0 if temp < 0 else temp         # converting negative to zero
                temp = 0 if x < 10 or x > 100 else temp # turning number of vaccinations to zero, for the age of < 10 and > 100
                temp = 0 if y < 2006 else temp
                # ------------------------------------------------------------------------
            
                ## single value prediction
                pred = temp*0.5
        
            ### case: either x or y is a vector 
            elif len(y) > 1 or len(x) > 1:
                    
                temp[np.where(temp < 0)] = 0    # converting negatives to zeros
                if len(x) == 1:
                    
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[:] = 0 if x < 10 or x > 100 else temp # vaccinations set to zero, for the age of < 10 and > 100
                    temp[np.where(y < 2006)] = 0              # turning past prediction (before 2007) to zero
                    # ------------------------------------------------------------------------
            
                    ## vector prediction
                    pred = (temp.T*0.5).tolist()
                
                elif len(y) == 1:
                        
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[:] = 0 if y < 2006 else temp # turning past prediction (before 2007) to zero
                    temp[np.where(x > 100)] = 0        # turning number of vaccinations to zero, after the age of 100 
                    temp[np.where(x < 10)] = 0        # turning number of vaccinations to zero, before the age of 10
                    # ------------------------------------------------------------------------
            
                    ## vector prediction
                    pred = (temp.T*0.5).tolist()
            
                else:
                    ## reshaping results and keeping only the desired ones
                    temp = temp.reshape((len(cols), len(rows)))
                    temp = temp[0:len(y),0:len(x)]
            
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[np.where(y < 2006), :] = 0  # turning past prediction (before 2007) to zero
                    temp[:, np.where(x > 100)] = 0    # turning number of vaccinations to zero, after the age og 100 
                    temp[:, np.where(x < 10)] = 0    # turning number of vaccinations to zero, before the age og 10 
                    # ------------------------------------------------------------------------
                
                    ## turning prediction to a series of lists, where each list represents a year
                    pred = (temp*0.5).tolist()
        
            
        else:
            print("Krefttype må spesifiseres, du kan velge mellom; 'Livmorhals', 'Anus', 'Munn, andre', 'Livmorlegeme', 'Livmor, usesifisert'")
            pred = 0
            fast_plot = False
            
        
    # --------------------------------------------------------------------------------------------------------------------
                
    #### Mens models
    # --------------------------------------------------------------------------------------------------------------------
    elif gender == 'menn':
        filename_load = 'beta_parameters/Mcancer_parameters_during.mat'
        loaded_data = sio.loadmat(filename_load)
            
        # extracting beta parameters
        beta_d_A = loaded_data['beta_during_Anus']
        beta_d_MO = loaded_data['beta_during_Mouth']
        
        if cancer_type == 'Anus':
            order = 8  # polynomial order
            Xmatrix = rf.gen_def_matrix(X, Y, k=order)
                
            # exponential model
            temp = np.exp(np.dot(Xmatrix,beta_d_A.T)) - C  # converting logarithmic prediction to exponential prediction
                
            ### case: both x and y are single values
            if len(temp) == 1:
                ## conditions
                # ------------------------------------------------------------------------
                temp = 0 if temp < 0 else temp         # converting negative to zero
                temp = 0 if x < 15 or x > 100 else temp # turning number of vaccinations to zero, for the age of < 15 and > 100
                temp = 0 if y < 2007 else temp
                # ------------------------------------------------------------------------
            
                ## single value prediction
                pred = temp
        
            ### case: either x or y is a vector 
            elif len(y) > 1 or len(x) > 1:
                    
                temp[np.where(temp < 0)] = 0    # converting negatives to zeros
                if len(x) == 1:
                    
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[:] = 0 if x < 15 or x > 100 else temp # vaccinations set to zero, for the age of < 15 and > 100
                    temp[np.where(y < 2007)] = 0              # turning past prediction (before 2007) to zero
                    # ------------------------------------------------------------------------
            
                    ## vector prediction
                    pred = (temp.T).tolist()
                
                elif len(y) == 1:
                        
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[:] = 0 if y < 2007 else temp # turning past prediction (before 2007) to zero
                    temp[np.where(x > 100)] = 0        # turning number of vaccinations to zero, after the age of 100 
                    temp[np.where(x < 15)] = 0        # turning number of vaccinations to zero, before the age of 15
                    # ------------------------------------------------------------------------
            
                    ## vector prediction
                    pred = (temp.T).tolist()
            
                else:
                    ## reshaping results and keeping only the desired ones
                    temp = temp.reshape((len(cols), len(rows)))
                    temp = temp[0:len(y),0:len(x)]
            
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[np.where(y < 2007), :] = 0  # turning past prediction (before 2007) to zero
                    temp[:, np.where(x > 100)] = 0    # turning number of vaccinations to zero, after the age og 100 
                    temp[:, np.where(x < 15)] = 0    # turning number of vaccinations to zero, before the age og 15 
                    # ------------------------------------------------------------------------
                
                    ## turning prediction to a series of lists, where each list represents a year
                    pred = temp.tolist()
               
        elif cancer_type == 'Munn, andre':
            order = 6  # polynomial order
            Xmatrix = rf.gen_def_matrix(X, Y, k=order)
                
            # exponential model
            temp = np.exp(np.dot(Xmatrix,beta_d_MO.T)) - C # converting logarithmic prediction to exponential prediction
                
            ### case: both x and y are single values
            if len(temp) == 1:
                
                ## conditions
                # ------------------------------------------------------------------------
                temp = 0 if temp < 0 else temp         # converting negative to zero
                temp = 0 if x < 10 or x > 100 else temp # turning number of vaccinations to zero, for the age of < 10 and > 100
                temp = 0 if y < 2007 or y > 2080 else temp
                # ------------------------------------------------------------------------
            
                ## single value prediction
                pred = temp
        
            ### case: either x or y is a vector 
            elif len(y) > 1 or len(x) > 1:
                    
                temp[np.where(temp < 0)] = 0    # converting negatives to zeros
                if len(x) == 1:
                        
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[:] = 0 if x < 10 or x > 100 else temp # vaccinations set to zero, for the age of < 10 and > 100
                    temp[np.where(y < 2007)] = 0               # turning past prediction (before 2007) to zero
                    temp[np.where(y > 2080)] = 0              
                    # ------------------------------------------------------------------------
            
                    ## vector prediction
                    pred = (temp.T).tolist()
                
                elif len(y) == 1:
                        
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[:] = 0 if y < 2007 or y > 2080 else temp # turning past prediction (before 2007) to zero
                    temp[np.where(x > 100)] = 0                   # turning number of vaccinations to zero, after the age of 100
                    temp[np.where(x < 10)] = 0                    # turning number of vaccinations to zero, before the age of 10
                    # ------------------------------------------------------------------------
            
                    ## vector prediction
                    pred = (temp.T).tolist()
            
                else:
                    ## reshaping results and keeping only the desired ones
                    temp = temp.reshape((len(cols), len(rows)))
                    temp = temp[0:len(y),0:len(x)]
            
                    ## conditions
                    # ------------------------------------------------------------------------
                    temp[np.where(y < 2007), :] = 0  # turning past prediction (before 2007) to zero
                    temp[np.where(y > 2080), :] = 0
                    temp[:, np.where(x > 100)] = 0   # turning number of vaccinations to zero, after the age og 100 
                    temp[:, np.where(x < 10)] = 0    # turning number of vaccinations to zero, before the age og 10 
                    # ------------------------------------------------------------------------
                
                    ## turning prediction to a series of lists, where each list represents a year
                    pred = temp.tolist()
        
        else:
            print("Krefttype må spesifiseres, du kan velge mellom; 'Anus', 'Munn, andre'")
            pred = 0
            fast_plot = False
            
    # --------------------------------------------------------------------------------------------------------------------
    
    if data_grouping > 1:
        g = data_grouping
        K = int(np.floor(len(pred[0])/g))

        pred, new_axis = pp.grouping(pred, group = data_grouping)
        
        if len(x) == 1:
            y = np.array([(y[0] + i*g) for i in range(K) ])
        else:
            x = np.array(new_axis)
    
    ### Plotting the results
    if fast_plot:
        if len(y) > 1 or len(x) > 1:
            if len(x) == 1:
                mpf.plot(y, pred, titl=""+str(gender)+" "+str(cancer_type)+": Alder "+str(x[0])+"", Xlabel='År',
                         Ylabel='Insidensrate', ltype = 'o-')
                
            elif len(y) == 1:
                mpf.plot(x, pred, titl=""+str(gender)+" "+str(cancer_type)+": År "+str(y[0])+"", Xlabel='Alder',
                         Ylabel='Insidensrate', ltype = 'o-')
            
            else:
                mpf.surface_plot(Z = np.array(pred).T, Y=x, X=y, Xlabel='År', Ylabel='Alder', 
                                 Zlabel=""+str(gender)+" Insidensrate")
    
    ## correcting axis 
    years = y
    ages = x
    
    return pred, ages, years