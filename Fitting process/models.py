import numpy as np
import Regression_functions as rf
import scipy.io as sio


def HPV_model(x, y, gender = 'female'):
    filename_load = 'beta_parameters/HPV_parameters.mat'
    loaded_data = sio.loadmat(filename_load)
    
    beta_male = loaded_data['beta_male']
    beta_female = loaded_data['beta_female']
    
    # x and y axis
    rows = Y
    cols = X
    [X,Y] = np.meshgrid(cols,rows)
    
    C = 0.0001 # constant used to tackle zeros
    
    if gender == 'female':
        order = 9   # polynomial order
        Xmatrix = rf.gen_def_matrix(X, Y, k=order)
        
        pred = ( np.exp(Xmatrix.dot(beta_female)) - C )/5
    
    if gender == 'male':
        order = 11  # polynomial order
        Xmatrix = rf.gen_def_matrix(X, Y, k=order)
        
        pred = ( np.exp(Xmatrix.dot(beta_male)) - C )/5
        
    return pred

def C_inc(x, y, cancer_type = '' , gender = 'female'):
    
    
    
    
    return pred

def M(x, y, cancer_type = '' , gender = 'female')
    
    
    
    
    return pred