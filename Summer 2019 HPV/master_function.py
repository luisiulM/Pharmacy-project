import warnings
warnings.filterwarnings("ignore")

import os
import numpy as np
import models as m 
import myplot as mpf
import scipy.io as sio
import preprocessor as pp
from matplotlib.pyplot import *
import Regression_functions as rf

def master_data(period_start = 2004, period_end = 2017, gender = 'menn', data_type = 'vaksiner', 
                cancer_type = '', perspective = 'Region'):
    '''
    This function processes and returns the desired data, ready to use, in correspondence with the specified 'gender', 
    'data_type', 'cancer_type' and perspective.
    
    # Parameters:
    period_...:  These parameters determine the start and end of the different years we want to consider. However, since there
                 are no fitted models for region data, only know data can be plotted. Thus, when trying to plot outside the 
                 known period, a warning will appear intructing the avaliable years.
    gender:      Each gender has their own data set and models, hence gender must be defined. For men -> 'menn' (which is the
                 default), for women -> 'kvinne'.
    data_type:   Here you specify the data that you want to use; HPV vaccines -> 'vaksiner', cancer data -> 'kreft' or for plots 
                 that require both -> 'begge'.
                 It is not necessary to specify cancer_type, if using HPV vaccine data.
    cancer_type: This parameter specifies the cancer type. Women's cancer Types; 'Livmorhals', 'Anus', 'Munn, andre',
                 'Livmorlegeme', 'Livmor, usesifisert'. Men's cancer Types; 'Anus', 'Munn, andre'.
    perspective: This parameter specifies perspective in which you want to observe the data, either in terms of region
                 or age. Region -> 'Region', age -> 'Alder'
              
    return: This function returns four parameters, the first being the extracted/predicted data, the second being data 
            corresponding to the x-axis (usefull only for plotting, since it carries no meaning), the third denotes the 
            years y, the fourth the x-axis label (grouped ages if considering 'Alder' or regions if considering 'Region')
            xlabel and lastly we have legends (which is needed if plotting line or bar plot).
    '''
    y = np.arange(period_start, period_end+1,1)
    
    #----------------------------------------------------------------------------------------------------------------
    if gender == 'menn':
        if data_type == 'vaksiner':
            filename = [r'data_files/menn/reseptregisteret_menn_norge.xls', r'data_files/menn/reseptmennfylker.xls']
            
            if perspective == 'Alder':
                ### extracting known data
                data, xlabel = pp.Prescription_data(filename[0], case = 'Age', grouping = 10)
                
                x = np.linspace(0,8,9)
                
                ## making legends
                legends = []
                for i in y:
                    legends.append( 'År: '+str(i)+'' )
                
                ### redusing
                if period_start > 2003 and period_end < 2019:
                    data = data[(period_start-2004):(len(data) - (2018-period_end))]
                 
                ## adding later years
                if period_start > 2003 and period_end > 2018: 
                    
                    start = period_start
                    if period_start < 2019:
                        data = data[(period_start-2004):]
                        start = 2019
                    
                    # extra data
                    temp, *_ = m.HPV_model(age_start = 0, age_end = 100, period_start = start,
                                           period_end = period_end, gender = gender, data_grouping = 10)
                    temp = np.array(temp)
                    if len(np.shape(temp)) > 1:
                        extra = temp[:,0:9]
                        extra[:,8] = sum(temp[:,8:].T)

                    else:
                        extra = temp[0:9]
                        extra[8] = sum(temp[8:].T)
                    # turning array -> list
                    extra = extra.tolist()
                    # adding new years
                    if period_start < 2019:
                        data = data + extra
                    else:
                        data = extra
                    
                ## adding previous years
                if period_start < 2004 and period_end < 2019:
                    
                    end = period_end
                    if period_end > 2003:
                        data = data[:(len(data) - (2018-period_end))]
                        end = 2003
                    
                    # extra data
                    temp, *_ = m.HPV_model(age_start = 0, age_end = 100, period_start = period_start,
                                             period_end = end, gender = gender, data_grouping = 10)
                    temp = np.array(temp)
                    if len(np.shape(temp)) > 1:
                        extra = temp[:,0:9]
                        extra[:,8] = sum(temp[:,8:].T)

                    else:
                        extra = temp[0:9]
                        extra[8] = sum(temp[8:].T)
                    # turning array -> list
                    extra = extra.tolist()
                    # adding new years
                    if period_end > 2003:
                        data = extra + data
                    else:
                        data = extra
                
                ## adding previous and later years
                if period_start < 2004 and period_end > 2018:
                    start = period_start
                    end = period_end
                    
                    # previous extra data
                    temp, *_ = m.HPV_model(age_start = 0, age_end = 100, period_start = start,
                                           period_end = 2003, gender = gender, data_grouping = 10)
                    temp = np.array(temp)
                    if len(np.shape(temp)) > 1:
                        extra_prev = temp[:,0:9]
                        extra_prev[:,8] = sum(temp[:,8:].T)

                    else:
                        extra_prev = temp[0:9]
                        extra_prev[8] = sum(temp[8:].T)
                    # turning array -> list
                    extra_prev = extra_prev.tolist()
                    
                    # later extra data
                    temp, *_ = m.HPV_model(age_start = 0, age_end = 100, period_start = 2019,
                                           period_end = end, gender = gender, data_grouping = 10)
                    temp = np.array(temp)
                    if len(np.shape(temp)) > 1:
                        extra_late = temp[:,0:9]
                        extra_late[:,8] = sum(temp[:,8:].T)

                    else:
                        extra_late = temp[0:9]
                        extra_late[8] = sum(temp[8:].T)
                    # turning array -> list
                    extra_late = extra_late.tolist()
                    
                    # adding new years
                    data = extra_prev + data + extra_late
                
            elif perspective == 'Region':
                ### extracting known data
                data, xlabel = pp.Prescription_data(filename[1], case = 'Region', grouping = 10)
                
                x = np.arange(0,18,1)
                y = np.arange(2004, 2018+1,1)
                
                ## making legends
                legends = []
                for i in y:
                    legends.append( 'År: '+str(i)+'' )
                
                ### redusing
                if period_start > 2003 and period_end < 2019:
                    data = data[(period_start-2004):(len(data) - (2018-period_end))]
                    
                    y = np.arange(period_start, period_end+1,1)
                    ## making legends
                    legends = []
                    for i in y:
                        legends.append( 'År: '+str(i)+'' )
                 
                elif period_start < 2004 or period_end > 2018:
                    print('There is no prediction model for Region data, thus only known data is available. From 2004 to 2018.')

                    
        elif data_type == 'kreft':
            filename = [r'data_files/menn/kreft.menn.53-17..xlsx', r'data_files/menn/menn.kreft.fylker.71-17.xlsx']
            
            if perspective == 'Alder':
                num_years = 65
                
                x = np.arange(0,9,1)
                xlabel = ['00–09', '10–19', '20–29', '30–39', '40–49', '50–59', '60–69', '70–79', '80+']
                ## making legends
                legends = []
                for i in y:
                    legends.append( 'År: '+str(i)+'' )
                
                
                if cancer_type == 'Anus':
                    ### extracting known data
                    data = pp.excel_extract(filename[0], keyword = ['År','all'], num_parameter = num_years,
                                            case = 'Insidensrate', cancer_type = 'Anus')
                    
                    ### redusing
                    if period_start > 1952 and period_end < 2018:
                        data = data[(period_start-1953):(len(data) - (2017-period_end))]
                 
                    ## adding later years
                    if period_start > 1952 and period_end > 2017: 
                    
                        start = period_start
                        if period_start < 2018:
                            data = data[(period_start-1953):]
                            start = 2018
                    
                        # extra data
                        temp, *_ = m.Future_model(age_start = 0, age_end = 100, period_start = start, period_end = period_end,
                                                  cancer_type = 'Anus', gender = gender, data_grouping = 10)
                        temp = np.array(temp)
                        if len(np.shape(temp)) > 1:
                            extra = temp[:,0:9]
                            extra[:,8] = sum(temp[:,8:].T)

                        else:
                            extra = temp[0:9]
                            extra[8] = sum(temp[8:].T)
                        # turning array -> list
                        extra = extra.tolist()
                        # adding new years
                        if period_start < 2018:
                            data = data + extra
                        else:
                            data = extra
                    
                    ## adding previous years
                    if period_start < 1953 and period_end < 2018:
                    
                        end = period_end
                        if period_end > 1952:
                            data = data[:(len(data) - (2017-period_end))]
                            end = 1952
                    
                        # extra data
                        temp, *_ = m.Cancer_model(age_start = 0, age_end = 100, period_start = period_start, period_end = end,
                                                  cancer_type = 'Anus', gender = gender, data_grouping = 10)
                        temp = np.array(temp)
                        if len(np.shape(temp)) > 1:
                            extra = temp[:,0:9]
                            extra[:,8] = sum(temp[:,8:].T)

                        else:
                            extra = temp[0:9]
                            extra[8] = sum(temp[8:].T)
                        # turning array -> list
                        extra = extra.tolist()
                        # adding new years
                        if period_end > 1952:
                            data = extra + data
                        else:
                            data = extra
                            
                    ## adding previous and later years
                    if period_start < 1953 and period_end > 2017:
                        start = period_start
                        end = period_end
                    
                        # previous extra data
                        temp, *_ = m.Cancer_model(age_start = 0, age_end = 100, period_start = start, period_end = 1952,
                                                  cancer_type = 'Anus', gender = gender, data_grouping = 10)
                        temp = np.array(temp)
                        if len(np.shape(temp)) > 1:
                            extra_prev = temp[:,0:9]
                            extra_prev[:,8] = sum(temp[:,8:].T)

                        else:
                            extra_prev = temp[0:9]
                            extra_prev[8] = sum(temp[8:].T)
                        # turning array -> list
                        extra_prev = extra_prev.tolist()
                    
                        # later extra data
                        temp, *_ = m.Future_model(age_start = 0, age_end = 100, period_start = 2018, period_end = end,
                                                  cancer_type = 'Anus', gender = gender, data_grouping = 10)
                        temp = np.array(temp)
                        if len(np.shape(temp)) > 1:
                            extra_late = temp[:,0:9]
                            extra_late[:,8] = sum(temp[:,8:].T)

                        else:
                            extra_late = temp[0:9]
                            extra_late[8] = sum(temp[8:].T)
                        # turning array -> list
                        extra_late = extra_late.tolist()
                    
                        # adding new years
                        data = extra_prev + data + extra_late
                    
                    
                elif cancer_type == 'Munn, andre':
                    ### extracting known data
                    data = pp.excel_extract(filename[0], keyword = ['År','all'], num_parameter = num_years,
                                            case = 'Insidensrate', cancer_type = 'Munn, andre')
                    
                    ### redusing
                    if period_start > 1952 and period_end < 2018:
                        data = data[(period_start-1953):(len(data) - (2017-period_end))]
                 
                    ## adding later years
                    if period_start > 1952 and period_end > 2017: 
                    
                        start = period_start
                        if period_start < 2018:
                            data = data[(period_start-1953):]
                            start = 2018
                    
                        # extra data
                        temp, *_ = m.Future_model(age_start = 0, age_end = 100, period_start = start, period_end = period_end,
                                               cancer_type = 'Munn, andre',gender = gender, data_grouping = 10)
                        temp = np.array(temp)
                        if len(np.shape(temp)) > 1:
                            extra = temp[:,0:9]
                            extra[:,8] = sum(temp[:,8:].T)
    
                        else:
                            extra = temp[0:9]
                            extra[8] = sum(temp[8:].T)
                        # turning array -> list
                        extra = extra.tolist()
                        # adding new years
                        if period_start < 2018:
                            data = data + extra
                        else:
                            data = extra
                    
                    ## adding previous years
                    if period_start < 1953 and period_end < 2018:
                    
                        end = period_end
                        if period_end > 1952:
                            data = data[:(len(data) - (2017-period_end))]
                            end = 1952
                    
                        # extra data
                        temp, *_ = m.Cancer_model(age_start = 0, age_end = 100, period_start = period_start, period_end = end,
                                                  cancer_type = 'Munn, andre', gender = gender, data_grouping = 10)
                        temp = np.array(temp)
                        if len(np.shape(temp)) > 1:
                            extra = temp[:,0:9]
                            extra[:,8] = sum(temp[:,8:].T)

                        else:
                            extra = temp[0:9]
                            extra[8] = sum(temp[8:].T)
                        # turning array -> list
                        extra = extra.tolist()
                        # adding new years
                        if period_end > 1952:
                            data = extra + data
                        else:
                            data = extra
                            
                    if period_start < 1953 and period_end > 2017:
                        start = period_start
                        end = period_end
                    
                        # previous extra data
                        temp, *_ = m.Cancer_model(age_start = 0, age_end = 100, period_start = start, period_end = 1952,
                                                  cancer_type = 'Munn, andre', gender = gender, data_grouping = 10)
                        temp = np.array(temp)
                        if len(np.shape(temp)) > 1:
                            extra_prev = temp[:,0:9]
                            extra_prev[:,8] = sum(temp[:,8:].T)

                        else:
                            extra_prev = temp[0:9]
                            extra_prev[8] = sum(temp[8:].T)
                        # turning array -> list
                        extra_prev = extra_prev.tolist()
                    
                        # later extra data
                        temp, *_ = m.Future_model(age_start = 0, age_end = 100, period_start = 2018, period_end = end,
                                                  cancer_type = 'Munn, andre', gender = gender, data_grouping = 10)
                        temp = np.array(temp)
                        if len(np.shape(temp)) > 1:
                            extra_late = temp[:,0:9]
                            extra_late[:,8] = sum(temp[:,8:].T)

                        else:
                            extra_late = temp[0:9]
                            extra_late[8] = sum(temp[8:].T)
                        # turning array -> list
                        extra_late = extra_late.tolist()
                    
                        # adding new years
                        data = extra_prev + data + extra_late
                
            
            elif perspective == 'Region':
                num_years = 47
                
                y = np.arange(1971, 2017+1,1)
                x = np.arange(1,20,1)
                xlabel = ['Østfold', 'Akershus', 'Oslo', 'Hedmark', 'Oppland','Buskerud', 'Vestfold', 'Telemark', 
                          'Aust-Agder', 'Vest-Agder', 'Rogaland', 'Hordaland', 'Sogn og Fjordane', 
                          'Møre og Romsdal', 'Sør-Trøndelag', 'Nord-Trøndelag', 'Nordland', 'Troms', 'Finnmark']
                ## making legends
                legends = []
                for i in y:
                    legends.append( 'År: '+str(i)+'' )
                
                if cancer_type == 'Anus':
                    ### extracting known data
                    data = pp.excel_extract(filename[1], keyword = ['År','all'], num_parameter = num_years, 
                                            case = 'Insidensrate', cancer_type = 'Anus')
                    data = list(np.array(data)[:,1:])
                    
                    ### redusing
                    if period_start > 1970 and period_end < 2018:
                        data = data[(period_start-1971):(len(data) - (2017-period_end))]
                        
                        y = np.arange(period_start, period_end+1,1)
                        ## making legends
                        legends = []
                        for i in y:
                            legends.append( 'År: '+str(i)+'' )
                 
                    elif period_start < 1971 or period_end > 2017:
                        print('There is no prediction model for Region data, thus only known data is available. From 1971 to 2017.')
                    
                    
                elif cancer_type == 'Munn, andre':
                    ### extracting known data
                    data = pp.excel_extract(filename[1], keyword = ['År','all'], num_parameter = num_years, 
                                            case = 'Insidensrate', cancer_type = 'Munn, andre')
                    data = list(np.array(data)[:,1:])
                
                    ### redusing
                    if period_start > 1970 and period_end < 2018:
                        data = data[(period_start-1971):(len(data) - (2017-period_end))]
                        
                        y = np.arange(period_start, period_end+1,1)
                        ## making legends
                        legends = []
                        for i in y:
                            legends.append( 'År: '+str(i)+'' )
                 
                    elif period_start < 1971 or period_end > 2017:
                        print('There is no prediction model for Region data, thus only known data is available. From 1971 to 2017.')
    #----------------------------------------------------------------------------------------------------------------
    elif gender == 'kvinne':
        if data_type == 'vaksiner':
            filename = [r'data_files/kvinne/reseptregisteret_women_norge.xls', r'data_files/kvinne/reseptkvinnerfylker.xls']
            
            if perspective == 'Alder':
                ### extracting known data
                data, xlabel = pp.Prescription_data(filename[0], case = 'Age', grouping = 10)
                
                x = np.linspace(0,8,9)
                
                ## making legends
                legends = []
                for i in y:
                    legends.append( 'År: '+str(i)+'' )
                
                ### redusing
                if period_start > 2003 and period_end < 2019:
                    data = data[(period_start-2004):(len(data) - (2018-period_end))]
                 
                ## adding later years
                if period_start > 2003 and period_end > 2018: 
                    
                    start = period_start
                    if period_start < 2019:
                        data = data[(period_start-2004):]
                        start = 2019
                    
                    # extra data
                    temp, *_ = m.HPV_model(age_start = 0, age_end = 100, period_start = start,
                                           period_end = period_end, gender = gender, data_grouping = 10)
                    temp = np.array(temp)
                    if len(np.shape(temp)) > 1:
                        extra = temp[:,0:9]
                        extra[:,8] = sum(temp[:,8:].T)

                    else:
                        extra = temp[0:9]
                        extra[8] = sum(temp[8:].T)
                    # turning array -> list
                    extra = extra.tolist()
                    # adding new years
                    if period_start < 2019:
                        data = data + extra
                    else:
                        data = extra
                    
                ## adding previous years
                if period_start < 2004 and period_end < 2019:
                    
                    end = period_end
                    if period_end > 2003:
                        data = data[:(len(data) - (2018-period_end))]
                        end = 2003
                    
                    # extra data
                    temp, *_ = m.HPV_model(age_start = 0, age_end = 100, period_start = period_start,
                                             period_end = end, gender = gender, data_grouping = 10)
                    temp = np.array(temp)
                    if len(np.shape(temp)) > 1:
                        extra = temp[:,0:9]
                        extra[:,8] = sum(temp[:,8:].T)

                    else:
                        extra = temp[0:9]
                        extra[8] = sum(temp[8:].T)
                    # turning array -> list
                    extra = extra.tolist()
                    # adding new years
                    if period_end > 2003:
                        data = extra + data
                    else:
                        data = extra
                
                ## adding previous and later years
                if period_start < 2004 and period_end > 2018:
                    start = period_start
                    end = period_end
                    
                    # previous extra data
                    temp, *_ = m.HPV_model(age_start = 0, age_end = 100, period_start = start,
                                           period_end = 2003, gender = gender, data_grouping = 10)
                    temp = np.array(temp)
                    if len(np.shape(temp)) > 1:
                        extra_prev = temp[:,0:9]
                        extra_prev[:,8] = sum(temp[:,8:].T)

                    else:
                        extra_prev = temp[0:9]
                        extra_prev[8] = sum(temp[8:].T)
                    # turning array -> list
                    extra_prev = extra_prev.tolist()
                    
                    # later extra data
                    temp, *_ = m.HPV_model(age_start = 0, age_end = 100, period_start = 2019,
                                           period_end = end, gender = gender, data_grouping = 10)
                    temp = np.array(temp)
                    if len(np.shape(temp)) > 1:
                        extra_late = temp[:,0:9]
                        extra_late[:,8] = sum(temp[:,8:].T)

                    else:
                        extra_late = temp[0:9]
                        extra_late[8] = sum(temp[8:].T)
                    # turning array -> list
                    extra_late = extra_late.tolist()
                    
                    # adding new years
                    data = extra_prev + data + extra_late
                
            elif perspective == 'Region':
                ### extracting known data
                data, xlabel = pp.Prescription_data(filename[1], case = 'Region', grouping = 10)
                
                x = np.arange(0,18,1)
                y = np.arange(2004, 2018+1,1)
                
                ## making legends
                legends = []
                for i in y:
                    legends.append( 'År: '+str(i)+'' )
                
                ### redusing
                if period_start > 2003 and period_end < 2019:
                    data = data[(period_start-2004):(len(data) - (2018-period_end))]
                    
                    y = np.arange(period_start, period_end+1,1)
                    ## making legends
                    legends = []
                    for i in y:
                        legends.append( 'År: '+str(i)+'' )
                 
                elif period_start < 2004 or period_end > 2018:
                    print('There is no prediction model for Region data, thus only known data is available. From 2004 to 2018.')

                    
        elif data_type == 'kreft':
            filename = [r'data_files/kvinne/kvinner.kreft.53-17.xlsx', r'data_files/kvinne/kvinner.kreft.fylker.71-17.xlsx']
            
            if perspective == 'Alder':
                num_years = 65
                
                x = np.arange(0,9,1)
                xlabel = ['00–09', '10–19', '20–29', '30–39', '40–49', '50–59', '60–69', '70–79', '80+']
                ## making legends
                legends = []
                for i in y:
                    legends.append( 'År: '+str(i)+'' )
                
                
                if cancer_type == 'Anus':
                    ### extracting known data
                    data = pp.excel_extract(filename[0], keyword = ['År','all'], num_parameter = num_years,
                                            case = 'Insidensrate', cancer_type = 'Anus')
                    
                    ### redusing
                    if period_start > 1952 and period_end < 2018:
                        data = data[(period_start-1953):(len(data) - (2017-period_end))]
                 
                    ## adding later years
                    if period_start > 1952 and period_end > 2017: 
                    
                        start = period_start
                        if period_start < 2018:
                            data = data[(period_start-1953):]
                            start = 2018
                    
                        # extra data
                        temp, *_ = m.Future_model(age_start = 0, age_end = 100, period_start = start, period_end = period_end,
                                                  cancer_type = 'Anus', gender = gender, data_grouping = 10)
                        temp = np.array(temp)
                        if len(np.shape(temp)) > 1:
                            extra = temp[:,0:9]
                            extra[:,8] = sum(temp[:,8:].T)

                        else:
                            extra = temp[0:9]
                            extra[8] = sum(temp[8:].T)
                        # turning array -> list
                        extra = extra.tolist()
                        # adding new years
                        if period_start < 2018:
                            data = data + extra
                        else:
                            data = extra
                    
                    ## adding previous years
                    if period_start < 1953 and period_end < 2018:
                    
                        end = period_end
                        if period_end > 1952:
                            data = data[:(len(data) - (2017-period_end))]
                            end = 1952
                    
                        # extra data
                        temp, *_ = m.Cancer_model(age_start = 0, age_end = 100, period_start = period_start, period_end = end,
                                                  cancer_type = 'Anus', gender = gender, data_grouping = 10)
                        temp = np.array(temp)
                        if len(np.shape(temp)) > 1:
                            extra = temp[:,0:9]
                            extra[:,8] = sum(temp[:,8:].T)

                        else:
                            extra = temp[0:9]
                            extra[8] = sum(temp[8:].T)
                        # turning array -> list
                        extra = extra.tolist()
                        # adding new years
                        if period_end > 1952:
                            data = extra + data
                        else:
                            data = extra
                            
                    ## adding previous and later years
                    if period_start < 1953 and period_end > 2017:
                        start = period_start
                        end = period_end
                    
                        # previous extra data
                        temp, *_ = m.Cancer_model(age_start = 0, age_end = 100, period_start = start, period_end = 1952,
                                                  cancer_type = 'Anus', gender = gender, data_grouping = 10)
                        temp = np.array(temp)
                        if len(np.shape(temp)) > 1:
                            extra_prev = temp[:,0:9]
                            extra_prev[:,8] = sum(temp[:,8:].T)

                        else:
                            extra_prev = temp[0:9]
                            extra_prev[8] = sum(temp[8:].T)
                        # turning array -> list
                        extra_prev = extra_prev.tolist()
                    
                        # later extra data
                        temp, *_ = m.Future_model(age_start = 0, age_end = 100, period_start = 2018, period_end = end,
                                                  cancer_type = 'Anus', gender = gender, data_grouping = 10)
                        temp = np.array(temp)
                        if len(np.shape(temp)) > 1:
                            extra_late = temp[:,0:9]
                            extra_late[:,8] = sum(temp[:,8:].T)

                        else:
                            extra_late = temp[0:9]
                            extra_late[8] = sum(temp[8:].T)
                        # turning array -> list
                        extra_late = extra_late.tolist()
                    
                        # adding new years
                        data = extra_prev + data + extra_late
                    
                    
                elif cancer_type == 'Munn, andre':
                    ### extracting known data
                    data = pp.excel_extract(filename[0], keyword = ['År','all'], num_parameter = num_years,
                                            case = 'Insidensrate', cancer_type = 'Munn, andre')
                    
                    ### redusing
                    if period_start > 1952 and period_end < 2018:
                        data = data[(period_start-1953):(len(data) - (2017-period_end))]
                 
                    ## adding later years
                    if period_start > 1952 and period_end > 2017: 
                    
                        start = period_start
                        if period_start < 2018:
                            data = data[(period_start-1953):]
                            start = 2018
                    
                        # extra data
                        temp, *_ = m.Future_model(age_start = 0, age_end = 100, period_start = start, period_end = period_end,
                                               cancer_type = 'Munn, andre',gender = gender, data_grouping = 10)
                        temp = np.array(temp)
                        if len(np.shape(temp)) > 1:
                            extra = temp[:,0:9]
                            extra[:,8] = sum(temp[:,8:].T)

                        else:
                            extra = temp[0:9]
                            extra[8] = sum(temp[8:].T)
                        # turning array -> list
                        extra = extra.tolist()
                        # adding new years
                        if period_start < 2018:
                            data = data + extra
                        else:
                            data = extra
                    
                    ## adding previous years
                    if period_start < 1953 and period_end < 2018:
                    
                        end = period_end
                        if period_end > 1952:
                            data = data[:(len(data) - (2017-period_end))]
                            end = 1952
                    
                        # extra data
                        temp, *_ = m.Cancer_model(age_start = 0, age_end = 100, period_start = period_start, period_end = end,
                                                  cancer_type = 'Munn, andre', gender = gender, data_grouping = 10)
                        temp = np.array(temp)
                        if len(np.shape(temp)) > 1:
                            extra = temp[:,0:9]
                            extra[:,8] = sum(temp[:,8:].T)

                        else:
                            extra = temp[0:9]
                            extra[8] = sum(temp[8:].T)
                        # turning array -> list
                        extra = extra.tolist()
                        # adding new years
                        if period_end > 1952:
                            data = extra + data
                        else:
                            data = extra
                            
                    if period_start < 1953 and period_end > 2017:
                        start = period_start
                        end = period_end
                    
                        # previous extra data
                        temp, *_ = m.Cancer_model(age_start = 0, age_end = 100, period_start = start, period_end = 1952,
                                                  cancer_type = 'Munn, andre', gender = gender, data_grouping = 10)
                        temp = np.array(temp)
                        if len(np.shape(temp)) > 1:
                            extra_prev = temp[:,0:9]
                            extra_prev[:,8] = sum(temp[:,8:].T)

                        else:
                            extra_prev = temp[0:9]
                            extra_prev[8] = sum(temp[8:].T)
                        # turning array -> list
                        extra_prev = extra_prev.tolist()
                    
                        # later extra data
                        temp, *_ = m.Future_model(age_start = 0, age_end = 100, period_start = 2018, period_end = end,
                                                  cancer_type = 'Munn, andre', gender = gender, data_grouping = 10)
                        temp = np.array(temp)
                        if len(np.shape(temp)) > 1:
                            extra_late = temp[:,0:9]
                            extra_late[:,8] = sum(temp[:,8:].T)

                        else:
                            extra_late = temp[0:9]
                            extra_late[8] = sum(temp[8:].T)
                        # turning array -> list
                        extra_late = extra_late.tolist()
                    
                        # adding new years
                        data = extra_prev + data + extra_late
                
                else:
                    ### extracting known data
                    data = pp.excel_extract(filename[0], keyword = ['År','all'], num_parameter = num_years,
                                            case = 'Insidensrate', cancer_type = cancer_type)
                    
                    ### redusing
                    if period_start > 1952 and period_end < 2018:
                        data = data[(period_start-1953):(len(data) - (2017-period_end))]
                 
                    ## adding later years
                    if period_start > 1952 and period_end > 2017: 
                    
                        start = period_start
                        if period_start < 2018:
                            data = data[(period_start-1953):]
                            start = 2018
                    
                        # extra data
                        temp, *_ = m.Future_model(age_start = 0, age_end = 100, period_start = start, period_end = period_end,
                                                  cancer_type = cancer_type, gender = gender, data_grouping = 10)
                        temp = np.array(temp)
                        if len(np.shape(temp)) > 1:
                            extra = temp[:,0:9]
                            extra[:,8] = sum(temp[:,8:].T)

                        else:
                            extra = temp[0:9]
                            extra[8] = sum(temp[8:].T)
                        # turning array -> list
                        extra = extra.tolist()
                        # adding new years
                        if period_start < 2018:
                            data = data + extra
                        else:
                            data = extra
                    
                    ## adding previous years
                    if period_start < 1953 and period_end < 2018:
                    
                        end = period_end
                        if period_end > 1952:
                            data = data[:(len(data) - (2017-period_end))]
                            end = 1952
                    
                        # extra data
                        temp, *_ = m.Cancer_model(age_start = 0, age_end = 100, period_start = period_start, period_end = end,
                                                  cancer_type = cancer_type, gender = gender, data_grouping = 10)
                        temp = np.array(temp)
                        if len(np.shape(temp)) > 1:
                            extra = temp[:,0:9]
                            extra[:,8] = sum(temp[:,8:].T)

                        else:
                            extra = temp[0:9]
                            extra[8] = sum(temp[8:].T)
                        # turning array -> list
                        extra = extra.tolist()
                        # adding new years
                        if period_end > 1952:
                            data = extra + data
                        else:
                            data = extra
                            
                    ## adding previous and later years
                    if period_start < 1953 and period_end > 2017:
                        start = period_start
                        end = period_end
                    
                        # previous extra data
                        temp, *_ = m.Cancer_model(age_start = 0, age_end = 100, period_start = start, period_end = 1952,
                                                  cancer_type = cancer_type, gender = gender, data_grouping = 10)
                        temp = np.array(temp)
                        if len(np.shape(temp)) > 1:
                            extra_prev = temp[:,0:9]
                            extra_prev[:,8] = sum(temp[:,8:].T)

                        else:
                            extra_prev = temp[0:9]
                            extra_prev[8] = sum(temp[8:].T)
                        # turning array -> list
                        extra_prev = extra_prev.tolist()
                    
                        # later extra data
                        temp, *_ = m.Future_model(age_start = 0, age_end = 100, period_start = 2018, period_end = end,
                                                  cancer_type = cancer_type, gender = gender, data_grouping = 10)
                        temp = np.array(temp)
                        if len(np.shape(temp)) > 1:
                            extra_late = temp[:,0:9]
                            extra_late[:,8] = sum(temp[:,8:].T)

                        else:
                            extra_late = temp[0:9]
                            extra_late[8] = sum(temp[8:].T)
                        # turning array -> list
                        extra_late = extra_late.tolist()
                    
                        # adding new years
                        data = extra_prev + data + extra_late
                        
            
            elif perspective == 'Region':
                num_years = 47
                
                y = np.arange(1971, 2017+1,1)
                x = np.arange(1,20,1)
                xlabel = ['Østfold', 'Akershus', 'Oslo', 'Hedmark', 'Oppland','Buskerud', 'Vestfold', 'Telemark', 
                          'Aust-Agder', 'Vest-Agder', 'Rogaland', 'Hordaland', 'Sogn og Fjordane', 
                          'Møre og Romsdal', 'Sør-Trøndelag', 'Nord-Trøndelag', 'Nordland', 'Troms', 'Finnmark']
                ## making legends
                legends = []
                for i in y:
                    legends.append( 'År: '+str(i)+'' )
                
                if cancer_type == 'Anus':
                    ### extracting known data
                    data = pp.excel_extract(filename[1], keyword = ['År','all'], num_parameter = num_years, 
                                            case = 'Insidensrate', cancer_type = 'Anus')
                    data = list(np.array(data)[:,1:])
                    
                    ### redusing
                    if period_start > 1970 and period_end < 2018:
                        data = data[(period_start-1971):(len(data) - (2017-period_end))]
                        
                        y = np.arange(period_start, period_end+1,1)
                        ## making legends
                        legends = []
                        for i in y:
                            legends.append( 'År: '+str(i)+'' )
                 
                    elif period_start < 1971 or period_end > 2017:
                        print('There is no prediction model for Region data, thus only known data is available. From 1971 to 2017.')
                    
                    
                elif cancer_type == 'Munn, andre':
                    ### extracting known data
                    data = pp.excel_extract(filename[1], keyword = ['År','all'], num_parameter = num_years, 
                                            case = 'Insidensrate', cancer_type = 'Munn, andre')
                    data = list(np.array(data)[:,1:])
                
                    ### redusing
                    if period_start > 1970 and period_end < 2018:
                        data = data[(period_start-1971):(len(data) - (2017-period_end))]
                        
                        y = np.arange(period_start, period_end+1,1)
                        ## making legends
                        legends = []
                        for i in y:
                            legends.append( 'År: '+str(i)+'' )
                 
                    elif period_start < 1971 or period_end > 2017:
                        print('There is no prediction model for Region data, thus only known data is available. From 1971 to 2017.')
                else:
                    ### extracting known data
                    data = pp.excel_extract(filename[1], keyword = ['År','all'], num_parameter = num_years, 
                                            case = 'Insidensrate', cancer_type = cancer_type)
                    data = list(np.array(data)[:,1:])
                    
                    ### redusing
                    if period_start > 1970 and period_end < 2018:
                        data = data[(period_start-1971):(len(data) - (2017-period_end))]
                        
                        y = np.arange(period_start, period_end+1,1)
                        ## making legends
                        legends = []
                        for i in y:
                            legends.append( 'År: '+str(i)+'' )
                 
                    elif period_start < 1971 or period_end > 2017:
                        print('There is no prediction model for Region data, thus only known data is available. From 1971 to 2017.')
    return data, x, y, xlabel, legends


def master_plot(period_start = 2004, period_end = 2017, gender = 'menn', data_type = 'vaksiner', 
                cancer_type = '', perspective = 'Region', plot_type = '3d', save = False):
    '''
    This function plots a desired plot, chosen by specifying the plot_type, in regards to data_type. Depending on the plot, you
    may also need to specify cancer_type while perpective and gender must always be defined.
    
    
    # Parameters:
    period_...:  These parameters determine the start and end of the different years we want to consider. However, since there
                 are no fitted models for region data, only know data can be plotted. Thus, when trying to plot outside the 
                 known period, a warning will appear intructing the avaliable years.
    gender:      Each gender has their own data set and models, hence gender must be defined. For men -> 'menn' (which is the
                 default), for women -> 'kvinne'.
    data_type:   Here you specify the data that you want to use; HPV vaccines -> 'vaksiner', cancer data -> 'kreft' or for plots 
                 that require both -> 'begge'.
                 It is not necessary to specify cancer_type, if using HPV vaccine data.
    cancer_type: This parameter specifies the cancer type. Women's cancer Types; 'Livmorhals', 'Anus', 'Munn, andre',
                 'Livmorlegeme', 'Livmor, usesifisert'. Men's cancer Types; 'Anus', 'Munn, andre'.
    perspective: This parameter specifies perspective in which you want to observe the data, either in terms of region
                 or age. Region -> 'Region', age -> 'Alder'
    plot_type:   Here one specifies the type of plot, one wants.
                 vaksiner; 'line', 'bar', '3d', 'contour'
                 kreft; 'line', 'bar', '3d', 'contour', 'comparison', 'comparison per year'
                 begge; 'bar line', 'bar line per year', 'scatter', 'scatter per year'
              
    return: This function returns the figure and axes, so that one may personalize the figure to their taste. (figure, axes)
    '''
    
    if gender == 'menn':
        #-----------------------------------------------------------------------
        if data_type == 'vaksiner':
            #-----------------------------------------------------------------------
            if perspective == 'Alder':
                ### collecting data
                data, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                           gender = 'menn', data_type = 'vaksiner', perspective = 'Alder')
                #-----------------------------------------------------------------------
                if plot_type == 'line':
                    ## plotting
                    fig, ax = mpf.plot(X = x, Y= data, Xlabel='Alder', Ylabel='HPV vaksiner', ltype = 'o-',
                                       titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                    ax.set_xticklabels([0]+xlabels)
                    ax.legend(legends)
                #-----------------------------------------------------------------------
                elif plot_type == 'bar':
                    ## plotting
                    fig, ax = mpf.barplot(X = xlabels, Y= data, Xlabel='Alder', Ylabel='HPV vaksiner', width = 0.8,
                                          titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                    ax.legend(legends)
                #-----------------------------------------------------------------------    
                elif plot_type == '3d':
                    ## plotting
                    fig, ax = mpf.surface_plot(Z = np.array(data).T, Y=x, X=y, Xlabel='År', Ylabel='Alder', 
                                               Zlabel="HPV vaksiner",
                                               titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                    ax.set_yticklabels(xlabels)
                #-----------------------------------------------------------------------    
                elif plot_type == 'contour':
                    ## plotting
                    fig, ax = mpf.contour3d_plot(Z = np.array(data).T, Y=x, X=y, Xlabel='År', Ylabel='Alder', 
                                                 Zlabel="HPV vaksiner",
                                                 titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                    ax.set_yticklabels(xlabels)
            #-----------------------------------------------------------------------        
            elif perspective == 'Region':
                ### collecting data
                data, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                           gender = 'menn', data_type = 'vaksiner', 
                                                           perspective = 'Region')
                #-----------------------------------------------------------------------
                if plot_type == 'line':
                    ## plotting
                    fig, ax = mpf.plot(X = x, Y = data, Xlabel='Region', Ylabel='HPV vaksiner', ltype = 'o-',
                                       titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                    locator=MaxNLocator(nbins=20)
                    ax.xaxis.set_major_locator(locator)
                    ax.set_xticklabels(['0']+list(xlabels), rotation=45)
                    ax.legend(legends)
                #-----------------------------------------------------------------------
                elif plot_type == 'bar':
                    ## plotting
                    fig, ax = mpf.barplot(X = xlabels, Y= data, Xlabel='Region', Ylabel='HPV vaksiner', width = 2,
                                          titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                    ax.legend(legends)
                #-----------------------------------------------------------------------
                elif plot_type == '3d':
                    ## plotting
                    fig, ax = mpf.surface_plot(Z = np.array(data).T, Y=x, X=y, Ylabel='Region', 
                                               Xlabel='År', Zlabel="HPV vaksiner",
                                               titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")

                    for i, label in enumerate(xlabels):
                        print('Index '+str(i)+' corresponds to '+str(label)+' region.')
                
                
                elif plot_type == 'contour':
                    ## plotting
                    fig, ax = mpf.contour3d_plot(Z = np.array(data).T, Y=x, X=y, Ylabel='Region', 
                                                 Xlabel='År', Zlabel="HPV vaksiner",
                                                 titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")

                    for i, label in enumerate(xlabels):
                        print('Index '+str(i)+' corresponds to '+str(label)+' region.')
                    
        #-----------------------------------------------------------------------
        elif data_type == 'kreft':
            #-----------------------------------------------------------------------
            if perspective == 'Alder':
                #-----------------------------------------------------------------------
                if plot_type == 'comparison':
                    if period_start != period_end:
                        print("Please select one year, if you want to plot a 'comparison' plot.")
                    
                    c_label = ['Anus','Munn, andre']
                    
                    ### collecting data
                    data_A, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                                 gender = gender, data_type = 'kreft', cancer_type = 'Anus', 
                                                                 perspective = 'Alder')
                    data_MO, *_ = master_data(period_start = period_start, period_end = period_end, 
                                              gender = gender, data_type = 'kreft', cancer_type = 'Munn, andre', 
                                              perspective = 'Alder')
                    
                    ## plotting
                    fig, ax = mpf.plot(X = x, Y = data_A, Xlabel='Alder', Ylabel='Insidensrate', ltype = 'o-',
                                       titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+" år "+str(y)+"")
                    ax.plot(x, np.array(data_MO).T, 'o-')
                    
                    locator=MaxNLocator(nbins=9)
                    ax.xaxis.set_major_locator(locator)
                    ax.set_xticklabels(['0'] + list(xlabels))
                    ax.legend(c_label)
                #-----------------------------------------------------------------------    
                elif plot_type == 'comparison per year':
                    c_label = ['Anus','Munn, andre']
                    
                    ### collecting data
                    data_A, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                                 gender = gender, data_type = 'kreft', cancer_type = 'Anus', 
                                                                 perspective = 'Alder')
                    data_MO, *_ = master_data(period_start = period_start, period_end = period_end, 
                                                 gender = gender, data_type = 'kreft', cancer_type = 'Munn, andre', 
                                                 perspective = 'Alder')
                    ## processing data
                    temp1 = np.zeros(len(data_A))
                    temp2 = np.zeros(len(data_MO))
                    
                    for iage in range(len(data_A[0])):
                        for iyear in range(len(data_A)):
                            temp1[iyear] = temp1[iyear] + data_A[iyear][iage]
                            temp2[iyear] = temp2[iyear] + data_MO[iyear][iage]

                    total_data_A = temp1
                    total_data_MO = temp2
                    
                    ## plotting 
                    fig, ax = mpf.plot(X = y, Y = total_data_A, Xlabel='År', Ylabel='Insidensrate', ltype = 'o-',
                                       titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                    ax.plot(y, total_data_MO, 'o-')
                    ax.legend(c_label)
                #-----------------------------------------------------------------------    
                elif cancer_type == 'Anus':
                    ### collecting data
                    data, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                               gender = gender, data_type = 'kreft', cancer_type = 'Anus', 
                                                               perspective = 'Alder')
                    if plot_type == 'line':
                        ## plotting
                        fig, ax = mpf.plot(X = x, Y= data, Xlabel='Alder', Ylabel='Insidensrate', ltype = 'o-',
                                           titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.set_xticklabels([0]+xlabels)
                        ax.legend(legends)
                
                    elif plot_type == 'bar':
                        ## plotting
                        fig, ax = mpf.barplot(X = xlabels, Y= data, Xlabel='Alder', Ylabel='Insidensrate', width = 0.8,
                                              titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.legend(legends)
                    
                    elif plot_type == '3d':
                        ## plotting
                        fig, ax = mpf.surface_plot(Z = np.array(data).T, Y=x, X=y, Xlabel='År', Ylabel='Alder', 
                                                   Zlabel='Insidensrate',
                                                   titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.set_yticklabels(xlabels)
                    
                    elif plot_type == 'contour':
                        ## plotting
                        fig, ax = mpf.contour3d_plot(Z = np.array(data).T, Y=x, X=y, Xlabel='År', Ylabel='Alder', 
                                                     Zlabel='Insidensrate',
                                                     titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.set_yticklabels(xlabels)
                #-----------------------------------------------------------------------
                elif cancer_type == 'Munn, andre':
                    ### collecting data
                    data, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                               gender = gender, data_type = 'kreft', 
                                                               cancer_type = 'Munn, andre', perspective = 'Alder')
                    if plot_type == 'line':
                        ## plotting
                        fig, ax = mpf.plot(X = x, Y= data, Xlabel='Alder', Ylabel='Insidensrate', ltype = 'o-',
                                           titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.set_xticklabels([0]+xlabels)
                        ax.legend(legends)
                
                    elif plot_type == 'bar':
                        ## plotting
                        fig, ax = mpf.barplot(X = xlabels, Y= data, Xlabel='Alder', Ylabel='Insidensrate', width = 0.8,
                                              titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.legend(legends)
                    
                    elif plot_type == '3d':
                        ## plotting
                        fig, ax = mpf.surface_plot(Z = np.array(data).T, Y=x, X=y, Xlabel='År', Ylabel='Alder', 
                                                   Zlabel='Insidensrate',
                                                   titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.set_yticklabels(xlabels)
                    
                    elif plot_type == 'contour':
                        ## plotting
                        fig, ax = mpf.contour3d_plot(Z = np.array(data).T, Y=x, X=y, Xlabel='År', Ylabel='Alder', 
                                                     Zlabel='Insidensrate',
                                                     titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.set_yticklabels(xlabels)
            #-----------------------------------------------------------------------
            elif perspective == 'Region':
                #-----------------------------------------------------------------------
                if plot_type == 'comparison':
                    if period_start != period_end:
                        print("Please select one year, if you want to plot a 'comparison' plot.")
                    
                    c_label = ['Anus','Munn, andre']
                    
                    ### collecting data
                    data_A, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                                 gender = gender, data_type = 'kreft', cancer_type = 'Anus', 
                                                                 perspective = 'Region')
                    data_MO, *_ = master_data(period_start = period_start, period_end = period_end, 
                                              gender = gender, data_type = 'kreft', cancer_type = 'Munn, andre', 
                                              perspective = 'Region')
                    
                    ## plotting
                    fig, ax = mpf.plot(X = x, Y = data_A, Xlabel='Region', Ylabel='Insidensrate', ltype = 'o-',
                                       titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                    ax.plot(x, np.array(data_MO).T, 'o-')
                    
                    locator=MaxNLocator(nbins=20)
                    ax.xaxis.set_major_locator(locator)
                    ax.set_xticklabels(['0'] + list(xlabels), rotation=45)
                    ax.legend(c_label)
                #-----------------------------------------------------------------------    
                elif plot_type == 'comparison per year':
                    c_label = ['Anus','Munn, andre']
                    
                    ### collecting data
                    data_A, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                                 gender = gender, data_type = 'kreft', cancer_type = 'Anus', 
                                                                 perspective = 'Region')
                    data_MO, *_ = master_data(period_start = period_start, period_end = period_end, 
                                              gender = gender, data_type = 'kreft', cancer_type = 'Munn, andre', 
                                              perspective = 'Region')
                    ## processing data
                    temp1 = np.zeros(len(data_A))
                    temp2 = np.zeros(len(data_MO))
                    
                    for iage in range(len(data_A[0])):
                        for iyear in range(len(data_A)):
                            temp1[iyear] = temp1[iyear] + data_A[iyear][iage]
                            temp2[iyear] = temp2[iyear] + data_MO[iyear][iage]

                    total_data_A = temp1
                    total_data_MO = temp2
                    
                    ## plotting 
                    fig, ax = mpf.plot(X = y, Y = total_data_A, Xlabel='År', Ylabel='Insidensrate', ltype = 'o-',
                                       titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                    ax.plot(y, total_data_MO, 'o-')
                    ax.legend(c_label)
                #-----------------------------------------------------------------------
                elif cancer_type == 'Anus':
                    ### collecting data
                    data, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                               gender = gender, data_type = 'kreft', cancer_type = 'Anus', 
                                                               perspective = 'Region')
                    if plot_type == 'line':
                        ## plotting
                        fig, ax = mpf.plot(X = x, Y = data, Xlabel='Region', Ylabel='Insidensrate', ltype = 'o-', 
                                           titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        locator=MaxNLocator(nbins=20)
                        ax.xaxis.set_major_locator(locator)
                        ax.set_xticklabels(['0']+list(xlabels), rotation=45)
                        ax.legend(legends)
            
                    elif plot_type == 'bar':
                        ## plotting
                        fig, ax = mpf.barplot(X = xlabels, Y= data, Xlabel='Region', Ylabel='Insidensrate', width = 2,
                                              titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.legend(legends)
                    
                    elif plot_type == '3d':
                        ## plotting
                        fig, ax = mpf.surface_plot(Z = np.array(data).T, Y=x, X=y, Ylabel='Region', 
                                                   Xlabel='År', Zlabel="Insidensrate",
                                                   titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")

                        for i, label in enumerate(xlabels):
                            print('Index '+str(i)+' corresponds to '+str(label)+' region.')
                
                
                    elif plot_type == 'contour':
                        ## plotting
                        fig, ax = mpf.contour3d_plot(Z = np.array(data).T, Y=x, X=y, Ylabel='Region', 
                                                     Xlabel='År', Zlabel="Insidensrate",
                                                     titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")

                        for i, label in enumerate(xlabels):
                            print('Index '+str(i)+' corresponds to '+str(label)+' region.')
                #-----------------------------------------------------------------------
                elif cancer_type == 'Munn, andre':
                    ### collecting data
                    data, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                               gender = gender, data_type = 'kreft', 
                                                               cancer_type = 'Munn, andre', perspective = 'Region')
                    if plot_type == 'line':
                        ## plotting
                        fig, ax = mpf.plot(X = x, Y = data, Xlabel='Region', Ylabel='Insidensrate', ltype = 'o-',
                                           titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        locator=MaxNLocator(nbins=20)
                        ax.xaxis.set_major_locator(locator)
                        ax.set_xticklabels(['0']+list(xlabels), rotation=45)
                        ax.legend(legends)
            
                    elif plot_type == 'bar':
                        ## plotting
                        fig, ax = mpf.barplot(X = xlabels, Y= data, Xlabel='Region', Ylabel='Insidensrate', width = 2,
                                              titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.legend(legends)
                    
                    elif plot_type == '3d':
                        ## plotting
                        fig, ax = mpf.surface_plot(Z = np.array(data).T, Y=x, X=y, Ylabel='Region', 
                                                   Xlabel='År', Zlabel="Insidensrate",
                                                   titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")

                        for i, label in enumerate(xlabels):
                            print('Index '+str(i)+' corresponds to '+str(label)+' region.')
                
                
                    elif plot_type == 'contour':
                        ## plotting
                        fig, ax = mpf.contour3d_plot(Z = np.array(data).T, Y=x, X=y, Ylabel='Region', 
                                                     Xlabel='År', Zlabel="Insidensrate",
                                                     titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")

                        for i, label in enumerate(xlabels):
                            print('Index '+str(i)+' corresponds to '+str(label)+' region.')
            
            
        #-----------------------------------------------------------------------
        elif data_type == 'begge':
            #-----------------------------------------------------------------------
            if perspective == 'Alder':
                ### collecting data
                data_vac, *_ = master_data(period_start = period_start, period_end = period_end, 
                                           gender = 'menn', data_type = 'vaksiner', perspective = 'Alder')
                #-----------------------------------------------------------------------
                if cancer_type == 'Anus':
                    ### collecting data
                    data, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                               gender = gender, data_type = 'kreft', cancer_type = 'Anus', 
                                                               perspective = 'Alder')
                    #-----------------------------------------------------------------------
                    if plot_type == 'bar line':
                        ## plotting
                        fig, ax = mpf.barplot(X = xlabels, Y= data, Xlabel='Alder', Ylabel='HPV vaksiner', width = 0.8,
                                              titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.legend(legends)
                        ax2 = ax.twinx()
                        for i in range(len(data_vac)):
                            ax2.plot(x, data_vac[i], 'o--')
                        ax2.set_ylabel('HPV vaksiner (lines)')
                        
                    #-----------------------------------------------------------------------
                    elif plot_type == 'bar line per year':
                        ## processing data
                        temp1 = np.zeros(len(data_vac))
                        temp2 = np.zeros(len(data))
                    
                        for iage in range(len(data_vac[0])):
                            for iyear in range(len(data_vac)):
                                temp1[iyear] = temp1[iyear] + data_vac[iyear][iage]
                                temp2[iyear] = temp2[iyear] + data[iyear][iage]

                        total_data_vac = temp1
                        total_data = temp2
                        
                        ## plotting 
                        fig, ax = mpf.barplot(X = y, Y= total_data, Xlabel='Alder', Ylabel='Insidensrate', width = 0,
                                              titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax2 = ax.twinx()
                        ax2.plot(y, total_data_vac, 'ro-')
                        ax2.set_ylabel('HPV vaksiner (lines)')
                    #-----------------------------------------------------------------------
                    elif plot_type == 'scatter':
                        ## plotting
                        fig, ax = mpf.scatterplot(data_vac, data, Xlabel='HPV vaksiner', 
                                                  Ylabel=''+str(cancer_type)+' '+str(perspective)+' Insidensrate', 
                                                  titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.legend(legends)
                        
                        
                    #-----------------------------------------------------------------------
                    elif plot_type =='scatter per year':
                        ## precessing data
                        ## processing data
                        temp1 = np.zeros(len(data_vac))
                        temp2 = np.zeros(len(data))
                    
                        for iage in range(len(data_vac[0])):
                            for iyear in range(len(data_vac)):
                                temp1[iyear] = temp1[iyear] + data_vac[iyear][iage]
                                temp2[iyear] = temp2[iyear] + data[iyear][iage]

                        total_data_vac = temp1
                        total_data = temp2
                        
                        ## plotting
                        fig, ax = mpf.scatterplot(total_data_vac, total_data, Xlabel='HPV vaksiner', 
                                                  Ylabel=''+str(cancer_type)+' '+str(perspective)+' Insidensrate', 
                                                  titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        
                #-----------------------------------------------------------------------     
                elif cancer_type == 'Munn, andre':
                    data, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                               gender = gender, data_type = 'kreft', 
                                                               cancer_type = 'Munn, andre', perspective = 'Alder')
                    #-----------------------------------------------------------------------
                    if plot_type == 'bar line':
                        ## plotting
                        fig, ax = mpf.barplot(X = xlabels, Y= data, Xlabel='Alder', Ylabel='HPV vaksiner', width = 0.8,
                                              titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.legend(legends)
                        ax2 = ax.twinx()
                        for i in range(len(data_vac)):
                            ax2.plot(x, data_vac[i], 'o--')
                        ax2.set_ylabel('HPV vaksiner (lines)')
                        
                    #-----------------------------------------------------------------------
                    elif plot_type == 'bar line per year':
                        ## processing data
                        temp1 = np.zeros(len(data_vac))
                        temp2 = np.zeros(len(data))
                    
                        for iage in range(len(data_vac[0])):
                            for iyear in range(len(data_vac)):
                                temp1[iyear] = temp1[iyear] + data_vac[iyear][iage]
                                temp2[iyear] = temp2[iyear] + data[iyear][iage]

                        total_data_vac = temp1
                        total_data = temp2
                        
                        ## plotting 
                        fig, ax = mpf.barplot(X = y, Y= total_data, Xlabel='Alder', Ylabel='Insidensrate', width = 0.8,
                                              titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax2 = ax.twinx()
                        ax2.plot(y, total_data_vac, 'ro-')
                        ax2.set_ylabel('HPV vaksiner (lines)')
                    #-----------------------------------------------------------------------
                    elif plot_type == 'scatter':
                        ## plotting
                        fig, ax = mpf.scatterplot(data_vac, data, Xlabel='HPV vaksiner', 
                                                  Ylabel=''+str(cancer_type)+' '+str(perspective)+' Insidensrate', 
                                                  titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.legend(legends)
                    #-----------------------------------------------------------------------
                    elif plot_type =='scatter per year':
                        ## precessing data
                        ## processing data
                        temp1 = np.zeros(len(data_vac))
                        temp2 = np.zeros(len(data))
                    
                        for iage in range(len(data_vac[0])):
                            for iyear in range(len(data_vac)):
                                temp1[iyear] = temp1[iyear] + data_vac[iyear][iage]
                                temp2[iyear] = temp2[iyear] + data[iyear][iage]

                        total_data_vac = temp1
                        total_data = temp2
                        
                        ## plotting
                        fig, ax = mpf.scatterplot(total_data_vac, total_data, Xlabel='HPV vaksiner', 
                                                  Ylabel=''+str(cancer_type)+' '+str(perspective)+' Insidensrate', 
                                                  titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        
            #-----------------------------------------------------------------------
            elif perspective == 'Region':
                ### collecting data
                data_vac, x, y, xlabels_vac, legends = master_data(period_start = period_start, period_end = period_end,
                                                                   gender = 'menn', data_type = 'vaksiner', 
                                                                   perspective = 'Region')
                #-----------------------------------------------------------------------
                if cancer_type == 'Anus':
                    ### collecting data
                    data, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                               gender = gender, data_type = 'kreft', 
                                                               cancer_type = 'Anus', perspective = 'Region')
                    ### aligning cancer and vaccine data
                    ordered_data = []
                    ordered_v = []
                    ordered_xlabels = []
                    ## year loop
                    for i in range(len(legends)):
                        temp = []
                        temp_v = []
    
                        ## cancer loop
                        for j in range(len(xlabels)):
                            ## vaccine loop
                            for k in range(len(xlabels_vac)):
                                # checking when they are equal
                                if xlabels[j] == xlabels_vac[k]:
                                    temp.append( data[i][j] )
                                    temp_v.append( data_vac[i][k] )
                                    ordered_xlabels.append( xlabels[j] )
    
                        ordered_data.append(temp)
                        ordered_v.append(temp_v)
                    
                    data = ordered_data
                    data_vac = ordered_v
                    xlabels = ordered_xlabels[0:17]
                    x = np.arange(0,len(xlabels))
                    #-----------------------------------------------------------------------
                    if plot_type == 'bar line':
                        ## plotting
                        fig, ax = mpf.barplot(X = xlabels, Y = data, Xlabel='Region', Ylabel='HPV vaksiner', width = 2,
                                              titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.legend(legends)
                        ax2 = ax.twinx()
                        for i in range(len(data_vac)):
                            ax2.plot(x, data_vac[i], 'o--')
                        ax2.set_ylabel('HPV vaksiner (lines)')
                        
                    #-----------------------------------------------------------------------
                    elif plot_type == 'bar line per year':
                        ## processing data
                        temp1 = np.zeros(len(data_vac))
                        temp2 = np.zeros(len(data))
                    
                        for iregion in range(len(data_vac[0])):
                            for iyear in range(len(data_vac)):
                                temp1[iyear] = temp1[iyear] + data_vac[iyear][iregion]
                                temp2[iyear] = temp2[iyear] + data[iyear][iregion]

                        total_data_vac = temp1
                        total_data = temp2
                        
                        ## plotting 
                        fig, ax = mpf.barplot(X = y, Y= total_data, Xlabel='Region', Ylabel='Insidensrate', width = 2,
                                              titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax2 = ax.twinx()
                        ax2.plot(y, total_data_vac, 'ro-')
                        ax2.set_ylabel('HPV vaksiner (lines)')
                    #-----------------------------------------------------------------------
                    elif plot_type == 'scatter':
                        ## plotting
                        fig, ax = mpf.scatterplot(data_vac, data, Xlabel='HPV vaksiner', 
                                                  Ylabel=''+str(cancer_type)+' '+str(perspective)+' Insidensrate', 
                                                  titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.legend(legends)
                    #-----------------------------------------------------------------------
                    elif plot_type =='scatter per year':
                        ## precessing data
                        temp1 = np.zeros(len(data_vac))
                        temp2 = np.zeros(len(data))
                    
                        for iage in range(len(data_vac[0])):
                            for iyear in range(len(data_vac)):
                                temp1[iyear] = temp1[iyear] + data_vac[iyear][iage]
                                temp2[iyear] = temp2[iyear] + data[iyear][iage]

                        total_data_vac = temp1
                        total_data = temp2
                        
                        ## plotting
                        fig, ax = mpf.scatterplot(total_data_vac, total_data, Xlabel='HPV vaksiner', 
                                                  Ylabel=''+str(cancer_type)+' '+str(perspective)+' Insidensrate', 
                                                  titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                #-----------------------------------------------------------------------
                elif cancer_type == 'Munn, andre':
                    ### collecting data
                    data, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                               gender = gender, data_type = 'kreft', 
                                                               cancer_type = 'Munn, andre', perspective = 'Region')
                    
                    ### aligning cancer and vaccine data
                    ordered_data = []
                    ordered_v = []
                    ordered_xlabels = []
                    ## year loop
                    for i in range(len(legends)):
                        temp = []
                        temp_v = []
    
                        ## cancer loop
                        for j in range(len(xlabels)):
                            ## vaccine loop
                            for k in range(len(xlabels_vac)):
                                # checking when they are equal
                                if xlabels[j] == xlabels_vac[k]:
                                    temp.append( data[i][j] )
                                    temp_v.append( data_vac[i][k] )
                                    ordered_xlabels.append( xlabels[j] )
    
                        ordered_data.append(temp)
                        ordered_v.append(temp_v)
                    
                    data = ordered_data
                    data_vac = ordered_v
                    xlabels = ordered_xlabels[0:17]
                    x = np.arange(0,len(xlabels))
                    #-----------------------------------------------------------------------
                    if plot_type == 'bar line':
                        ## plotting
                        fig, ax = mpf.barplot(X = xlabels, Y= data, Xlabel='Region', Ylabel='HPV vaksiner', width = 2,
                                              titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.legend(legends)
                        ax2 = ax.twinx()
                        for i in range(len(data_vac)):
                            ax2.plot(x, data_vac[i], 'o--')
                        ax2.set_ylabel('HPV vaksiner (lines)')
                        
                    #-----------------------------------------------------------------------
                    elif plot_type == 'bar line per year':
                        ## processing data
                        temp1 = np.zeros(len(data_vac))
                        temp2 = np.zeros(len(data))
                    
                        for iregion in range(len(data_vac[0])):
                            for iyear in range(len(data_vac)):
                                temp1[iyear] = temp1[iyear] + data_vac[iyear][iregion]
                                temp2[iyear] = temp2[iyear] + data[iyear][iregion]

                        total_data_vac = temp1
                        total_data = temp2
                        
                        ## plotting 
                        fig, ax = mpf.barplot(X = y, Y= total_data, Xlabel='Region', Ylabel='Insidensrate', width = 2,
                                              titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax2 = ax.twinx()
                        ax2.plot(y, total_data_vac, 'ro-')
                        ax2.set_ylabel('HPV vaksiner (lines)')
                    #-----------------------------------------------------------------------
                    elif plot_type == 'scatter':
                        ## plotting
                        fig, ax = mpf.scatterplot(data_vac, data, Xlabel='HPV vaksiner', 
                                                  Ylabel=''+str(cancer_type)+' '+str(perspective)+' Insidensrate', 
                                                  titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.legend(legends)
                    #-----------------------------------------------------------------------
                    elif plot_type =='scatter per year':
                        ## precessing data
                        temp1 = np.zeros(len(data_vac))
                        temp2 = np.zeros(len(data))
                    
                        for iage in range(len(data_vac[0])):
                            for iyear in range(len(data_vac)):
                                temp1[iyear] = temp1[iyear] + data_vac[iyear][iage]
                                temp2[iyear] = temp2[iyear] + data[iyear][iage]

                        total_data_vac = temp1
                        total_data = temp2
                        
                        ## plotting
                        fig, ax = mpf.scatterplot(total_data_vac, total_data, Xlabel='HPV vaksiner', 
                                                  Ylabel=''+str(cancer_type)+' '+str(perspective)+' Insidensrate', 
                                                  titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
    #-----------------------------------------------------------------------
    elif gender == 'kvinne':
        #-----------------------------------------------------------------------
        if data_type == 'vaksiner':
            #-----------------------------------------------------------------------
            if perspective == 'Alder':
                ### collecting data
                data, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                           gender = gender, data_type = 'vaksiner', perspective = 'Alder')
                #-----------------------------------------------------------------------
                if plot_type == 'line':
                    ## plotting
                    fig, ax = mpf.plot(X = x, Y= data, Xlabel='Alder', Ylabel='HPV vaksiner', ltype = 'o-',
                                       titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                    ax.set_xticklabels([0]+xlabels)
                    ax.legend(legends)
                #-----------------------------------------------------------------------
                elif plot_type == 'bar':
                    ## plotting
                    fig, ax = mpf.barplot(X = xlabels, Y= data, Xlabel='Alder', Ylabel='HPV vaksiner', width = 0.8,
                                          titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                    ax.legend(legends)
                #-----------------------------------------------------------------------    
                elif plot_type == '3d':
                    ## plotting
                    fig, ax = mpf.surface_plot(Z = np.array(data).T, Y=x, X=y, Xlabel='År', Ylabel='Alder', 
                                               Zlabel="HPV vaksiner",
                                               titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                    ax.set_yticklabels(xlabels)
                #-----------------------------------------------------------------------    
                elif plot_type == 'contour':
                    ## plotting
                    fig, ax = mpf.contour3d_plot(Z = np.array(data).T, Y=x, X=y, Xlabel='År', Ylabel='Alder', 
                                                 Zlabel="HPV vaksiner",
                                                 titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                    ax.set_yticklabels(xlabels)
            #-----------------------------------------------------------------------        
            elif perspective == 'Region':
                ### collecting data
                data, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                           gender = gender, data_type = 'vaksiner', 
                                                           perspective = 'Region')
                #-----------------------------------------------------------------------
                if plot_type == 'line':
                    ## plotting
                    fig, ax = mpf.plot(X = x, Y = data, Xlabel='Region', Ylabel='HPV vaksiner', ltype = 'o-',
                                       titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                    locator=MaxNLocator(nbins=20)
                    ax.xaxis.set_major_locator(locator)
                    ax.set_xticklabels(['0']+list(xlabels), rotation=45)
                    ax.legend(legends)
                #-----------------------------------------------------------------------
                elif plot_type == 'bar':
                    ## plotting
                    fig, ax = mpf.barplot(X = xlabels, Y= data, Xlabel='Region', Ylabel='HPV vaksiner', width = 2,
                                          titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                    ax.legend(legends)
                #-----------------------------------------------------------------------
                elif plot_type == '3d':
                    ## plotting
                    fig, ax = mpf.surface_plot(Z = np.array(data).T, Y=x, X=y, Ylabel='Region', 
                                               Xlabel='År', Zlabel="HPV vaksiner",
                                               titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")

                    for i, label in enumerate(xlabels):
                        print('Index '+str(i)+' corresponds to '+str(label)+' region.')
                
                
                elif plot_type == 'contour':
                    ## plotting
                    fig, ax = mpf.contour3d_plot(Z = np.array(data).T, Y=x, X=y, Ylabel='Region', 
                                                 Xlabel='År', Zlabel="HPV vaksiner",
                                                 titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")

                    for i, label in enumerate(xlabels):
                        print('Index '+str(i)+' corresponds to '+str(label)+' region.')
                    
        #-----------------------------------------------------------------------
        elif data_type == 'kreft':
            #-----------------------------------------------------------------------
            if perspective == 'Alder':
                #-----------------------------------------------------------------------
                if plot_type == 'comparison':
                    if period_start != period_end:
                        print("Please select one year, if you want to plot a 'comparison' plot.")
                    
                    c_label = ['Livmorhals','Anus','Munn, andre','Livmorlegeme','Livmor, usesifisert']
                    
                    ### collecting data
                    data_LH, *_ = master_data(period_start = period_start, period_end = period_end, 
                                              gender = gender, data_type = 'kreft', cancer_type = 'Livmorhals', 
                                              perspective = 'Alder')
                    data_A, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                                 gender = gender, data_type = 'kreft', cancer_type = 'Anus', 
                                                                 perspective = 'Alder')
                    data_MO, *_ = master_data(period_start = period_start, period_end = period_end, 
                                              gender = gender, data_type = 'kreft', cancer_type = 'Munn, andre', 
                                              perspective = 'Alder')
                    data_LL, *_ = master_data(period_start = period_start, period_end = period_end, 
                                              gender = gender, data_type = 'kreft', cancer_type = 'Livmorlegeme', 
                                              perspective = 'Alder')
                    data_L, *_ = master_data(period_start = period_start, period_end = period_end, 
                                             gender = gender, data_type = 'kreft', cancer_type = 'Livmor, usesifisert', 
                                             perspective = 'Alder')
                    
                    ## plotting
                    fig, ax = mpf.plot(X = x, Y = data_LH, Xlabel='Alder', Ylabel='Insidensrate', ltype = 'o-',
                                       titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                    ax.plot(x, np.array(data_A).T, 'o-')
                    ax.plot(x, np.array(data_MO).T, 'o-')
                    ax.plot(x, np.array(data_LL).T, 'o-')
                    ax.plot(x, np.array(data_L).T, 'o-')
                    
                    locator=MaxNLocator(nbins=9)
                    ax.xaxis.set_major_locator(locator)
                    ax.set_xticklabels(['0'] + list(xlabels))
                    ax.legend(c_label)
                #-----------------------------------------------------------------------    
                elif plot_type == 'comparison per year':
                    c_label = ['Livmorhals','Anus','Munn, andre','Livmorlegeme','Livmor, usesifisert']
                    
                    ### collecting data
                    data_LH, *_ = master_data(period_start = period_start, period_end = period_end, 
                                              gender = gender, data_type = 'kreft', cancer_type = 'Livmorhals', 
                                              perspective = 'Alder')
                    data_A, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                                 gender = gender, data_type = 'kreft', cancer_type = 'Anus', 
                                                                 perspective = 'Alder')
                    data_MO, *_ = master_data(period_start = period_start, period_end = period_end, 
                                              gender = gender, data_type = 'kreft', cancer_type = 'Munn, andre', 
                                              perspective = 'Alder')
                    data_LL, *_ = master_data(period_start = period_start, period_end = period_end, 
                                              gender = gender, data_type = 'kreft', cancer_type = 'Livmorlegeme', 
                                              perspective = 'Alder')
                    data_L, *_ = master_data(period_start = period_start, period_end = period_end, 
                                             gender = gender, data_type = 'kreft', cancer_type = 'Livmor, usesifisert', 
                                             perspective = 'Alder')
                    ## processing data
                    temp1 = np.zeros(len(data_LH))
                    temp2 = np.zeros(len(data_A))
                    temp3 = np.zeros(len(data_MO))
                    temp4 = np.zeros(len(data_LL))
                    temp5 = np.zeros(len(data_L))
                    
                    for iage in range(len(data_A[0])):
                        for iyear in range(len(data_A)):
                            temp1[iyear] = temp1[iyear] + data_LH[iyear][iage]
                            temp2[iyear] = temp2[iyear] + data_A[iyear][iage]
                            temp3[iyear] = temp3[iyear] + data_MO[iyear][iage]
                            temp4[iyear] = temp4[iyear] + data_LL[iyear][iage]
                            temp5[iyear] = temp5[iyear] + data_L[iyear][iage]

                    total_data_LH = temp1
                    total_data_A = temp2
                    total_data_MO = temp3
                    total_data_LL = temp4
                    total_data_L = temp5
                    
                    ## plotting 
                    fig, ax = mpf.plot(X = y, Y = total_data_LH, Xlabel='År', Ylabel='Insidensrate', ltype = 'o-',
                                       titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                    ax.plot(y, np.array(total_data_A).T, 'o-')
                    ax.plot(y, np.array(total_data_MO).T, 'o-')
                    ax.plot(y, np.array(total_data_LL).T, 'o-')
                    ax.plot(y, np.array(total_data_L).T, 'o-')
                    ax.legend(c_label)
                #-----------------------------------------------------------------------    
                elif cancer_type == 'Anus':
                    ### collecting data
                    data, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                               gender = gender, data_type = 'kreft', cancer_type = 'Anus', 
                                                               perspective = 'Alder')
                    if plot_type == 'line':
                        ## plotting
                        fig, ax = mpf.plot(X = x, Y= data, Xlabel='Alder', Ylabel='Insidensrate', ltype = 'o-',
                                           titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.set_xticklabels([0]+xlabels)
                        ax.legend(legends)
                
                    elif plot_type == 'bar':
                        ## plotting
                        fig, ax = mpf.barplot(X = xlabels, Y= data, Xlabel='Alder', Ylabel='Insidensrate', width = 0.8,
                                              titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.legend(legends)
                    
                    elif plot_type == '3d':
                        ## plotting
                        fig, ax = mpf.surface_plot(Z = np.array(data).T, Y=x, X=y, Xlabel='År', Ylabel='Alder', 
                                                   Zlabel='Insidensrate',
                                                   titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.set_yticklabels(xlabels)
                    
                    elif plot_type == 'contour':
                        ## plotting
                        fig, ax = mpf.contour3d_plot(Z = np.array(data).T, Y=x, X=y, Xlabel='År', Ylabel='Alder', 
                                                     Zlabel='Insidensrate',
                                                     titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.set_yticklabels(xlabels)
                #-----------------------------------------------------------------------
                elif cancer_type == 'Munn, andre':
                    ### collecting data
                    data, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                               gender = gender, data_type = 'kreft', 
                                                               cancer_type = 'Munn, andre', perspective = 'Alder')
                    if plot_type == 'line':
                        ## plotting
                        fig, ax = mpf.plot(X = x, Y= data, Xlabel='Alder', Ylabel='Insidensrate', ltype = 'o-',
                                           titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.set_xticklabels([0]+xlabels)
                        ax.legend(legends)
                
                    elif plot_type == 'bar':
                        ## plotting
                        fig, ax = mpf.barplot(X = xlabels, Y= data, Xlabel='Alder', Ylabel='Insidensrate', width = 0.8,
                                              titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.legend(legends)
                    
                    elif plot_type == '3d':
                        ## plotting
                        fig, ax = mpf.surface_plot(Z = np.array(data).T, Y=x, X=y, Xlabel='År', Ylabel='Alder', 
                                                   Zlabel='Insidensrate',
                                                   titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.set_yticklabels(xlabels)
                    
                    elif plot_type == 'contour':
                        ## plotting
                        fig, ax = mpf.contour3d_plot(Z = np.array(data).T, Y=x, X=y, Xlabel='År', Ylabel='Alder', 
                                                     Zlabel='Insidensrate',
                                                     titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.set_yticklabels(xlabels)
                #-----------------------------------------------------------------------
                else:
                    ### collecting data
                    data, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                               gender = gender, data_type = 'kreft', cancer_type = cancer_type, 
                                                               perspective = 'Alder')
                    if plot_type == 'line':
                        ## plotting
                        fig, ax = mpf.plot(X = x, Y= data, Xlabel='Alder', Ylabel='Insidensrate', ltype = 'o-',
                                           titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.set_xticklabels([0]+xlabels)
                        ax.legend(legends)
                
                    elif plot_type == 'bar':
                        ## plotting
                        fig, ax = mpf.barplot(X = xlabels, Y= data, Xlabel='Alder', Ylabel='Insidensrate', width = 0.8,
                                              titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.legend(legends)
                    
                    elif plot_type == '3d':
                        ## plotting
                        fig, ax = mpf.surface_plot(Z = np.array(data).T, Y=x, X=y, Xlabel='År', Ylabel='Alder', 
                                                   Zlabel='Insidensrate',
                                                   titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.set_yticklabels(xlabels)
                    
                    elif plot_type == 'contour':
                        ## plotting
                        fig, ax = mpf.contour3d_plot(Z = np.array(data).T, Y=x, X=y, Xlabel='År', Ylabel='Alder', 
                                                     Zlabel='Insidensrate',
                                                     titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.set_yticklabels(xlabels)
            #-----------------------------------------------------------------------
            elif perspective == 'Region':
                #-----------------------------------------------------------------------
                if plot_type == 'comparison':
                    if period_start != period_end:
                        print("Please select one year, if you want to plot a 'comparison' plot.")
                    
                    c_label = ['Livmorhals','Anus','Munn, andre','Livmorlegeme','Livmor, usesifisert']
                    
                    ### collecting data
                    data_LH, *_ = master_data(period_start = period_start, period_end = period_end, 
                                              gender = gender, data_type = 'kreft', cancer_type = 'Livmorhals', 
                                              perspective = perspective)
                    data_A, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                                 gender = gender, data_type = 'kreft', cancer_type = 'Anus', 
                                                                 perspective = perspective)
                    data_MO, *_ = master_data(period_start = period_start, period_end = period_end, 
                                              gender = gender, data_type = 'kreft', cancer_type = 'Munn, andre', 
                                              perspective = perspective)
                    data_LL, *_ = master_data(period_start = period_start, period_end = period_end, 
                                              gender = gender, data_type = 'kreft', cancer_type = 'Livmorlegeme', 
                                              perspective = perspective)
                    data_L, *_ = master_data(period_start = period_start, period_end = period_end, 
                                             gender = gender, data_type = 'kreft', cancer_type = 'Livmor, usesifisert', 
                                             perspective = perspective)
                    
                    ## plotting
                    fig, ax = mpf.plot(X = x, Y = data_LH, Xlabel='Region', Ylabel='Insidensrate', ltype = 'o-',
                                       titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+" år "+str(y)+"")
                    ax.plot(x, np.array(data_A).T, 'o-')
                    ax.plot(x, np.array(data_MO).T, 'o-')
                    ax.plot(x, np.array(data_LL).T, 'o-')
                    ax.plot(x, np.array(data_L).T, 'o-')
                    
                    locator=MaxNLocator(nbins=20)
                    ax.xaxis.set_major_locator(locator)
                    ax.set_xticklabels(['0'] + list(xlabels), rotation=45)
                    ax.legend(c_label)
                #-----------------------------------------------------------------------    
                elif plot_type == 'comparison per year':
                    c_label = ['Livmorhals','Anus','Munn, andre','Livmorlegeme','Livmor, usesifisert']
                    
                    ### collecting data
                    data_LH, *_ = master_data(period_start = period_start, period_end = period_end, 
                                              gender = gender, data_type = 'kreft', cancer_type = 'Livmorhals', 
                                              perspective = perspective)
                    data_A, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                                 gender = gender, data_type = 'kreft', cancer_type = 'Anus', 
                                                                 perspective = perspective)
                    data_MO, *_ = master_data(period_start = period_start, period_end = period_end, 
                                              gender = gender, data_type = 'kreft', cancer_type = 'Munn, andre', 
                                              perspective = perspective)
                    data_LL, *_ = master_data(period_start = period_start, period_end = period_end, 
                                              gender = gender, data_type = 'kreft', cancer_type = 'Livmorlegeme', 
                                              perspective = perspective)
                    data_L, *_ = master_data(period_start = period_start, period_end = period_end, 
                                             gender = gender, data_type = 'kreft', cancer_type = 'Livmor, usesifisert', 
                                             perspective = perspective)
                    ## processing data
                    temp1 = np.zeros(len(data_LH))
                    temp2 = np.zeros(len(data_A))
                    temp3 = np.zeros(len(data_MO))
                    temp4 = np.zeros(len(data_LL))
                    temp5 = np.zeros(len(data_L))
                    
                    for iage in range(len(data_A[0])):
                        for iyear in range(len(data_A)):
                            temp1[iyear] = temp1[iyear] + data_LH[iyear][iage]
                            temp2[iyear] = temp2[iyear] + data_A[iyear][iage]
                            temp3[iyear] = temp3[iyear] + data_MO[iyear][iage]
                            temp4[iyear] = temp4[iyear] + data_LL[iyear][iage]
                            temp5[iyear] = temp5[iyear] + data_L[iyear][iage]

                    total_data_LH = temp1
                    total_data_A = temp2
                    total_data_MO = temp3
                    total_data_LL = temp4
                    total_data_L = temp5
                    
                    ## plotting 
                    fig, ax = mpf.plot(X = y, Y = total_data_LH, Xlabel='År', Ylabel='Insidensrate', ltype = 'o-',
                                       titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                    ax.plot(y, np.array(total_data_A).T, 'o-')
                    ax.plot(y, np.array(total_data_MO).T, 'o-')
                    ax.plot(y, np.array(total_data_LL).T, 'o-')
                    ax.plot(y, np.array(total_data_L).T, 'o-')
                    ax.legend(c_label)
                #-----------------------------------------------------------------------
                elif cancer_type == 'Anus':
                    ### collecting data
                    data, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                               gender = gender, data_type = 'kreft', cancer_type = 'Anus', 
                                                               perspective = 'Region')
                    if plot_type == 'line':
                        ## plotting
                        fig, ax = mpf.plot(X = x, Y = data, Xlabel='Region', Ylabel='Insidensrate', ltype = 'o-', 
                                           titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        locator=MaxNLocator(nbins=20)
                        ax.xaxis.set_major_locator(locator)
                        ax.set_xticklabels(['0']+list(xlabels), rotation=45)
                        ax.legend(legends)
            
                    elif plot_type == 'bar':
                        ## plotting
                        fig, ax = mpf.barplot(X = xlabels, Y= data, Xlabel='Region', Ylabel='Insidensrate', width = 2,
                                              titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.legend(legends)
                    
                    elif plot_type == '3d':
                        ## plotting
                        fig, ax = mpf.surface_plot(Z = np.array(data).T, Y=x, X=y, Ylabel='Region', 
                                                   Xlabel='År', Zlabel="Insidensrate",
                                                   titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")

                        for i, label in enumerate(xlabels):
                            print('Index '+str(i)+' corresponds to '+str(label)+' region.')
                
                
                    elif plot_type == 'contour':
                        ## plotting
                        fig, ax = mpf.contour3d_plot(Z = np.array(data).T, Y=x, X=y, Ylabel='Region', 
                                                     Xlabel='År', Zlabel="Insidensrate",
                                                     titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")

                        for i, label in enumerate(xlabels):
                            print('Index '+str(i)+' corresponds to '+str(label)+' region.')
                #-----------------------------------------------------------------------
                elif cancer_type == 'Munn, andre':
                    ### collecting data
                    data, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                               gender = gender, data_type = 'kreft', 
                                                               cancer_type = 'Munn, andre', perspective = 'Region')
                    if plot_type == 'line':
                        ## plotting
                        fig, ax = mpf.plot(X = x, Y = data, Xlabel='Region', Ylabel='Insidensrate', ltype = 'o-',
                                           titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        locator=MaxNLocator(nbins=20)
                        ax.xaxis.set_major_locator(locator)
                        ax.set_xticklabels(['0']+list(xlabels), rotation=45)
                        ax.legend(legends)
            
                    elif plot_type == 'bar':
                        ## plotting
                        fig, ax = mpf.barplot(X = xlabels, Y= data, Xlabel='Region', Ylabel='Insidensrate', width = 2,
                                              titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.legend(legends)
                    
                    elif plot_type == '3d':
                        ## plotting
                        fig, ax = mpf.surface_plot(Z = np.array(data).T, Y=x, X=y, Ylabel='Region', 
                                                   Xlabel='År', Zlabel="Insidensrate",
                                                   titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")

                        for i, label in enumerate(xlabels):
                            print('Index '+str(i)+' corresponds to '+str(label)+' region.')
                
                
                    elif plot_type == 'contour':
                        ## plotting
                        fig, ax = mpf.contour3d_plot(Z = np.array(data).T, Y=x, X=y, Ylabel='Region', 
                                                     Xlabel='År', Zlabel="Insidensrate",
                                                     titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")

                        for i, label in enumerate(xlabels):
                            print('Index '+str(i)+' corresponds to '+str(label)+' region.')
                #-----------------------------------------------------------------------
                else:
                    ### collecting data
                    data, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                               gender = gender, data_type = 'kreft', cancer_type = cancer_type, 
                                                               perspective = 'Region')
                    if plot_type == 'line':
                        ## plotting
                        fig, ax = mpf.plot(X = x, Y = data, Xlabel='Region', Ylabel='Insidensrate', ltype = 'o-', 
                                           titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        locator=MaxNLocator(nbins=20)
                        ax.xaxis.set_major_locator(locator)
                        ax.set_xticklabels(['0']+list(xlabels), rotation=45)
                        ax.legend(legends)
            
                    elif plot_type == 'bar':
                        ## plotting
                        fig, ax = mpf.barplot(X = xlabels, Y= data, Xlabel='Region', Ylabel='Insidensrate', width = 2,
                                              titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.legend(legends)
                    
                    elif plot_type == '3d':
                        ## plotting
                        fig, ax = mpf.surface_plot(Z = np.array(data).T, Y=x, X=y, Ylabel='Region', 
                                                   Xlabel='År', Zlabel="Insidensrate",
                                                   titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")

                        for i, label in enumerate(xlabels):
                            print('Index '+str(i)+' corresponds to '+str(label)+' region.')
                
                
                    elif plot_type == 'contour':
                        ## plotting
                        fig, ax = mpf.contour3d_plot(Z = np.array(data).T, Y=x, X=y, Ylabel='Region', 
                                                     Xlabel='År', Zlabel="Insidensrate",
                                                     titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")

                        for i, label in enumerate(xlabels):
                            print('Index '+str(i)+' corresponds to '+str(label)+' region.')
        #-----------------------------------------------------------------------
        elif data_type == 'begge':
            #-----------------------------------------------------------------------
            if perspective == 'Alder':
                ### collecting data
                data_vac, *_ = master_data(period_start = period_start, period_end = period_end, 
                                           gender = gender, data_type = 'vaksiner', perspective = 'Alder')
                #-----------------------------------------------------------------------
                if cancer_type == 'Anus':
                    ### collecting data
                    data, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                               gender = gender, data_type = 'kreft', cancer_type = 'Anus', 
                                                               perspective = 'Alder')
                    #-----------------------------------------------------------------------
                    if plot_type == 'bar line':
                        ## plotting
                        fig, ax = mpf.barplot(X = xlabels, Y= data, Xlabel='Alder', Ylabel='HPV vaksiner', width = 0.8,
                                              titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.legend(legends)
                        ax2 = ax.twinx()
                        for i in range(len(data_vac)):
                            ax2.plot(x, data_vac[i], 'o--')
                        ax2.set_ylabel('HPV vaksiner (lines)')
                        
                    #-----------------------------------------------------------------------
                    elif plot_type == 'bar line per year':
                        ## processing data
                        temp1 = np.zeros(len(data_vac))
                        temp2 = np.zeros(len(data))
                    
                        for iage in range(len(data_vac[0])):
                            for iyear in range(len(data_vac)):
                                temp1[iyear] = temp1[iyear] + data_vac[iyear][iage]
                                temp2[iyear] = temp2[iyear] + data[iyear][iage]

                        total_data_vac = temp1
                        total_data = temp2
                        
                        ## plotting 
                        fig, ax = mpf.barplot(X = y, Y= total_data, Xlabel='Alder', Ylabel='Insidensrate', width = 0,
                                              titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax2 = ax.twinx()
                        ax2.plot(y, total_data_vac, 'ro-')
                        ax2.set_ylabel('HPV vaksiner (lines)')
                    #-----------------------------------------------------------------------
                    elif plot_type == 'scatter':
                        ## plotting
                        fig, ax = mpf.scatterplot(data_vac, data, Xlabel='HPV vaksiner', 
                                                  Ylabel=''+str(cancer_type)+' '+str(perspective)+' Insidensrate', 
                                                  titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.legend(legends)
                        
                        
                    #-----------------------------------------------------------------------
                    elif plot_type =='scatter per year':
                        ## precessing data
                        ## processing data
                        temp1 = np.zeros(len(data_vac))
                        temp2 = np.zeros(len(data))
                    
                        for iage in range(len(data_vac[0])):
                            for iyear in range(len(data_vac)):
                                temp1[iyear] = temp1[iyear] + data_vac[iyear][iage]
                                temp2[iyear] = temp2[iyear] + data[iyear][iage]

                        total_data_vac = temp1
                        total_data = temp2
                        
                        ## plotting
                        fig, ax = mpf.scatterplot(total_data_vac, total_data, Xlabel='HPV vaksiner', 
                                                  Ylabel=''+str(cancer_type)+' '+str(perspective)+' Insidensrate', 
                                                  titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        
                #-----------------------------------------------------------------------     
                elif cancer_type == 'Munn, andre':
                    data, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                               gender = gender, data_type = 'kreft', 
                                                               cancer_type = 'Munn, andre', perspective = 'Alder')
                    #-----------------------------------------------------------------------
                    if plot_type == 'bar line':
                        ## plotting
                        fig, ax = mpf.barplot(X = xlabels, Y= data, Xlabel='Alder', Ylabel='HPV vaksiner', width = 0.8,
                                              titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.legend(legends)
                        ax2 = ax.twinx()
                        for i in range(len(data_vac)):
                            ax2.plot(x, data_vac[i], 'o--')
                        ax2.set_ylabel('HPV vaksiner (lines)')
                        
                    #-----------------------------------------------------------------------
                    elif plot_type == 'bar line per year':
                        ## processing data
                        temp1 = np.zeros(len(data_vac))
                        temp2 = np.zeros(len(data))
                    
                        for iage in range(len(data_vac[0])):
                            for iyear in range(len(data_vac)):
                                temp1[iyear] = temp1[iyear] + data_vac[iyear][iage]
                                temp2[iyear] = temp2[iyear] + data[iyear][iage]

                        total_data_vac = temp1
                        total_data = temp2
                        
                        ## plotting 
                        fig, ax = mpf.barplot(X = y, Y= total_data, Xlabel='Alder', Ylabel='Insidensrate', width = 0.8,
                                              titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax2 = ax.twinx()
                        ax2.plot(y, total_data_vac, 'ro-')
                        ax2.set_ylabel('HPV vaksiner (lines)')
                    #-----------------------------------------------------------------------
                    elif plot_type == 'scatter':
                        ## plotting
                        fig, ax = mpf.scatterplot(data_vac, data, Xlabel='HPV vaksiner', 
                                                  Ylabel=''+str(cancer_type)+' '+str(perspective)+' Insidensrate', 
                                                  titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.legend(legends)
                    #-----------------------------------------------------------------------
                    elif plot_type =='scatter per year':
                        ## precessing data
                        ## processing data
                        temp1 = np.zeros(len(data_vac))
                        temp2 = np.zeros(len(data))
                    
                        for iage in range(len(data_vac[0])):
                            for iyear in range(len(data_vac)):
                                temp1[iyear] = temp1[iyear] + data_vac[iyear][iage]
                                temp2[iyear] = temp2[iyear] + data[iyear][iage]

                        total_data_vac = temp1
                        total_data = temp2
                        
                        ## plotting
                        fig, ax = mpf.scatterplot(total_data_vac, total_data, Xlabel='HPV vaksiner', 
                                                  Ylabel=''+str(cancer_type)+' '+str(perspective)+' Insidensrate', 
                                                  titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                #-----------------------------------------------------------------------
                else:
                    ### collecting data
                    data, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                               gender = gender, data_type = 'kreft', cancer_type = cancer_type, 
                                                               perspective = 'Alder')
                    #-----------------------------------------------------------------------
                    if plot_type == 'bar line':
                        ## plotting
                        fig, ax = mpf.barplot(X = xlabels, Y= data, Xlabel='Alder', Ylabel='HPV vaksiner', width = 0.8,
                                              titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.legend(legends)
                        ax2 = ax.twinx()
                        for i in range(len(data_vac)):
                            ax2.plot(x, data_vac[i], 'o--')
                        ax2.set_ylabel('HPV vaksiner (lines)')
                        
                    #-----------------------------------------------------------------------
                    elif plot_type == 'bar line per year':
                        ## processing data
                        temp1 = np.zeros(len(data_vac))
                        temp2 = np.zeros(len(data))
                     
                        for iage in range(len(data_vac[0])):
                            for iyear in range(len(data_vac)):
                                temp1[iyear] = temp1[iyear] + data_vac[iyear][iage]
                                temp2[iyear] = temp2[iyear] + data[iyear][iage]

                        total_data_vac = temp1
                        total_data = temp2
                        
                        ## plotting 
                        fig, ax = mpf.barplot(X = y, Y= total_data, Xlabel='Alder', Ylabel='Insidensrate', width = 0,
                                              titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax2 = ax.twinx()
                        ax2.plot(y, total_data_vac, 'ro-')
                        ax2.set_ylabel('HPV vaksiner (lines)')
                    #-----------------------------------------------------------------------
                    elif plot_type == 'scatter':
                        ## plotting
                        fig, ax = mpf.scatterplot(data_vac, data, Xlabel='HPV vaksiner', 
                                                  Ylabel=''+str(cancer_type)+' '+str(perspective)+' Insidensrate', 
                                                  titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.legend(legends)
                        
                        
                    #-----------------------------------------------------------------------
                    elif plot_type =='scatter per year':
                        ## processing data
                        temp1 = np.zeros(len(data_vac))
                        temp2 = np.zeros(len(data))
                    
                        for iage in range(len(data_vac[0])):
                            for iyear in range(len(data_vac)):
                                temp1[iyear] = temp1[iyear] + data_vac[iyear][iage]
                                temp2[iyear] = temp2[iyear] + data[iyear][iage]

                        total_data_vac = temp1
                        total_data = temp2
                        
                        ## plotting
                        fig, ax = mpf.scatterplot(total_data_vac, total_data, Xlabel='HPV vaksiner', 
                                                  Ylabel=''+str(cancer_type)+' '+str(perspective)+' Insidensrate', 
                                                  titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"") 
            #-----------------------------------------------------------------------
            elif perspective == 'Region':
                ### collecting data
                data_vac, x, y, xlabels_vac, legends = master_data(period_start = period_start, period_end = period_end,
                                                                   gender = gender, data_type = 'vaksiner', 
                                                                   perspective = 'Region')
                #-----------------------------------------------------------------------
                if cancer_type == 'Anus':
                    ### collecting data
                    data, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                               gender = gender, data_type = 'kreft', 
                                                               cancer_type = 'Anus', perspective = 'Region')
                    ### aligning cancer and vaccine data
                    ordered_data = []
                    ordered_v = []
                    ordered_xlabels = []
                    ## year loop
                    for i in range(len(legends)):
                        temp = []
                        temp_v = []
    
                        ## cancer loop
                        for j in range(len(xlabels)):
                            ## vaccine loop
                            for k in range(len(xlabels_vac)):
                                # checking when they are equal
                                if xlabels[j] == xlabels_vac[k]:
                                    temp.append( data[i][j] )
                                    temp_v.append( data_vac[i][k] )
                                    ordered_xlabels.append( xlabels[j] )
    
                        ordered_data.append(temp)
                        ordered_v.append(temp_v)
                    
                    data = ordered_data
                    data_vac = ordered_v
                    xlabels = ordered_xlabels[0:17]
                    x = np.arange(0,len(xlabels))
                    #-----------------------------------------------------------------------
                    if plot_type == 'bar line':
                        ## plotting
                        fig, ax = mpf.barplot(X = xlabels, Y = data, Xlabel='Region', Ylabel='HPV vaksiner', width = 2,
                                              titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.legend(legends)
                        ax2 = ax.twinx()
                        for i in range(len(data_vac)):
                            ax2.plot(x, data_vac[i], 'o--')
                        ax2.set_ylabel('HPV vaksiner (lines)')
                        
                    #-----------------------------------------------------------------------
                    elif plot_type == 'bar line per year':
                        ## processing data
                        temp1 = np.zeros(len(data_vac))
                        temp2 = np.zeros(len(data))
                    
                        for iregion in range(len(data_vac[0])):
                            for iyear in range(len(data_vac)):
                                temp1[iyear] = temp1[iyear] + data_vac[iyear][iregion]
                                temp2[iyear] = temp2[iyear] + data[iyear][iregion]

                        total_data_vac = temp1
                        total_data = temp2
                        
                        ## plotting 
                        fig, ax = mpf.barplot(X = y, Y= total_data, Xlabel='Region', Ylabel='Insidensrate', width = 2,
                                              titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax2 = ax.twinx()
                        ax2.plot(y, total_data_vac, 'ro-')
                        ax2.set_ylabel('HPV vaksiner (lines)')
                    #-----------------------------------------------------------------------
                    elif plot_type == 'scatter':
                        ## plotting
                        fig, ax = mpf.scatterplot(data_vac, data, Xlabel='HPV vaksiner', 
                                                  Ylabel=''+str(cancer_type)+' '+str(perspective)+' Insidensrate', 
                                                  titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.legend(legends)
                    #-----------------------------------------------------------------------
                    elif plot_type =='scatter per year':
                        ## precessing data
                        temp1 = np.zeros(len(data_vac))
                        temp2 = np.zeros(len(data))
                    
                        for iage in range(len(data_vac[0])):
                            for iyear in range(len(data_vac)):
                                temp1[iyear] = temp1[iyear] + data_vac[iyear][iage]
                                temp2[iyear] = temp2[iyear] + data[iyear][iage]

                        total_data_vac = temp1
                        total_data = temp2
                        
                        ## plotting
                        fig, ax = mpf.scatterplot(total_data_vac, total_data, Xlabel='HPV vaksiner', 
                                                  Ylabel=''+str(cancer_type)+' '+str(perspective)+' Insidensrate', 
                                                  titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                #-----------------------------------------------------------------------
                else:
                    ### collecting data
                    data, x, y, xlabels, legends = master_data(period_start = period_start, period_end = period_end, 
                                                               gender = gender, data_type = 'kreft', 
                                                               cancer_type = cancer_type, perspective = 'Region')
                    
                    ### aligning cancer and vaccine data
                    ordered_data = []
                    ordered_v = []
                    ordered_xlabels = []
                    ## year loop
                    for i in range(len(legends)):
                        temp = []
                        temp_v = []
    
                        ## cancer loop
                        for j in range(len(xlabels)):
                            ## vaccine loop
                            for k in range(len(xlabels_vac)):
                                # checking when they are equal
                                if xlabels[j] == xlabels_vac[k]:
                                    temp.append( data[i][j] )
                                    temp_v.append( data_vac[i][k] )
                                    ordered_xlabels.append( xlabels[j] )
    
                        ordered_data.append(temp)
                        ordered_v.append(temp_v)
                    
                    data = ordered_data
                    data_vac = ordered_v
                    xlabels = ordered_xlabels[0:17]
                    x = np.arange(0,len(xlabels))
                    #-----------------------------------------------------------------------
                    if plot_type == 'bar line':
                        ## plotting
                        fig, ax = mpf.barplot(X = xlabels, Y= data, Xlabel='Region', Ylabel='HPV vaksiner', width = 2,
                                              titl=""+str(gender)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.legend(legends)
                        ax2 = ax.twinx()
                        for i in range(len(data_vac)):
                            ax2.plot(x, data_vac[i], 'o--')
                        ax2.set_ylabel('HPV vaksiner (lines)')
                        
                    #-----------------------------------------------------------------------
                    elif plot_type == 'bar line per year':
                        ## processing data
                        temp1 = np.zeros(len(data_vac))
                        temp2 = np.zeros(len(data))
                    
                        for iregion in range(len(data_vac[0])):
                            for iyear in range(len(data_vac)):
                                temp1[iyear] = temp1[iyear] + data_vac[iyear][iregion]
                                temp2[iyear] = temp2[iyear] + data[iyear][iregion]

                        total_data_vac = temp1
                        total_data = temp2
                        
                        ## plotting 
                        fig, ax = mpf.barplot(X = y, Y= total_data, Xlabel='Region', Ylabel='Insidensrate', width = 2,
                                              titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax2 = ax.twinx()
                        ax2.plot(y, total_data_vac, 'ro-')
                        ax2.set_ylabel('HPV vaksiner (lines)')
                    #-----------------------------------------------------------------------
                    elif plot_type == 'scatter':
                        ## plotting
                        fig, ax = mpf.scatterplot(data_vac, data, Xlabel='HPV vaksiner', 
                                                  Ylabel=''+str(cancer_type)+' '+str(perspective)+' Insidensrate', 
                                                  titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"")
                        ax.legend(legends)
                    #-----------------------------------------------------------------------
                    elif plot_type =='scatter per year':
                        ## precessing data
                        temp1 = np.zeros(len(data_vac))
                        temp2 = np.zeros(len(data))
                    
                        for iage in range(len(data_vac[0])):
                            for iyear in range(len(data_vac)):
                                temp1[iyear] = temp1[iyear] + data_vac[iyear][iage]
                                temp2[iyear] = temp2[iyear] + data[iyear][iage]

                        total_data_vac = temp1
                        total_data = temp2 
                        
                        ## plotting
                        fig, ax = mpf.scatterplot(total_data_vac, total_data, Xlabel='HPV vaksiner', 
                                                  Ylabel=''+str(cancer_type)+' '+str(perspective)+' Insidensrate', 
                                                  titl=""+str(gender)+" "+str(cancer_type)+" "+str(data_type)+" "+str(perspective)+" "+str(plot_type)+"") 
    #----------------------------------------------------------------------------------------------------------------
    
    if save:
        ### creating folders
        pp.create_folder('plots')
        pp.create_folder('plots/'+str(plot_type)+'')
    
        fig.savefig("plots/"+str(plot_type)+"/"+str(gender)+"_"+str(cancer_type)+"_"+str(perspective)+"_"+str(plot_type)+".png")
        
        
    return fig, ax