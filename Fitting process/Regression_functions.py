import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import LinearLocator
from matplotlib.ticker import FormatStrFormatter
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
import sklearn.linear_model as lm
from numba import jit
import sklearn.metrics as skm
import scipy as sp


#@jit
def gen_def_matrix(x, y, k=1):
    exp_x = []
    exp_y = []
    
    x = np.ravel(x)
    y = np.ravel(y)
    
    
    xb = np.ones((x.size, 1))
    for i in range(1, k+1):
        for j in range(i+1):
            #print('x and y exponent: ', ((i-j),j))
            xb = np.c_[xb, (x**(i-j))*(y**j)]
            
    return xb

def gen_beta(data, output, lam=0):

    dim = np.shape(data)[1]
    U, sig_v, V_star = sp.linalg.svd(data)
    sig = sp.linalg.diagsvd(sig_v, U.shape[0], V_star.shape[1])
    if lam == 0:
        beta = np.conjugate(V_star).T @ sp.linalg.pinv(sig) @ np.conjugate(U).T @ output
    else:
        beta = np.linalg.inv(data.T @ data + lam * np.eye(dim)).dot(data.T.dot(output))
    
    return beta

#@jit
def OLS(data_train, output_train, data_test, output_test):

    output = output_train.ravel()


    beta = gen_beta(data_train, output)
    
    ## test data
    op = data_test.dot(beta)
    output_predict = op.reshape(output_test.shape)

    MSE = mean_squared_error(output_test, output_predict)
    R2 = r2_score(output_test, output_predict)

    ## traning data
    ot = data_train.dot(beta)

    MSEt = mean_squared_error(output, ot)
    R2t = r2_score(output, ot)
    
    return MSE, R2, MSEt, R2t, beta

#@jit
def Ridge(data_train, output_train, data_test, output_test, lam=0.01):

    output = output_train.ravel()

    beta = gen_beta(data_train, output, lam)
    
    ## evaluating test data
    op = data_test.dot(beta)
    output_predict = op.reshape(output_test.shape)

    MSE = mean_squared_error(output_test, output_predict)
    R2 = r2_score(output_test, output_predict)
    
    ## evaluating traning data
    ot = data_train.dot(beta)

    MSEt = mean_squared_error(output, ot)
    R2t = r2_score(output, ot)
    
    return MSE, R2, MSEt, R2t, beta

#@jit
def Lasso(data_train, output_train, data_test, output_test, lam=0.0001, n_iter=10000, k=1, graph=True):

    output = output_train.ravel()

    lasso_reg = lm.Lasso(alpha=lam,max_iter=n_iter)

    lasso_reg.fit(data_train, output)
    
    ## evaluating test data
    op = lasso_reg.predict(data_test)
    output_predict = op.reshape(output_test.shape)

    MSE = mean_squared_error(output_test, output_predict)
    R2 = r2_score(output_test, output_predict)
    
    ## evaluating training data
    ot = lasso_reg.predict(data_train)

    MSEt = mean_squared_error(output, ot)
    R2t = r2_score(output, ot)

    
    return MSE, R2, MSEt, R2t, lasso_reg.coef_

def cross_validation(data, output, k=5, lam=0.001, method='OLS'):

    o = output.ravel()

    shf = np.random.permutation(o.size)

    d = data[shf]
    o = o[shf].reshape(output.shape)

    size_fold = int(np.ceil(len(o) / k))
    tMSE_test = 0
    tR2_test = 0
    aMSE_test = np.zeros(k)
    aR2_test = np.zeros(k)
    tMSE_train = 0
    tR2_train = 0
    aMSE_train = np.zeros(k)
    aR2_train = np.zeros(k)
    b = np.empty((k,np.shape(data)[1]))
    for i in range(k):
        start_val = size_fold*i
        end_val = min(size_fold*(i+1), len(d))
        data_test = d[start_val:end_val]
        output_test = o[start_val:end_val]
        data_train = np.r_[d[0:start_val], d[end_val:]]
        output_train = np.r_[o[0:start_val], o[end_val:]]

        if(method == 'OLS'):
            MSE, R2, MSEt, R2t, b[i] = OLS(data_train, output_train, data_test, output_test)
        elif(method == 'Ridge'):
            MSE, R2, MSEt, R2t, b[i] = Ridge(data_train, output_train, data_test, output_test, lam)
        elif(method == 'Lasso'):
            MSE, R2, MSEt, R2t, b[i] = Lasso(data_train, output_train, data_test, output_test, lam)

        tMSE_test = tMSE_test + MSE
        tR2_test = tR2_test + R2
        aMSE_test[i] = MSE
        aR2_test[i] = R2

        tMSE_train = tMSE_train + MSEt
        tR2_train = tR2_train + R2t
        aMSE_train[i] = MSEt
        aR2_train[i] = R2t

        print("The train average MSE for fold %d is: %.05f; the train average R^2-score for fold %d is: %.02f" % (i+1,MSEt,i+1, R2t))
        print("The test average MSE for fold %d is: %.05f; the test average R^2-score for fold %d is: %.02f" % (i+1,MSE,i+1, R2))

    

    preds = np.empty(k)
    bias = 0
    var = 0
    for i in range(len(data)):
        for j in range(k):
            preds[j] = b[j] @ data[i]
        ex = np.mean(preds)
        bias = bias + (output[i] - ex)**2
        var = var + np.var(preds)

    tMSE_test = tMSE_test / k
    tR2_test = tR2_test / k
    tMSE_train = tMSE_train / k
    tR2_train = tR2_train / k
    with open("results_cross_validation.txt", mode="a") as f:
        f.write("##############################################\n")
        f.write("Method used: %s; Number of folds: %d" % (method, k))
        if method != "OLS":
            f.write(";Lambda: %.06f" % lam)
        f.write("\nThe test average MSE is: %.05f; the test average R^2-score is: %.02f\n" % (tMSE_test, tR2_test))
        f.write("The train average MSE is: %.05f; the train average R^2-score is: %.02f\n" % (tMSE_train, tR2_train))
        f.write("The variance: %.05f\n" % (var/len(data)))
        f.write("The bias is: %.05f\n" % (bias/len(data)))
    print("The test average MSE is: %.05f; the test average R^2-score is: %.02f" % (tMSE_test, tR2_test))
    print("The train average MSE is: %.05f; the train average R^2-score is: %.02f" % (tMSE_train, tR2_train))
    print("The variance is: ", var/len(data))
    print("The bias is: ", bias/len(data))
    return aMSE_test, aR2_test, aMSE_train, aR2_train, tMSE_train, tR2_train, tMSE_test, tR2_test

def bootstrap(data, output, it=100, lam=0.001, method='OLS'):

    tMSE_test = 0
    tR2_test = 0
    aMSE_test = np.zeros(it)
    aR2_test = np.zeros(it)
    tMSE_train = 0
    tR2_train = 0
    aMSE_train = np.zeros(it)
    aR2_train = np.zeros(it)
    for i in range(it):
        d, o = resample(data, output)

        if(method == 'OLS'):
            MSE, R2, MSEt, R2t, b = OLS(d, o, data, output)
        elif(method == 'Ridge'):
            MSE, R2, MSEt, R2t, b = Ridge(d, o, data, output, lam)
        elif(method == 'Lasso'):
            MSE, R2, MSEt, R2t, b = Lasso(d, o, data, output, lam)
        
        tMSE_test = tMSE_test + MSE
        tR2_test = tR2_test + R2
        aMSE_test[i] = MSE
        aR2_test[i] = R2

        tMSE_train = tMSE_train + MSEt
        tR2_train = tR2_train + R2t
        aMSE_train[i] = MSEt
        aR2_train[i] = R2t
    
    tMSE_test = tMSE_test / it
    tR2_test = tR2_test / it
    tMSE_train = tMSE_train / it
    tR2_train = tR2_train / it

    print("The average train MSE is: %.05f; and the average train R2 is: %.02f" % (tMSE_train, tR2_train))
    print("The average test MSE is: %.05f; and the average test R2 is: %.02f" % (tMSE_test, tR2_test))
    return aMSE_train, aR2_train, aMSE_test, aR2_test, tMSE_train, tR2_train, tMSE_test, tR2_test

def model_evaluation(data, output, lamb=0.0001, lams = [0.0001, 0.001, 0.01, 0.1, 1], lasso_iteration=10000, kfolds= [5, 6, 7, 8, 9, 10]):

    # Here we decide the data that cross_validation will be executed with. 
    data_run = data
    output_run = output

    ks = kfolds # Defining testing k-folds

    ### spliting into training and test data
    train, test, output_train, output_test = train_test_split(data_run, output_run, 
                                                              test_size=0.40, train_size=0.60, random_state=711)
    
    print ("Ordinary Least Squares method:")
    print ("-------------------------------")
    M, R, Mt, Rt, b = OLS(train, output_train, test, output_test)

    print("""The train MSE is: %.04f and the r^2-score is: %.04f.
    The test MSE is: %.04f and the r^2-score is: %.04f""" % (Mt,Rt,M,R))
    print("--------------------------------------------------------------------------")
    
    print ("Ridge regression method:")
    print ("-------------------------")
    M, R, Mt, Rt, b = Ridge(train, output_train, test, output_test, lam=lamb)

    print("""The train MSE is: %.04f and the r^2-score is: %.04f.
    The test MSE is: %.04f and the r^2-score is: %.04f""" % (Mt,Rt,M,R))
    print("--------------------------------------------------------------------------")
    
    print ("Lasso regression method:")
    print ("-------------------------")
    M, R, Mt, Rt, b = Lasso(train, output_train, test, output_test, lam=lamb, n_iter = lasso_iteration )

    print("""The train MSE is: %.04f and the r^2-score is: %.04f.
    The test MSE is: %.04f and the r^2-score is: %.04f""" % (Mt,Rt,M,R))
    print("--------------------------------------------------------------------------")
    
    
    print("Cross-validation OLS")
    print ("--------------------")
    temp = cross_validation(data_run, output_run, k = ks)
    print("-----------------------------------------------------------------------")

    print("Cross-validation Ridge")
    print ("----------------------")
    temp = cross_validation(data_run, output_run, k=ks, lam=lamb, method="Ridge")
    print("-----------------------------------------------------------------------")
    
    print("Cross-validation Lasso")
    print ("----------------------")
    temp = cross_validation(data_run, output_run, k=ks,lam=lamb, method="Lasso")
    
    ## Testing different parameters for Ridge and Lasso

    M_O = np.empty(len(lams))
    R_O = np.empty(len(lams))
    Mt_O = np.empty(len(lams))
    Rt_O = np.empty(len(lams))
    M_R = np.empty(len(lams))
    R_R = np.empty(len(lams))
    Mt_R = np.empty(len(lams))
    Rt_R = np.empty(len(lams))
    M_L = np.empty(len(lams))
    R_L = np.empty(len(lams))
    Mt_L = np.empty(len(lams))
    Rt_L = np.empty(len(lams))

    for i, lam in enumerate(lams):

        M_O[i], R_O[i], Mt_O[i], Rt_O[i], b_O = OLS(train, output_train, test, output_test)
        M_R[i], R_R[i], Mt_R[i], Rt_R[i], b_R = Ridge(train, output_train, test, output_test, lam=lam)
        M_L[i], R_L[i], Mt_L[i], Rt_L[i], b_L = Lasso(train, output_train, test, output_test, lam=lam)
    
    # Plot our performance on both the training and test data
    plt.semilogx(lams, Rt_O, 'b',label='Train (OLS)')
    plt.semilogx(lams, R_O,'--b',label='Test (OLS)')
    plt.semilogx(lams, Rt_R,'r',label='Train (Ridge)',linewidth=1)
    plt.semilogx(lams, R_R,'--r',label='Test (Ridge)',linewidth=1)
    plt.semilogx(lams, Rt_L, 'g',label='Train (LASSO)')
    plt.semilogx(lams, R_L, '--g',label='Test (LASSO)')

    fig = plt.gcf()
    fig.set_size_inches(10.0, 6.0)

    plt.legend(loc='lower left',fontsize=16)
    plt.xlim([min(lams), max(lams)])
    plt.xlabel(r'$\lambda$',fontsize=16)
    plt.ylabel('Performance',fontsize=16)
    plt.title(r'R2-score vs. $\lambda$', fontsize=16)
    plt.tick_params(labelsize=16)
    plt.show()

    # Plot our performance on both the training and test data
    plt.semilogx(lams, Mt_O, 'b',label='Train (OLS)')
    plt.semilogx(lams, M_O,'--b',label='Test (OLS)')
    plt.semilogx(lams, Mt_R,'r',label='Train (Ridge)',linewidth=1)
    plt.semilogx(lams, M_R,'--r',label='Test (Ridge)',linewidth=1)
    plt.semilogx(lams, Mt_L, 'g',label='Train (LASSO)')
    plt.semilogx(lams, M_L, '--g',label='Test (LASSO)')

    fig = plt.gcf()
    fig.set_size_inches(10.0, 6.0)

    plt.legend(loc='lower left',fontsize=16)
    plt.xlim([min(lams), max(lams)])
    plt.xlabel(r'$\lambda$',fontsize=16)
    plt.ylabel('MSE',fontsize=16)
    plt.title(r'MSE vs. $\lambda$', fontsize=16)
    plt.tick_params(labelsize=16)
    plt.show()
    