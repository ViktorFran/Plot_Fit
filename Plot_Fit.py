#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 21:18:08 2021

@author: victor
"""
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from scipy import optimize
import lmfit
from inspect import signature
import random

while True:    
    header_number = 0
    file = input("Drag and Drop File, or File Path: \n").strip().strip("\'").strip("\"")
    
    header_correct = False
    while header_correct == False:
        try:
            dataframe = pd.read_csv(file, header=header_number)
            header_correct = True
            dataframe = pd.read_csv(file, header = header_number + 1)
        except pd.errors.ParserError:
            header_number += 1
    
    print("")
    print("Data Columns: ")
    i = 1
    for columns in list(dataframe.columns):
        print(str(i) + ": " + columns)
        i += 1
    
    print("")
    x_data = dataframe.loc[:,list(dataframe.columns)[int(input("x variable?: "))-1]]
    y_data = dataframe.loc[:,list(dataframe.columns)[int(input("y variable?: "))-1]]
        
    do_fit = input("Fit? (Y/n): ") or "y"
    if do_fit == "y" or do_fit == "Y":
        do_fit = True
    else:
        do_fit = False
    
    if do_fit == True:    
        R = 8.31446 #J/K*mol 
        ######################################################################
        ### Define Model Functions ###
        ######################################################################
        ### Add new functions here
        def biMolecularDuplex(x, N, a, D, b, H, Tm):
            return N+a*x+((b-a)*x+D-N)*(0.25*np.exp(H/R*(1/(273.15+Tm)-1/(273.15+x)))*((8*np.exp(H/R*(1/(273.15+x)-1/(273.15+Tm)))+1)**(0.5)-1))
        
        def linear(x, m, b):
            return m*x + b
        
        def tetraMolecular(x, N, a, D, b, H, Tm):
            C = 8
            K = np.exp((H/R)*((1/Tm)-(1/x)))
            u = np.cbrt(np.sqrt(3)*np.sqrt(256*(C**3)*(K**3)+27*(C**2)*(K**4))+9*C*(K**2))
            u2 = np.cbrt(18)*C
            u3 = 4*np.cbrt(2/3)*K
            u4 = ((-8*(-4*C-K))/C)-32
            u5 = ((u/u2)-(u3/u))
            v = (1/2)*np.sqrt(u5)
            w = (1/2)*np.sqrt((u4)/(4*np.sqrt(u5))-u5)
            # return N+a*x+((b-a)*x+D-N)*(v - w + 1)
            return (v - w + 1)

        ### Add new functions to the dictionary
        model_dict = {  
            "biMolecularDuplex" : biMolecularDuplex,
            "linear" : linear,
            "tetraMolecular" : tetraMolecular,
            }
        
        ### Define initial value for each function
        init_dict = {
            "biMolecularDuplex" : dict(N=0.1, a=0.001, D=0.1, b=0.001, H=200000, Tm=65),
            "linear" : dict(m=0, b=0),
            "tetraMolecular" : dict(N=0.1, a=0.001, D=0.1, b=0.001, H=130000, Tm=65),
            }
        
        ######################################################################
        print("")
        print("Model Functions: ")
        i = 1
        for key in sorted(model_dict):
            print(str(i)+": ", key)
            i += 1
        
        print("")
        model_number = int(input("Choose model function: ")) - 1
        model_name = sorted(model_dict.keys())[model_number]
        chosen_model = model_dict[model_name]
        
        var_count = len(signature(chosen_model).parameters) - 1
        n_count = len(x_data.index)

        #######################################################################
        ### optimize.curve_fit ### 
        #######################################################################
        # param, param_coeff = optimize.curve_fit(chosen_model, x_data, y_data, p0=[0.1, 0.001, 0.1, 0.001, 200000, 65], bounds=(-np.inf, np.inf))
        # # plt.plot(x_data, chosen_model(x_data, *param), linestyle='-', color='red')
        # # plt.show()
        # residuals = y_data - chosen_model(x_data, *param)
        # ss_res = np.sum(residuals**2)
        # ss_tot = np.sum((y_data-np.mean(y_data))**2)
        # r2 = 1 - (ss_res / ss_tot)
        # print("R2:", r2)
        # print(param)
        
        #######################################################################
        ### limfit ###
        #######################################################################
        model = lmfit.Model(chosen_model)
        initial = model.make_params(**init_dict[model_name])
        # print('parameter names: {}'.format(model.param_names))
        # print('independent variables: {}'.format(model.independent_vars))
        fitted = model.fit(data=y_data, params=initial, x=x_data, method='leastsq', nan_policy='omit', max_nfev=1000)
        print("-"*50)
        print(fitted.fit_report(show_correl=False))
        # print("Tm:", fitted.params.get('Tm').value, "+/-", fitted.params.get('Tm').stderr)
        
        ssr = np.sum((y_data - fitted.best_fit)**2)
        sst = np.sum((y_data - np.mean(fitted.best_fit))**2)
        # r2 = 1 - fitted.residual.var() / np.var(y_data)
        r2 = 1-ssr/sst
        r2_adj = 1-(1-r2)*(n_count-1)/(n_count-var_count-1)
        print("")
        print("R2:   " + "\t" + str(r2))
        print("R2adj:\t" + str(r2_adj))
        print("-"*50)
        plt.scatter(x_data, y_data, marker='.')
        plt.plot(x_data, fitted.best_fit, linestyle='-', color='red')
        try:
            plt.annotate('Tm: '+ str(fitted.params.get('Tm').value)+ " +/- "+str(fitted.params.get('Tm').stderr)+"\n"+"R2: " + str(r2), xy=(0, 1), xytext=(4, -4), va='top', xycoords='axes fraction', textcoords='offset points')
        except:
            pass
        plt.title(file)
        plt.xlabel(str(x_data.name).strip())
        plt.ylabel(str(y_data.name).strip())
        plt.show()
        
        # for i in range(5):
        #     fitted.fit(data=y_data, param=fitted.params * random.random(), x=x_data, nan_policy='omit')
        #     print(fitted.fit_report(show_correl=False))
        #     plt.scatter(x_data, y_data, marker='.')
        #     plt.plot(x_data, fitted.best_fit, linestyle='-', color='red')
        #     plt.title(file)
        #     plt.xlabel(str(x_data.name).strip())
        #     plt.ylabel(str(y_data.name).strip())
        #     plt.show()

        ### residual plot ###
        # fitted.plot()
    else:
        plt.scatter(x_data, y_data, marker='.')
        plt.title(file)
        plt.xlabel(str(x_data.name).strip())
        plt.ylabel(str(y_data.name).strip())
        plt.show()
    print("")    
