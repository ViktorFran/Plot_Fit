#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 22 21:07:44 2021

@author: victor
"""
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import lmfit
  
header_number = 0
file = "~/Downloads/test.csv"

header_correct = False
while header_correct == False:
    try:
        dataframe = pd.read_csv(file, header=header_number)
        header_correct = True
        dataframe = pd.read_csv(file, header = header_number + 1)
    except pd.errors.ParserError:
        header_number += 1

print("")
# print("Data Columns: ")
i = 1
for columns in list(dataframe.columns):
    print(str(i) + ": " + columns)
    i += 1

# print("")
x_data = dataframe.loc[:,list(dataframe.columns)[0]]
y_data = dataframe.loc[:,list(dataframe.columns)[1]]

def tetraMolecular(x, H, Tm):
    R = 8.31446
    C = 8
    K = np.exp((H/R)*((1/(Tm+273.15))-(1/x)))
    u = np.cbrt(np.sqrt(3)*np.sqrt(256*(C**3)*(K**3)+27*(C**2)*(K**4))+9*C*(K**2))
    u2 = np.cbrt(18)*C
    u3 = 4*np.cbrt(2/3)*K
    u4 = ((-8*(-4*C-K))/C)-32
    u5 = ((u/u2)-(u3/u))
    v = (1/2)*np.sqrt(u5)
    w = (1/2)*np.sqrt((u4)/(4*np.sqrt(u5))-u5)
    # return N+a*x+((b-a)*x+D-N)*(v - w + 1)
    return (v - w + 1)

# model = lmfit.Model(tetraMolecular)
# params = model.make_params(H=30000, Tm=65)

# fit_tet = model.fit(data=y_data, params, x=x_data)
# print(fit_tet)

H = 200000        
x = np.array(range(20,90))

for i in range(200):
    Tm = i*0.5 + 0.5
    y = tetraMolecular(x, H, Tm)
    plt.title(str(Tm))
    plt.plot(x,y)
    plt.show()    
