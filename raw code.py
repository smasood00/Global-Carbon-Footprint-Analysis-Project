#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
from datascience import *
from sklearn.linear_model import LinearRegression
import matplotlib
get_ipython().run_line_magic('matplotlib', 'inline')
import matplotlib.pyplot as plt
plt.style.use('fivethirtyeight')


# In[2]:


# Utilities 
def correlation_from_table(df, col_x, col_y):
    standard_units_x = (df[col_x] - np.mean(df[col_x] ) )/ np.std(df[col_x])
                        
    standard_units_y = (df[col_y] - np.mean(df[col_y] ) )/ np.std(df[col_y])
                        
    correlation = np.mean(standard_units_x * standard_units_y)
    return correlation

def slope(df, col_x, col_y):
    r = correlation_from_table(df, col_x, col_y)
    return r*np.std(df[col_x])/np.std(df[col_y])
                    
def intercept(df, col_x, col_y):
    return np.mean(df[col_y]) - slope(df, col_x, col_y)*np.mean(df[col_x])

def standard_units(data):
    standard_units_x = (data - np.mean(data)) / np.std(data)
    return standard_units_x


# In[ ]:


def visualize(year):
    
    #datacleaning
    one = nfa_all.drop(columns=["ISO alpha-3 code", "UN_subregion", "crop_land", "forest_land", "fishing_ground", "grazing_land", "built_up_land", "total"])
    two = one.loc[one["year"] == year]
    three = two.loc[two["record"] == "EFConsPerCap"]
    four = three.dropna()
    #econ level
    low = four["Percapita GDP (2010 USD)"].quantile(.33)
    middle = four["Percapita GDP (2010 USD)"].quantile(.66)
    high = four["Percapita GDP (2010 USD)"].quantile(1)
    
    #splitting country by econ levels
    lowcountries = four.loc[four["Percapita GDP (2010 USD)"] < low]
    middlecountries = four.loc[(four["Percapita GDP (2010 USD)"] > low) & (four["Percapita GDP (2010 USD)"] <= middle)]

    highcountries = four.loc[(four["Percapita GDP (2010 USD)"] > middle) & (four["Percapita GDP (2010 USD)"] <= high)]
    
    #calculating linear regression for lowcountries
    X_low = lowcountries.iloc[:, 4].values.reshape(-1, 1)  # values converts it into a numpy array
    Y_low = lowcountries.iloc[:, 5].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor_low = LinearRegression()  # create object for the class
    linear_regressor_low.fit(X_low, Y_low)  # perform linear regression
    Y_pred_low = linear_regressor_low.predict(X_low)  #
    
    lowcountryslope = linear_regressor_low.coef_

    lowcountryintercept = linear_regressor_low.intercept_
    
    plt.scatter(X_low, Y_low)
    plt.plot(X_low, Y_pred_low, color='red')
    plt.title('low countries '+ "(" + str(year) + ")")
    plt.xlabel("carbon (metric tons)")
    plt.ylabel("Percapita GDP (2010 USD)")
    plt.xlim(0, 2)
    plt.ylim(0, 2600)
    plt.show()
    
     #calculating linear regression for middle countries
    X_middle = middlecountries.iloc[:, 4].values.reshape(-1, 1)  # values converts it into a numpy array
    Y_middle = middlecountries.iloc[:, 5].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor_middle = LinearRegression()  # create object for the class
    linear_regressor_middle.fit(X_middle, Y_middle)  # perform linear regression
    Y_pred_middle = linear_regressor_middle.predict(X_middle)  #
    
    
    middlecountryslope = linear_regressor_middle.coef_

    middlecountryintercept = linear_regressor_middle.intercept_
    
    plt.scatter(X_middle, Y_middle)
    plt.plot(X_middle, Y_pred_middle, color='red')
    plt.title('middle countries '+ "(" + str(year) + ")")
    plt.xlabel("carbon (metric tons)")
    plt.ylabel("Percapita GDP (2010 USD)")
    plt.xlim(0, 6)
    plt.ylim(0, 12000)
    plt.show()
    
      #calculating linear regression for high countries
    X_high = highcountries.iloc[:, 4].values.reshape(-1, 1)  # values converts it into a numpy array
    Y_high = highcountries.iloc[:, 5].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor_high = LinearRegression()  # create object for the class
    linear_regressor_high.fit(X_high, Y_high)  # perform linear regression
    Y_pred_high = linear_regressor_high.predict(X_high)  #
    
    highcountryslope = linear_regressor_high.coef_

    highcountryintercept = linear_regressor_high.intercept_    
    
    plt.scatter(X_high, Y_high)
    plt.plot(X_high, Y_pred_high, color='red')
    plt.title("high countries " + "(" + str(year) + ")")
    plt.xlabel("carbon (metric tons)")
    plt.ylabel("Percapita GDP (2010 USD)")
    plt.xlim(0,15)
    plt.ylim(0, 90000)
    plt.show()
    
    #calculating linear regression for world

    X_world = four.iloc[:, 4].values.reshape(-1, 1)  # values converts it into a numpy array
    Y_world = four.iloc[:, 5].values.reshape(-1, 1)  # -1 means that calculate the dimension of rows, but have 1 column
    linear_regressor_world = LinearRegression()  # create object for the class
    linear_regressor_world.fit(X_world, Y_world)  # perform linear regression
    Y_pred_world = linear_regressor_world.predict(X_world)  #
    
    
    worldcountryslope = linear_regressor_world.coef_

    worldcountryintercept = linear_regressor_world.intercept_    
    
    
    plt.scatter(X_world, Y_world)
    plt.plot(X_world, Y_pred_world, color='red')
    plt.title("world " + "(" + str(year) + ")")
    plt.xlabel("carbon (metric tons)")
    plt.ylabel("Percapita GDP (2010 USD)")
    plt.xlim(0,15)
    plt.ylim(0, 100000)
    plt.show()
    
    return make_array(lowcountryslope, lowcountryintercept, middlecountryslope, middlecountryintercept,
                      highcountryslope, highcountryintercept, worldcountryslope, worldcountryintercept)

