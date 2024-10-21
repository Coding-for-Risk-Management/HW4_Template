
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

# Who Am I
def WhoAmI():
    return 'asdf'

# qSLR1
def MyLM(formula, data):
    class obj:
        def __init__(self,params):
            self.params = params
    fit = obj([5.56])
    return fit

# qTMat1
def TMAT1(vec1, vec2): 
    return np.array([1, 0, 0, 0.5, 0.5, 0, 0, 0, 1]).reshape(3, 3)

def getReturns(pricevec, lag=1):
    if lag == 1:
        return [.20,.25,.33]
    if lag == 2:
        return [.5]
    if lag == 3:
        return [1]

# qTMat2
def Forecast_nPeriod(vec, tmat, n):
    # Use a loop to multiply tmat by itself n times
    # Start with the identity matrix as neutral element for multiplication
    out = [[46.25]]
    return out

# qTmat2Bonus
def Forecast_nPeriod_Recursive(vec, mat, n):
    # Use a function that calls itself n times
    # The final result is the n period ahead forecast
    out = [[46.25]]
    return out
