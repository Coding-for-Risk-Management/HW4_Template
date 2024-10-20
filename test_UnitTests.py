from scipy.stats import norm
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf
from sklearn.datasets import load_iris
from rdatasets import data

# qSLR1
def test_MyLM():
    # Unit Tests
    data = load_iris()
    iris = pd.DataFrame(data=data.data, columns=data.feature_names)
    # Rename columns to remove spaces and parentheses
    iris.columns = iris.columns.str.replace(
        ' ', '_').str.replace('(', '').str.replace(')', '')
    # Define the formula with the modified column names
    myFormula = "sepal_length_cm ~ petal_length_cm"
    fit_iris = smf.ols(formula=myFormula, data=iris).fit()

    x = python_hw4.MyLM(myFormula, iris)
    assert np.round(x.params[0], 4) == 4.3066

# qTMat1
def test_TMAT1():
    rLast = np.repeat(['A', 'B', 'C'], [3, 4, 5])
    rNow = np.repeat(['A', 'B', 'C'], [5, 2, 5])

    out = python_hw4.TMAT1(rLast, rNow)
    answer = np.array([1, 0, 0, 0.5, 0.5, 0, 0, 0, 1]).reshape(3, 3)
    assert np.array_equal(out, answer)

# qTMat2
def test_Forecast_nPeriod():
    # Unit Tests:
    initialStates = np.array([[20, 30, 10]])  # Row vector in numpy
    # 3x3 transition matrix
    tmat = np.array([[1, .5, 0], [0, .5, 0], [0, 0, 1]])

    states1 = python_hw4.Forecast_nPeriod(initialStates, tmat, 1)
    states2 = python_hw4.Forecast_nPeriod(initialStates, tmat, 2)
    states3 = python_hw4.Forecast_nPeriod(initialStates, tmat, 3)

    assert np.isclose(states3[0][0], 46.25)

# qTmat2Bonus
def test_Forecast_nPeriod_Recursive():
    # Example vector and matrix setup
    initialStates = np.array([[20, 30, 10]])  # Row vector in numpy
    tmat = np.array([[1, 0.5, 0], [0, 0.5, 0], [0, 0, 1]])  # Transition matrix

    # Unit Tests:
    states1 = python_hw4.Forecast_nPeriod_Recursive(initialStates, tmat, 1)
    states2 = python_hw4.Forecast_nPeriod_Recursive(initialStates, tmat, 2)
    states3 = python_hw4.Forecast_nPeriod_Recursive(initialStates, tmat, 3)
    
    test_result = np.isclose(states3[0][0], 46.25)
    print("Test Passed:", test_result)

# qBondDuration
def test_getBondDuration():
    # Unit tests
    y = 0.03
    face = 2000000
    couponRate = 0.04
    m = 10

    x = python_hw4.getBondDuration(y, face, couponRate, m)
    assert np.isclose(round(x, 2), 8.51)

    x = python_hw4.getBondDuration(y, face, couponRate, m, 1)
    assert np.isclose(round(x, 2), 8.51)

    x = python_hw4.getBondDuration(y, face, couponRate, m, 2)
    assert np.isclose(round(x, 2), 8.42)

# qGetReturnsLag
def test_getReturns():
    # Example data vector
    x = [100, 120, 150, 200]

    # Testing the function with various lags
    rets1 = python_hw4.getReturns(x, 1)
    rets2 = python_hw4.getReturns(x, 2)
    rets3 = python_hw4.getReturns(x, 3)

    # Test assertions
    assert np.isclose(round(rets1[0], 2), 0.20)
    assert np.isclose(round(rets1[1], 2), 0.25)
    assert np.isclose(round(rets1[2], 2), 0.33)
    assert np.isclose(round(rets2[0], 2), 0.50)
    assert np.isclose(round(rets3[0], 2), 1.00)
