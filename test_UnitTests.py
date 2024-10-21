import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

import HW4_Functions_File

# WhoAmI
def test_WhoAmI():
    assert HW4_Functions_File.WhoAmI() != ''
    
# qSLR1
def test_MyLM():
    hardcoded_iris = {
      'sepal_length_cm': [5.1, 4.9, 4.7, 4.6, 5.0],
      'petal_length_cm': [1.4, 1.4, 1.3, 1.5, 1.4]
    }
    hardcoded_df = pd.DataFrame(hardcoded_iris)
    myFormula = "sepal_length_cm ~ petal_length_cm"
    x = HW4_Functions_File.MyLM(myFormula, hardcoded_df)
    assert np.round(x.params[0], 2) == 5.56

# qTMat1
def test_TMAT1():
    rLast = np.repeat(['A', 'B', 'C'], [3, 4, 5])
    rNow = np.repeat(['A', 'B', 'C'], [5, 2, 5])

    out = HW4_Functions_File.TMAT1(rLast, rNow)
    answer = np.array([1, 0, 0, 0.5, 0.5, 0, 0, 0, 1]).reshape(3, 3)
    assert np.array_equal(out, answer)

# qTMat2
def test_Forecast_nPeriod():
    
    # Unit Tests:
    initialStates = np.array([[20, 30, 10]])  # Row vector in numpy
    # 3x3 transition matrix
    tmat = np.array([[1, .5, 0], [0, .5, 0], [0, 0, 1]])

    states1 = HW4_Functions_File.Forecast_nPeriod(initialStates, tmat, 1)
    states2 = HW4_Functions_File.Forecast_nPeriod(initialStates, tmat, 2)
    states3 = HW4_Functions_File.Forecast_nPeriod(initialStates, tmat, 3)

    assert np.isclose(states3[0][0], 46.25)

# qTmat2Bonus
def test_Forecast_nPeriod_Recursive():
    # Example vector and matrix setup
    initialStates = np.array([[20, 30, 10]])  # Row vector in numpy
    tmat = np.array([[1, 0.5, 0], [0, 0.5, 0], [0, 0, 1]])  # Transition matrix

    # Unit Tests:
    states1 = HW4_Functions_File.Forecast_nPeriod_Recursive(initialStates, tmat, 1)
    states2 = HW4_Functions_File.Forecast_nPeriod_Recursive(initialStates, tmat, 2)
    states3 = HW4_Functions_File.Forecast_nPeriod_Recursive(initialStates, tmat, 3)
    
    test_result = np.isclose(states3[0][0], 46.25)

# qGetReturnsLag
def test_getReturns():
    # Example data vector
    x = [100, 120, 150, 200]

    # Testing the function with various lags
    rets1 = HW4_Functions_File.getReturns(x, 1)
    rets2 = HW4_Functions_File.getReturns(x, 2)
    rets3 = HW4_Functions_File.getReturns(x, 3)

    # Test assertions
    assert np.isclose(round(rets1[0], 2), 0.20)
    assert np.isclose(round(rets1[1], 2), 0.25)
    assert np.isclose(round(rets1[2], 2), 0.33)
    assert np.isclose(round(rets2[0], 2), 0.50)
    assert np.isclose(round(rets3[0], 2), 1.00)
