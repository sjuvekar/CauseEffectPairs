from sklearn.gaussian_process import GaussianProcess
import numpy as np
import independent_component as indcomp
from scipy.stats import pearsonr

def gaussian_fit_likelihood(x, y):
    nrow = len(x)
    # Add random noise to data to remove duplicates
    newX = np.array(x + np.random.rand(nrow) / 100)
    newY = np.array(y + np.random.rand(nrow) / 100)
    newX = newX.reshape(nrow, 1)
    newY = newY.reshape(nrow, 1)
    ret = [-1.0, -1.0, 0.0, 0.0]
    try:
      g = GaussianProcess()
      g.fit(newX, newY)
      ret[0] = g.reduced_likelihood_function_value_
      err = y - g.predict(newX)
      p = pearsonr(err, x)
      ret[2] = p[0]
      ret[3] = p[1]
      g.fit(newY, newX)
      ret[1] = g.reduced_likelihood_function_value_
      ind_ret = indcomp.independent_component(x, y)
      ret = ret + ind_ret
    except:
      return [-1.0, -1.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0]
    return ret
