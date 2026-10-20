"""Portable logistic inference for existing shot models."""
import math
import numpy as np


def estimate(model, features):
    row = np.asarray([features[name] for name in model['features']])
    z = float(((row-np.asarray(model['mean']))/np.asarray(model['scale'])) @ np.asarray(model['weights']) + model['bias'])
    return 1/(1+math.exp(-max(-40, min(40, z))))
