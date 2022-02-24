import numpy as np
from scipy.integrate import odeint
from scipy.interpolate import InterpolatedUnivariateSpline as spline
from scipy.optimize import curve_fit
from os import listdir
import matplotlib.pyplot as plt
from sklearn.metrics import  r2_score

def function(t, eta, km, k1, k2,w): 
    x = odeint(model,1E-5,t, args=(eta, km, k1, k2,w))
    return x[:,0]

def model(x,t,eta, km, k1, k2,w):
    x_thld = w*(1+k1/k2)
    K = (k1*k2/(k1+k2))
    if (x <= x_thld):
        return (1/eta)*(force(t)-km*x-K*x)
    else:
        return (1/eta)*(force(t)-km*x-k2*(x-w))

#p0 = (.01, 4, 1.5, 5, 1)

pwd = '../RawData/'
directories = listdir(pwd)
for dir in directories:
    files = listdir(pwd + dir)
    pars = []
    for file in files:
        data = np.loadtxt(pwd + dir + '/' + file)
        t, f, x = data[::200,0]/1000, data[::200,1], data[::200,2]
        i_max = f.argmax()
        t, f, x = t[:i_max], f[:i_max], x[:i_max]
        x = x[0] - x
        if len(t) > 3:
            force = spline(t, f, k=3)
            popt, pcov = curve_fit(function, t, x, bounds=((0.6, 3, 1.5, 3., 0.2), (6, 30, 15, 30, 2)))
            r2 = r2_score(x,function(t,*popt))
            if r2 > .75:
                pars.append(popt)
    np.savetxt('../Results/' + dir + '.txt', pars)