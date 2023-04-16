import numpy as np
from scipy import integrate, interpolate
from scipy import optimize

ar = open("SocialMediaUsers.csv").read().split("\n")[1:]
ar = ar[:len(ar)-1]

twitter = []
facebook = []
snapchat = []
xSnapchat = []
c = 0

for i,a in enumerate(ar):
  arI = a.split(",")[1:]
  twitter.append(float(arI[0]))
  facebook.append(float(arI[1]))

  if arI[2] != "0":
      xSnapchat.append(c)
      snapchat.append(float(arI[2]))
      c = c + 1

x = np.arange(0, len(ar))

x_data = x
y_data = np.array(facebook)

def f(y, t, k):
  
    s = y[0]
    i = y[1]
    r = y[2]

    beta = k[0]
    gamma = k[1]
    delta = k[2]
    
    dsdt = -beta * s * i / (s + i + r) + delta * r / (s + i + r)
    didt = beta * s * i / (s + i + r) - gamma * i / (s + i + r)
    drdt = gamma * i / (s + i + r) - delta * r / (s + i + r)
    
    return [dsdt, didt, drdt]

def SolvedFunc(x, paramAr):
    funcLambda = lambda y, t: f(y, t, paramAr)
    solved = integrate.odeint(funcLambda ,y0 ,x)
    return solved[:,1]

def Loss(p):
    return y_data - SolvedFunc(x_data,p)
  

guess = [0,0,0]
y0 = [0,30,0]

(params,kvg) = optimize.leastsq(Loss, guess, args=(),
                           Dfun=None, full_output=0, col_deriv=0,
                           ftol=1.49012e-08, xtol=1.49012e-08, gtol=0.0,
                           maxfev=0, epsfcn=None, factor=100, diag=None)

print(params)
