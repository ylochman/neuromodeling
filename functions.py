import numpy as np

class FunctionExtended(object):
    def __init__(self, func):
        self.func = func

    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)

    def __add__(self, other):
        def summed(*args, **kwargs):
            return self(*args, **kwargs) + other(*args, **kwargs)
        return FunctionExtended(summed)
        
    def __sub__(self, other):
        def diff(*args, **kwargs):
            return self(*args, **kwargs) - other(*args, **kwargs)
        return FunctionExtended(diff)

    def __div__(self, other):
        def divided(*args, **kwargs):
            return self(*args, **kwargs) / other(*args, **kwargs)
        return FunctionExtended(divided)

    def __mul__(self, other):
        def multiplied(*args, **kwargs):
            return self(*args, **kwargs) * other(*args, **kwargs)
        return FunctionExtended(multiplied)
        
    def constmul(self, alpha):
        def constmultiplied(*args, **kwargs):
            return self(*args, **kwargs) * alpha
        return FunctionExtended(constmultiplied)
    
    def constdiv(self, alpha):
        def constdivided(*args, **kwargs):
            return self(*args, **kwargs) / alpha
        return FunctionExtended(constdivided)
        
    def __matmul__(self, other):
        def composed(*args, **kwargs):
            return self(other(*args, **kwargs))
        return FunctionExtended(composed)
        
    __rmul__ = __mul__


def newton_root(f,Df,x0,epsilon,max_iter,alpha=1):
    '''Approximate solution of f(x)=0 by Newton's method.

    Parameters
    ----------
    f : function
        Function for which we are searching for a solution f(x)=0.
    Df : function
        Derivative of f(x).
    x0 : number
        Initial guess for a solution f(x)=0.
    epsilon : number
        Stopping criteria is abs(f(x)) < epsilon.
    max_iter : integer
        Maximum number of iterations of Newton's method.

    Returns
    -------
    xn : number
        Implement Newton's method: compute the linear approximation
        of f(x) at xn and find x intercept by the formula
            x = xn - f(xn)/Df(xn)
        Continue until abs(f(xn)) < epsilon and return xn.
        If Df(xn) == 0, return None. If the number of iterations
        exceeds max_iter, then return None.

    Examples
    --------
    >>> f = lambda x: x**2 - x - 1
    >>> Df = lambda x: 2*x - 1
    >>> newton(f,Df,1,1e-8,10)
    Found solution after 5 iterations.
    1.618033988749989
    '''
    xn = x0
    for n in range(0,max_iter):
        fxn = f(xn)
        if abs(fxn) < epsilon:
            return xn
        Dfxn = Df(xn)
        if Dfxn == 0:
            return None
        xn = xn - alpha * fxn/Dfxn
    return None

def newton_roots(params_L, params_fast, I, C, x0, eps=0.1, max_iter=1000):
    dV = membrane_potential_derivative(params_L, params_fast, I, C)
    d2V = membrane_potential_second_derivative(params_L, params_fast, I, C)
    x0_true = []
    for x0_ in x0:
        x = newton_root(dV, d2V, x0_, eps, max_iter, alpha=1)
        x0_true.append(x)
    return x0_true

def sigmoid(x):
    return 1. / (1 + np.exp(-x))

def leak_current(params):
    """Returns a leak current as a function of V
    """
    E_x = params['E_x']
    g = params['g']
    return FunctionExtended(lambda V: g * (V - E_x))

def activation(params):
    """Returns an (in)activation as a function of V
    """
    k = params['k']
    V_half = params['V_half']
    return FunctionExtended(lambda V: sigmoid((V - V_half) / k))

def instantaneous_current(params):
    """Returns an instantaneous current as a function of V
    """
    k = params['k']
    V_half = params['V_half']
    E_x = params['E_x']
    g = params['g']
    return FunctionExtended(lambda V: sigmoid((V - V_half) / k) * g * (V - E_x))

def membrane_potential_derivative(params_L, params_fast, I, C):
    I_L = leak_current(params_L)
    I_inst = instantaneous_current(params_fast)
    return FunctionExtended(lambda V: (I - I_L(V) - I_inst(V)) / C)

def membrane_potential_second_derivative(params_L, params_fast, I, C):
    g_L = params_L['g']
    g_x = params_fast['g']
    E_x = params_fast['E_x']
    k = params_fast['k']
    m = activation(params_fast)
    return FunctionExtended(lambda V: - 1 / C * (g_L + g_x * ((m(V) - m(V) ** 2) / k * (V - E_x) + m(V))))