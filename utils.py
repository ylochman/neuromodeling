from functions import *
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt
import seaborn as sns
sns.set_style('darkgrid')
from scipy.integrate import odeint

def is_stable(x, f, eps=0.1):
    if f(x-eps) > 0 and f(x+eps) < 0:
        return True
    else:
        return False

def plot_fn(f, xlim, label=None):
    """
    xlim = (x_min, x_max)
    """
    x = np.linspace(*xlim, 1000)
    y = list(map(f, x))
    plt.plot(x, y, label=label)

def plot_axis(xlim=(-500,500), ylim=(-500,500), xlabel='V, mV', ylabel='I, A'):
    if xlim[0] <= 0  and xlim[1] >= 0:
        plt.vlines(0, *ylim, 'gray', '-')
    if ylim[0] <= 0  and ylim[1] >= 0:
        plt.hlines(0, *xlim, 'gray', '-')
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel(ylabel, fontsize=15)

def plot_IV_leak_current(params, xlim, ylim, color='C0'):
    plot_fn(leak_current(params), xlim)
    plt.vlines(params['E_x'], *ylim, color, '--', alpha=0.5)
    plt.title('I-V curve for a leak current', fontsize=15)

def plot_IV_instantaneous_current(params, xlim, ylim, color='C0'):
    plot_fn(instantaneous_current(params), xlim)
    plot_axis(xlim, ylim)
    plt.vlines(params['V_half'], *ylim, 'gray', '--', alpha=0.5)
    if params['E_x'] > params['V_half']:
        plt.vlines(params['E_x'], *ylim, color, '--', alpha=0.5)
    plt.title('I-V curve for an instantaneous current', fontsize=15)
    
def plot_activation(params, xlim, ylim, color='C0'):
    plot_fn(activation(params), xlim)
    plot_axis(xlim, (0,1), ylabel='activation m')
    plt.title('Activation function', fontsize=15)

stability_type = {
0: 'nostability',
1: 'monostability',
2: 'bistability'
}

def plot_phase_diagram(params_L, params_fast, I, C, xlim, ylim, x0=None, eps=0.1, max_iter=1000, arrowlen=3):
    dV = membrane_potential_derivative(params_L, params_fast, I, C)
    plot_fn(dV , xlim, 'One-dimensional system')
    plot_axis(xlim, ylim, xlabel='membrane potential, V (mV)', ylabel='derivative of membrane potential, V (mV/ms)')
    if x0 is not None:
        x0_true = newton_roots(params_L, params_fast, I, C, x0, eps, max_iter)
        num_stable = 0
        for x in x0_true:
            stable = is_stable(x, dV, eps=0.1)
            plt.scatter(x, 0, color='k' if stable else 'w', zorder=100, edgecolors='k')
            if not stable:
                plt.arrow(x,0,arrowlen,0,head_width=3,head_length=1, fc='k', ec='k',zorder=90)
                plt.arrow(x,0,-arrowlen,0,head_width=3,head_length=1, fc='k', ec='k',zorder=90)
            else:
                plt.arrow(x-arrowlen-1.5,0,arrowlen,0,head_width=3,head_length=1, fc='k', ec='k',zorder=90)
                plt.arrow(x+arrowlen+1.5,0,-arrowlen,0,head_width=3,head_length=1, fc='k', ec='k',zorder=90)
                num_stable += 1
        plt.title('Phase diagram ({}, I={})'.format(stability_type[num_stable], I), fontsize=15)
    else:
        plt.title('Phase diagram (I={})'.format(I), fontsize=15)
            

def plot_trajectories(params_L, params_fast, I, C, x0=None, tlim=(0,14), xlim=(-100,100), num=500):
    dV = membrane_potential_derivative(params_L, params_fast, I, C)
    t = np.linspace(*tlim, num=num)
    for V0 in range(*xlim):
        solution = odeint(lambda V, t: dV(V), V0, t)
        plt.plot(t, solution, c='C0')
    y = tlim[-1]
    arrowlen = 4
    plot_axis(tlim, xlim, xlabel='time (ms)', ylabel='membrane potential, V (mV)')
    if x0 is not None:
        x0_true = newton_roots(params_L, params_fast, I, C, x0)
        num_stable = 0
        for x in x0_true:
            stable = is_stable(x, dV, eps=0.1)
            plt.scatter(y,x, color='k' if stable else 'w', zorder=100, edgecolors='k', s=75)
            plt.vlines(y,*xlim,color='k')
            if not stable:
                plt.arrow(y,x,0,arrowlen,head_width=0.2,head_length=3, fc='k', ec='k',zorder=90)
                plt.arrow(y,x,0,-arrowlen,head_width=0.2,head_length=3, fc='k', ec='k',zorder=90)
            else:
                plt.arrow(y,x-arrowlen-4,0,arrowlen,head_width=0.2,head_length=3, fc='k', ec='k',zorder=90)
                plt.arrow(y,x+arrowlen+4,0,-arrowlen,head_width=0.2,head_length=3, fc='k', ec='k',zorder=90)
                num_stable += 1
        plt.title('Trajectories ({}, I={})'.format(stability_type[num_stable], I), fontsize=15)
    else:
        plt.title('Trajectories (I={})'.format(I), fontsize=15)
