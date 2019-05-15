from sympy import Symbol, Function, Eq
from sympy import diff, solve, plot, init_printing, init_session, pretty
init_printing(use_unicode=True)
import numpy as np
import matplotlib as mpl
mpl.use('TkAgg')
import seaborn as sns
sns.set_style('darkgrid')
from matplotlib import pyplot as plt

from utils import *

class OneDimensionalSystem():
    def __init__(self, fn, x, r):
        self.fn = fn
        self.x = x
        self.r = r
        self.dx = Function("x'")
        self.dfn = diff(self.fn, self.x)
        self.d2fn = diff(self.dfn, self.x)
        self.roots = solve(self.fn, self.x)
        self.xlim = (-10, 10)
        self.ylim = (0, 100)
        self.rcParams = (15, 5)

    def display_fn(self, notebook=True):
        """Display function fn(x, r) which is essentially dx/dt with a single parameter r
        """
        # display(Eq(self.dx, self.fn))
        if notebook:
            display(Eq(self.dx, self.fn))
        else:
            print(pretty(Eq(self.dx, self.fn)))
    
    def get_real_roots(self, r_value):
        """Get zero roots of fn(x) (for fixed r value)
        Return: {root_1: stability, ..., root_n: stability}
                stability: 0 (unstable)
                           1 (stable)
                           2 (right half-stable)
                           3 (left half-stable)
        """
        real_roots = dict()
        fn = self.fn.subs([(self.r, r_value)])
        dfn = self.dfn.subs([(self.r, r_value)])
        d2fn = self.d2fn.subs([(self.r, r_value)])
        all_roots = solve(fn, self.x)
        for root in all_roots:
            if complex(root).imag == 0:
                stability = 0 if dfn.subs([(self.x, float(root))]) > 0 else \
                            1 if dfn.subs([(self.x, float(root))]) < 0 else \
                            2 if d2fn.subs([(self.x, float(root))]) < 0 else 3
                real_roots[float(root)] = stability
        return real_roots
    
    def plot_phase_diagrams(self, r_values=[-5, 0, 5]):
        mpl.rcParams['figure.figsize'] = self.rcParams
        
        fig, axes = plt.subplots(1, len(r_values))
        for (r_value, ax)  in zip(r_values, axes):
            roots = self.get_real_roots(r_value)
            p = plot(self.fn.subs([(self.r, r_value)]), (self.x, *self.xlim), show=False)
            move_sympyplot_to_axes(p, ax)
            for root in roots:
                plot_equilibrium(root, roots[root], axis=ax)
            plot_axis(self.xlim, self.ylim, xlabel=None, ylabel=None, axis=ax)
        plt.show()
        
    def plot_bifurcation_diagram(self, rlim=(-5, 5)):
        p = plot(self.roots[0], (self.r, *rlim), show=False)
        for root in self.roots[1:]:
            p.extend(plot(root, (self.r, *rlim), show=False))
        p.show()

def example():
    t = Symbol('t')
    x = Symbol('x')
    r = Symbol('r')

    f = r + x**2

    ODS = OneDimensionalSystem(f, x, r)
    ODS.display_fn(notebook=False)
    ODS.plot_phase_diagrams(r_values=[-10, 0, 10])
    ODS.plot_bifurcation_diagram()
    
        
if __name__ == "__main__":
    init_printing()
    example()
