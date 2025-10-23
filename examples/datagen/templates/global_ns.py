from numpy import pi, sin, cos, exp, sqrt, log, arctan2, max, array, linspace, conj, transpose
from petram.helper.variables import variable

import warnings
#warnings.simplefilter("error")
warnings.simplefilter("default")

# element order
order = 4

# profile smoothing
sig_diff = 0.00001

# constants
freq = 54e6
ny = 0.0
w = freq * 2 * pi           # omega
ky = ny*w/3e8
kz = 2*pi/0.8

Zi = 1
Zim = 1
Ai = 2
Aim = 1
        # T
temp = 3000 # eV
imp_frac = 0.0

dner = 0.4e20
nemax = 1e18
nemin = 0.0001

dter = 450
temax = 50
temin = 1

B0 = 5.3
R0 = 6.2

q0 = 1.60217662e-19
e0 = 8.8541878176e-12
mass = 9.1e-31

@variable.jit.float
def dens_jit(x):
    ne = dner* (-x) + nemin if x <= 0 else nemin #[m-3]
    if ne > nemax: 
       ne = nemax
    return ne

@variable.float(dependency=('denss',))
def dens_smooth(x, denss=None):
    return denss

@variable.jit.float
def temp_jit(x):
    temp = dter * (-x) + temin if x <= 0 else temin
    if temp > temax: 
       temp  = temax
    return temp

from numba import njit
@njit("float64[:](float64)")
def mag_jit_f(x):
    B0R0 = B0*R0
    R = 8.31 + x
    mag = B0R0/R
    return array([0, 0., mag])

@variable.jit.float
def mag_jit(x):
    return mag_jit_f(x)[2]

@variable.float(dependency=('denss', 'temps'))
def eld(x, denss=None, temps=None):
    dens = denss
    temp = temps
    wpx2 = dens * q0**2/(mass*e0)
    vth = np.sqrt(2*temp*q0/mass)
    z0 = w**2/vth**2/kz**2
    return 2*sqrt(pi)*wpx2*w/(kz*vth)**3 * exp(-z0)

from petram.phys.nlj1d import simple_jacB_1D, simple_hessB_1D
mag_jac = simple_jacB_1D(mag_jit_f)
mag_hess = simple_hessB_1D(mag_jit_f)
