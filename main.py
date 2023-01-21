import sys
import math
import h5py

import torch
import torch.nn as nn

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from torchqg.qg import to_spectral, to_physical, QgModel
from torchqg.sgs import Constant # MLdiv

import torchqg.workflow as workflow

plt.rcParams.update({'mathtext.fontset':'cm'})

# A framework for the evaluation of turbulence closures used in mesoscale ocean large-eddy simulations.
# Graham and Ringler (2013).

def t_unit():
  # T_d(x) = 1.2 x 10^6/pi s
  return 1.2e6

def l_unit():
  # L_d(x) = 504 x 10^4/pi m
  return (504e4 / math.pi) 

Lx = 2*math.pi
Ly = 2*math.pi
Nx = 512
Ny = 512

dt = 480 / t_unit() # 480s
mu = 1.25e-8 / l_unit()**(-1) # 1.25e-8m^-1
nu = 352 / l_unit()**2 / t_unit()**(-1) # 352m^2/s for (512^2) and 22m^2s^-1 for (2048^2)

# Wind stress forcing.
def Fs(i, sol, dt, t, grid):
  phi_x = math.pi * math.sin(1.2e-6 / t_unit()**(-1) * t)
  phi_y = math.pi * math.sin(1.2e-6 * math.pi / t_unit()**(-1) * t / 3)
  y = torch.cos(4 * grid.y + phi_y).view(grid.Ny, 1) - torch.cos(4 * grid.x + phi_x).view(1, grid.Nx)

  yh = to_spectral(y)
  K = torch.sqrt(grid.krsq)
  yh[K < 3.0] = 0
  yh[K > 5.0] = 0
  yh[0, 0] = 0

  e0 = 1.75e-18 / t_unit()**(-3)
  ei = 0.5 * grid.int_sq(yh) / (grid.Lx * grid.Ly)
  yh *= torch.sqrt(e0 / ei)
  return yh

eta = torch.zeros([Ny, Nx], dtype=torch.float64, requires_grad=True)

# High res model.
h = QgModel(
  name='\\mathcal{F}',
  Nx=Nx,
  Ny=Ny,
  Lx=Lx,
  Ly=Ly,
  dt=dt,
  t0=0.0,
  B=0.0,    # Planetary vorticity y-gradient
  mu=mu,    # Linear drag
  nu=nu,    # Viscosity coefficient
  nv=1,     # Hyperviscous order (nv=1 is viscosity)
  eta=eta,  # Topographic PV
  source=Fs # Source term
)

# Initial conditions.
init_data_path = 'output/spinup_10000_dump' # None
iters = 100
data_store_name = 'warmed_'+str(iters) # geo_'+str(iters) # 'warmed_100'
if init_data_path is not None:
  filename = init_data_path + '.h5'
  hf = h5py.File(filename, 'r')
  ft = hf.get('time')[:]
  fq = hf.get('dns_q')[:]
  
  fq = torch.from_numpy(fq[-1])
  fqh = to_spectral(fq)

  h.pde.cur.t = ft[-1]
  h.pde.sol = fqh
else:
  h.init_randn(0.01, [3.0, 5.0])


# Set up spectral filter kernel.
h.kernel = h.grid.cutoff

print(h)
# Low res model(s).
scale = 4

Nxl = int(Nx / scale)
Nyl = int(Ny / scale)

eta_m = torch.zeros([Nyl, Nxl], dtype=torch.float64, requires_grad=True)

# Null-parametrization model.
m1 = QgModel(
  name='',
  Nx=Nxl,
  Ny=Nyl,
  Lx=Lx,
  Ly=Ly,
  dt=dt,
  t0=0.0,
  B=0.0,     # Planetary vorticity y-gradient
  mu=mu,     # Linear drag
  nu=nu,     # Viscosity coefficient
  nv=1,      # Hyperviscous order (nv=1 is viscosity)
  eta=eta_m, # Topographic PV
  source=Fs, # Source term
  sgs=Constant(c=0.0) # Subgrid-scale term (replace with yours)
)

# Initialize from DNS vorticity field.
m1.pde.sol = h.filter(m1.grid, scale, h.pde.sol)
print('LES Model: ', m1)

# Will produce two images in folder `output` with the final fields after <iters> iterations.
workflow.workflow(
  dir='output/',
  name=data_store_name,
  iters=iters,# 10000,  # Model iterations; 6.15min/1K iters on 1CPU
  steps=100,    # Discrete steps
  scale=scale,  # Kernel scale
  diags=[       # Diagnostics
    workflow.diag_fields,
  ],
  system=h,       # DNS system
  # models=[],
  models=[m1],     # LES without model
  dump=True
)
