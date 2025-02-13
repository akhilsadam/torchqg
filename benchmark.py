import sys
import math

import torch
import torch.nn as nn

import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from qg import to_spectral, to_physical, QgModel
from sgs import Constant # MLdiv

import os
from tqdm import tqdm

plt.rcParams.update({'mathtext.fontset':'cm'})

# A framework for the evaluation of turbulence closures used in mesoscale ocean large-eddy simulations.
# Graham and Ringler (2013).

def t_unit():
  return 1.2e6

def l_unit():
  return (504e4 / math.pi)

Lx = 2*math.pi
Ly = 2*math.pi
Nx = 512
Ny = 512

iterations = 100
dt = 120 / t_unit() # 480s
mu = 1.25e-8 / l_unit()**(-1) # 1.25e-8m^-1
nu = 352 / l_unit()**2 / t_unit()**(-1) # 22m^2s^-1 for the simulation (2048^2)

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
h.init_randn(0.01, [3.0, 5.0])
# Set up spectral filter kernel.
h.kernel = h.grid.cutoff

print(h)

def workflow(
  dir,
  iters, 
  steps,
  scale,
  system,
):
  """
  Args:
    steps int: Total number of steps that will be stored
  """
  t0 = system.pde.cur.t
  store_les = int(iters / steps) # Data will be stored every <store_les> time steps
  store_dns = store_les * scale

  Nx = system.grid.Nx
  Ny = system.grid.Ny
  
  # DNS
  dns  = torch.zeros([steps, 4, Ny,  Nx ], dtype=torch.float64)
  time = torch.zeros([steps])

  def visitor_dns(m, cur, it):
    # High res
    if it % store_dns == 0:
      i = int(it / store_dns)
      q, p, u, v = m.update()

      dns[i] = torch.stack((q, p, u, v))

      # step time
      time[i] = cur.t - t0
    return None

  with torch.no_grad():
    for it in tqdm(range(iters * scale)):
      system.pde.step(system)
      visitor_dns(system, system.pde.cur, it)
      
  os.makedirs(dir, exist_ok=True)
  torch.save(dns, os.path.join(dir, 'dns_0.pt'))
  
workflow(
  dir='bench',
  iters=iterations,
  steps=100,
  scale=4, # 4x more DNS steps than LES steps
  system=h
)
