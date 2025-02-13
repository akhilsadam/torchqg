import math
import torch

from solver import to_spectral


def t_unit():
  return 1.2e6

def l_unit():
  return (504e4 / math.pi)

# Wind stress forcing (becomes vorticity source term).
def source(i, sol, dt, t, grid):
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

params = {
  "Lx": 2 * math.pi,
  "Ly": 2 * math.pi,
  "Nx": 512,
  "Ny": 512,
  "iterations": 2,
  "scale": 200,
  "dt": 120 / t_unit(),  # 480s
  "t0": 0.0, # Initial time
  "B" : 0.0, # Planetary vorticity y-gradient
  "mu": 1.25e-8 / l_unit()**(-1),  # 1.25e-8m^-1
  "nu": 352 / l_unit()**2 / t_unit()**(-1),  # 22m^2s^-1 for the simulation (2048^2)
  "nv": 1, # Hyperviscous order (nv=1 is viscosity)
  "eta": torch.zeros([512, 512], dtype=torch.float64, requires_grad=True), # Topographic potential vorticity
  "source": source, # Source term
  "init": lambda f : f.init_randn(0.01, [3.0, 5.0])
}


