import torch
torch.manual_seed(0)

from solver import PsuedoSpectralSolver, to_spectral

import os
from tqdm import tqdm
from time import time

from iox.default import params
  
from matplotlib import pyplot as plt
import numpy as np

# @torch.compile
def solve(params, dir, iterations, scale, save=False, **kwargs):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  system = PsuedoSpectralSolver(**params).to(device)
  print(system)
   
  Nx = system.grid.Nx
  Ny = system.grid.Ny
  dns  = torch.zeros([iterations+1, system.n_outputs, Ny,  Nx ], dtype=torch.float64)
  
  with torch.no_grad():
    for it in tqdm(range(iterations * scale)):
      system.pde.step(system)
      
      if it % scale == 0:
        i = int(it / scale)
        dns[i] = torch.stack(list(system.update())) # q, p, u, v

  dns[-1] = torch.stack(list(system.update()))
  # os.makedirs(dir, exist_ok=True)
  # if save:
  #   print('Saving reference dns_0.pt...')
  #   torch.save(dns, os.path.join(dir, 'dns_0.pt'))
  return dns

 
 # Wind stress forcing (becomes vorticity source term).
def curl(fx,fy):
  # shape ...xy 
  dfy_dx = torch.gradient(fy, dim = -1)[0]
  dfx_dy = torch.gradient(fx, dim = -2)[0]
  return dfy_dx - dfx_dy

# @torch.compile
def source(i, sol, dt, t, grid):
  fx = -(1/(2*torch.pi)) * torch.cos(grid.y).view(grid.Ny, 1) + (0*grid.x).view(1, grid.Nx)
  fy = (0*grid.y).view(grid.Ny, 1) + (0 * grid.x).view(1, grid.Nx)
  yh = to_spectral(curl(fx,fy))
  return yh 

def test_dqg(vars, save=False, path='out/deepak-qg/'):
  i, Nx, T, dt, nu = vars
  path += f'{i}/'
  
  N = int(T/dt)
  plot_freq = N//20
  params.update({
  "Nx": Nx,
  "Ny": Nx,
  "iterations": int(round(N/plot_freq)), # only plot final state
  "scale": plot_freq, #,
  "dt": dt,  # 480s
  "t0": 0.0, # Initial time
  "B" : 0.0, # Planetary vorticity y-gradient
  "mu": 0.0,  # bottom friction term
  "nu": nu,
  "init": lambda f : f.init_randn(0.01, [3.0, 5.0]) # 0-start
  })  
  
  params.update({
    "eta": torch.zeros([params['Nx'],params['Ny']], dtype=torch.float64, requires_grad=True), # Topographic potential vorticity
    "source": source, # Source term
  })
  
  
  # High res model.
  time_start = time()
  dns_dev = solve(params, dir='bench', save=save, **params)
  time_end = time()
  _dt = time_end - time_start
  print(f'vars: {vars}')
  print('Time taken:', _dt)
  
  os.makedirs(path, exist_ok=True)

  for j in range(dns_dev.shape[0]):
  
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    data = dns_dev[j].cpu().numpy()
    uvmax = max(np.abs(data[2]).max(), np.abs(data[3]).max())
    for i, ax in enumerate(axs.flat):
      di = data[i]
      if i==0:
        dvdx = np.gradient(data[3], axis = -1)
        dudy = np.gradient(data[2], axis = -2)
        di = dvdx - dudy
      if i > 1:
        ax.imshow(di, vmin=-uvmax, vmax=uvmax)
      else:
        ax.imshow(di)
      ax.set_title(['w', 'p', 'u', 'v'][i])
    plt.tight_layout()
    os.makedirs(path, exist_ok=True)
    plt.savefig(path+f'bench_qg_{j}.png')
    plt.close()
    
  fig, ax = plt.subplots()
  dvdx = torch.gradient(dns_dev[-1, -1], dim = -1)[0]
  dudy = torch.gradient(dns_dev[-1, -2], dim = -2)[0]
  w = dvdx - dudy
  ax.imshow(w.cpu().numpy())
  ax.set_title('w')
  plt.tight_layout()
  plt.savefig(path + 'bench_qgw.png')
  plt.close()
  
  return _dt

if __name__ == '__main__':
  
  runs = [
  # [0, 32, 40, 1e-2, 1e-5],
  # [1, 64, 40, 1e-2, 1e-5],
  # [2, 128, 40, 1e-2, 1e-5],
  # [2, 256, 40, 1e-2, 1e-5],
  [3, 512, 40, 1e-2, 1e-5],
  [4, 1024, 40, 1e-2, 1e-5], 
  [5, 2048, 40, 1e-2, 1e-5], 
  ]
  
  dts = [test_dqg(run) for run in runs]
  print(dts)