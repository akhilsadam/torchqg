import torch
torch.manual_seed(0)

from solver import PsuedoSpectralSolver, to_spectral, physical_curl

import os
from tqdm import tqdm
from time import time
from matplotlib import pyplot as plt
import numpy as np
import jpcm

from iox.default import params


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def solve(params):

  system = PsuedoSpectralSolver(**params).to(device)
  print(system)
   
  iterations = params['iterations']
  scale = params['scale']
   
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
  
  return dns

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
  "eta_penalty": 0.1,
  "init": lambda f : f.init_randn_persist((0.01, [3.0, 5.0])) # 0-start
  })  

  mask = torch.zeros([params['Nx'],params['Ny']], dtype=torch.float64, requires_grad=False).to(device)
  if i > 5:
    # mask[params['Nx']//4:3*params['Nx']//4, params['Ny']//4:3*params['Ny']//4] = 1 
    x_mask = torch.linspace(-1, 1, params['Nx'])
    y_mask = torch.linspace(-1, 1, params['Ny'])
    mask = torch.stack(torch.meshgrid(x_mask, y_mask))
    mask = (mask[0]**2 + (mask[1]-0.5)**2 < 0.07).to(torch.float64)
    mask = mask.to(device) 
  
  
  def source(i, sol, dt, t, grid):
    fx = -(50/(2*torch.pi)) * torch.cos(grid.y).view(grid.Ny, 1) + (0*grid.x).view(1, grid.Nx)
    fy = (0*grid.y).view(grid.Ny, 1) + (0 * grid.x).view(1, grid.Nx)
    yh = to_spectral(physical_curl(fx,fy) * (1-mask))
    return yh 

  params.update({
    "eta": torch.zeros([params['Nx'],params['Ny']], dtype=torch.float64, requires_grad=True), # Topographic potential vorticity
    "source": source, # Source term
    "mask": lambda *args: mask,
  })
  
  # High res model.
  time_start = time()
  dns_dev = solve(params)
  time_end = time()
  _dt = time_end - time_start
  print(f'vars: {vars}')
  print('Time taken:', _dt)
  
  os.makedirs(path, exist_ok=True)

  for j in range(dns_dev.shape[0]):
  
    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    data = dns_dev[j].cpu().numpy()
    uvmax = max(np.abs(data[2]).max(), np.abs(data[3]).max())
    # wmax = np.abs(data[0]).max()
    pmax = np.abs(data[1]).max()
    for i, ax in enumerate(axs.flat):
      di = data[i]
      if i > 1:
        ax.imshow(di, vmin=-uvmax, vmax=uvmax, cmap='seismic')
      elif i==0:
        wmax = np.abs(di).max()
        ax.imshow(di, vmin=-wmax, vmax=wmax, cmap='seismic') # ,
      else:
        ax.imshow(di, vmin=-pmax, vmax=pmax, cmap='seismic')
        # quiver plot
        dy = data[-1]
        dx = data[-2]
        _Nx = params['Nx']
        _Ny = params['Ny']
        res=64
        ax.quiver(np.arange(0, _Nx, _Nx//res), np.arange(0, _Ny, _Ny//res), dx[::_Nx//res, ::_Ny//res], dy[::_Nx//res, ::_Ny//res], scale=32)
      ax.set_title(['w', 'streamfunction', 'u', 'v'][i])
    plt.tight_layout()
    os.makedirs(path, exist_ok=True)
    plt.savefig(path+f'bench_qg_{j}.png')
    plt.close()
  
  return _dt


if __name__ == '__main__':
  
  runs = [
  # [0, 32, 40, 1e-2, 1e-5],
  # [1, 64, 40, 1e-2, 1e-5],
  # [2, 128, 40, 1e-2, 1e-5],
  # [2, 256, 40, 1e-2, 1e-5],
  # [3, 512, 40, 1e-2, 1e-5], # no mask
  # [4, 1024, 40, 1e-2, 1e-5], 
  # [5, 2048, 40, 1e-2, 1e-5],  
  ################# begin masking
  # [6, 512, 40, 1e-2, 1e-5], # square mask
  # [7, 512, 40, 1e-3, 1e-5], # square mask
  # [8, 512, 40, 1e-2, 1e-5], # square mask
  [9, 512, 40, 1e-3, 1e-2], # round mask
  [10, 512, 40, 1e-3, 1e-5], # round mask
  
  ]
  
  dts = [test_dqg(run) for run in runs]
  print(dts)