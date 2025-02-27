import torch
torch.manual_seed(0)

from solver import PsuedoSpectralSolver, to_spectral, physical_curl

import os
from tqdm import tqdm
from time import time
from matplotlib import pyplot as plt
import numpy as np

from iox.default import params


def solve(params):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

def test_benchmark(save=False, _dir='bench'):
  # High res model.
  time_start = time()
  dns_dev = solve(params, save=save, **params)
  time_end = time()
  _dt = time_end - time_start
  print(f'vars: {vars}')
  print('Time taken:', _dt)
  
  if save:
    os.makedirs(dir=_dir, exist_ok=True)
    print('Saving reference dns_0.pt...')
    torch.save(dns_dev, os.path.join(_dir, 'dns_0.pt'))
  
  else:
    # compare output to reference dns_0.pt
    ref = torch.load('bench/dns_0.pt', weights_only=True)
    print('MSE:', torch.nn.functional.mse_loss(dns_dev, ref))
    assert torch.allclose(dns_dev, ref, atol=1e-20), f'Benchmark failed'


if __name__ == '__main__':
  test_benchmark(True)
