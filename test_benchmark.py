import torch
torch.manual_seed(0)

from solver import PsuedoSpectralSolver

import os
from tqdm import tqdm

from iox.default import params

def test_benchmark(save=False):
  # High res model.
  system = PsuedoSpectralSolver(**params)

  def solve(system, dir, iterations, scale, **kwargs):
    Nx = system.grid.Nx
    Ny = system.grid.Ny
    dns  = torch.zeros([iterations, system.n_outputs, Ny,  Nx ], dtype=torch.float64)
    
    with torch.no_grad():
      for it in tqdm(range(iterations * scale)):
        system.pde.step(system)
        
        if it % scale == 0:
          i = int(it / scale)
          dns[i] = torch.stack(list(system.update())) # q, p, u, v

    os.makedirs(dir, exist_ok=True)
    if save:
      print('Saving reference dns_0.pt...')
      torch.save(dns, os.path.join(dir, 'dns_0.pt'))
    return dns

  print(system)
  dns_dev = solve(system, dir='bench', **params)

  # compare output to reference dns_0.pt
  ref = torch.load('bench/dns_0.pt', weights_only=True)
  print('MSE:', torch.nn.functional.mse_loss(dns_dev, ref))
  assert torch.allclose(dns_dev, ref, atol=1e-20), f'Benchmark failed'


if __name__ == '__main__':
  test_benchmark(True)