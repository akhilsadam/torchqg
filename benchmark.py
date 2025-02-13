import torch
torch.manual_seed(0)

from qg import QgModel

import os
from tqdm import tqdm

from iox.default import params

# High res model.
system = QgModel(
  name='\\mathcal{F}',
  **params,
)

def solver(system, dir, iterations, scale, **kwargs):
  Nx = system.grid.Nx
  Ny = system.grid.Ny
  
  dns  = torch.zeros([iterations, 4, Ny,  Nx ], dtype=torch.float64)
  
  with torch.no_grad():
    for it in tqdm(range(iterations * scale)):
      
      system.pde.step(system)
      
      if it % scale == 0:
        i = int(it / scale)
        q, p, u, v = system.update()
        dns[i] = torch.stack((q, p, u, v))

  os.makedirs(dir, exist_ok=True)
  # torch.save(dns, os.path.join(dir, 'dns_0.pt'))
  return dns

print(system)
dns_dev = solver(system, dir='bench', **params)

# compare output to reference dns_0.pt
ref = torch.load('bench/dns_0.pt', weights_only=True)
assert torch.allclose(dns_dev, ref, atol=1e-6), f'Benchmark failed; error = {torch.nn.functional.mse_loss(dns_dev,ref)}'
