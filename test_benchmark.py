import torch
torch.manual_seed(0)

from solver import PsuedoSpectralSolver

import os
from tqdm import tqdm

from iox.default import params


def solve(params, dir, iterations, scale, save=False, **kwargs):
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
  system = PsuedoSpectralSolver(**params).to(device)
  print(system)
   
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

 
  

def test_benchmark(save=False):
  # High res model.
  dns_dev = solve(params, dir='bench', save=save, **params)

  # compare output to reference dns_0.pt
  ref = torch.load('bench/dns_0.pt', weights_only=True)
  print('MSE:', torch.nn.functional.mse_loss(dns_dev, ref))
  assert torch.allclose(dns_dev, ref, atol=1e-20), f'Benchmark failed'


if __name__ == '__main__':
  test_benchmark(True)
  
  # plotting
  
  # dns_dev = solve(params, dir='bench', save=False, **params)
  
  # from matplotlib import pyplot as plt
  
  # fig, axs = plt.subplots(2, 2, figsize=(10, 10))
  # for i, ax in enumerate(axs.flat):
  #   ax.imshow(dns_dev[-1, i].cpu().numpy())
  #   ax.set_title(['q', 'p', 'u', 'v'][i])
  # plt.tight_layout()
  # os.makedirs('out', exist_ok=True)
  # plt.savefig('out/bench.png')
  
  # fig, ax = plt.subplots()
  # dvdx = torch.gradient(dns_dev[-1, -1], dim = 0)[0]
  # dudy = torch.gradient(dns_dev[-1, -2], dim = 1)[0]
  # w = dvdx - dudy
  # ax.imshow(w.cpu().numpy())
  # ax.set_title('w')
  # plt.tight_layout()
  # plt.savefig('out/bench_w.png')
  