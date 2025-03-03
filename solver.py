import math
import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.fft

from src.grid import TwoGrid
from src.timestepper import ForwardEuler, RungeKutta2, RungeKutta4
from src.pde import Pde, Eq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('device =', device)

# Utility functions
def to_spectral(y): 
  return torch.fft.rfftn(y, norm='forward')

def to_physical(y): 
  return torch.fft.irfftn(y, norm='forward')

def physical_curl(u, v):
  # shape ...xy 
  dv_dx = torch.gradient(v, dim = -1)[0]
  du_dy = torch.gradient(u, dim = -2)[0]
  return dv_dx - du_dy

def compute_uv(sol, grid):
  qh = sol.clone()
  ph = -qh * grid.irsq
  uh = -1j * grid.ky * ph
  vh =  1j * grid.kr * ph
  return qh, ph, uh, vh

def compute_uv_masked(sol, base_mask, mask, grid):
  qh = sol.clone()
  
  # add penalty term based on qh
  # - lap p = qh becomes
  # - lap p = qh - penalty
  # - lap p = qh * maskh
  
  # A
  # ph = -qh * grid.irsq
  # p = to_physical(ph)
  # pm = p * mask
  # qmp = grid.krsq * to_spectral(pm)
  # penalty = - qmp
  
  # _qh = - penalty
  # ph = -_qh * grid.irsq

  # B
  # solve poisson equation s.t. lap p = 0 with bc p = _q_bc
  # qh = to_spectral(to_physical(qh) * (1-mask))
  # ph = -qh * grid.irsq

  # for _ in range(30):
  #   # convolve _bc with ph:
  #   source = qh + to_spectral(to_physical(ph) * (mask))
  #   ph = -source * grid.irsq
  
  # C 
  # # fixed-point (addding A p_t+1 where A is -k^2)
  
  qh = to_spectral(to_physical(qh) * (1-base_mask))
  ph = -qh * grid.irsq
  for _ in range(5):
    # convolve _bc with ph:
    ph -= to_spectral(to_physical(ph) * (mask)) * grid.irsq     
   
  # D
  # fixed-point but opposite penalization? TODO fix
  
  # qh = to_spectral(to_physical(qh) * (1-mask))
  # ph = -qh * grid.irsq
  # for _ in range(5):
  #   # convolve _bc with ph:
  #   ph -= to_spectral(to_physical(ph) * (1-base_mask)) * grid.krsq     
  
  uh = -1j * grid.ky * ph
  vh =  1j * grid.kr * ph   
  
  # uh = to_spectral(to_physical(uh) * (1-mask))
  # vh = to_spectral(to_physical(vh) * (1-mask))
  # qh = (1j * grid.kr * vh) - (-1j * grid.ky * uh) # curl(u,v)
  # ph = -qh * grid.irsq
    
  return qh, ph, uh, vh

def to_physical_vars(qh, uh, vh):
  q = to_physical(qh)
  u = to_physical(uh)
  v = to_physical(vh)
  return q, u, v

def to_physical_vars_les(qh, uh, vh, eta, da):
  eh = to_spectral(eta)
  qhh, uhh, vhh, ehh = da.increase(qh), da.increase(uh), da.increase(vh), da.increase(eh)
  q, u, v, e = to_physical(qhh), to_physical(uhh), to_physical(vhh), to_physical(ehh)
  return q, u, v, e

def compute_uvq(u, v, qe):
  uq = u * qe
  vq = v * qe
  return uq, vq

def update_spectral(S, uq, vq, grid, les=False):
  uqh = to_spectral(uq)
  vqh = to_spectral(vq)
  if les:
    uqh = grid.reduce(uqh)
    vqh = grid.reduce(vqh)
  S[:] = -1j * grid.kr * uqh - 1j * grid.ky * vqh
  grid.dealias(S[:])

def apply_source(S, source, i, sol, dt, t, grid):
  if source:
    S[:] += source(i, sol, dt, t, grid)
    
def apply_boundary(S, u, v, dt, _eta, chi, v_solid):
  if chi is not None:
      eta = _eta*dt
      du = (u - v_solid[0]) * chi
      dv = (v - v_solid[1]) * chi
      curl = to_spectral(physical_curl(du, dv) / eta)
      S[:] -= curl

def apply_sgs(S, sgs, solver, i, sol, grid):
  if sgs:
    S[:] += sgs.predict(solver, i, sol, grid)


class PsuedoSpectralSolver(nn.Module):
  def __init__(self, Nx, Ny, Lx, Ly, dt, t0, B, mu, nu, nv, eta, eta_penalty, source, init, mask, mask_velocity=None, sgs=None, **kwargs):
    """
    Notation: 
    - q, $\omega$ is potential vorticity
    - p, $\psi$ is streamfunction
    - u, $u$ is x-axis velocity
    - v, $v$ is y-axis velocity
    - Variables in spectral space contain an h
    - Spatially filtered variables contain an _

    Args:
    """
    super(PsuedoSpectralSolver, self).__init__()
    self.n_outputs = 4
    self.B = B
    self.mu = mu
    self.nu = nu
    self.nv = nv
    self.eta = eta.to(device)
    self.eta_penalty = eta_penalty
    
    self.grid = TwoGrid(device, Nx=Nx, Ny=Ny, Lx=Lx, Ly=Ly)
    self.linear_term = self._linear_term(self.grid)

    if sgs:
      # use 3/2 rule
      self.eq = Eq(grid=self.grid, linear_term=self.linear_term, nonlinear_term=self.dns)
      self.da = TwoGrid(device, Nx=int((3./2.)*Nx), Ny=int((3./2.)*Ny), Lx=Lx, Ly=Ly, dealias=1/3)
    else:
      # use 2/3 rule
      self.eq = Eq(grid=self.grid, linear_term=self.linear_term, nonlinear_term=self.dns)
      
      
    self.stepper = RungeKutta4(eq=self.eq)

    self.pde = Pde(dt=dt, t0=t0, eq=self.eq, stepper=self.stepper)
    self.source = source
    self.mask = mask
    self.mask_velocity = mask_velocity
    
    self.kernel = self.grid.cutoff
      
    self.sgs = sgs
    
    init(self) # Initialize IC.
    
    _mask, _ = self.solve_mask(0, self.pde.sol, dt, t0, self.grid)
    self.pde.sol = to_spectral(to_physical(self.pde.sol) * (1-_mask))

  def __str__(self):
    return """Qg model
       Grid: [{nx},{ny}] in [{lx},{ly}]
       μ: {mu}
       ν: {nu}
       β: {beta}
       dt: {dt}
       """.format(
      nx=self.grid.Nx, 
      ny=self.grid.Ny, 
      lx=self.grid.Lx,
      ly=self.grid.Ly,
      mu=self.mu,
      nu=self.nu,
      beta=self.B,
      dt=self.pde.cur.dt)

  def dns(self, i, S, sol, dt, t, grid):
    base_mask, _mask, _mask_v = self.solve_mask(i, sol, dt, t, grid)
    qh, _, uh, vh = compute_uv(sol, grid)
    q, u, v = to_physical_vars(qh, uh, vh)
    
    qe = q + self.eta
    uq, vq = compute_uvq(u, v, qe) 
       
    update_spectral(S, uq, vq, grid)
    apply_source(S, self.source, i, sol, dt, t, grid)
    S += self.linear_term * sol
    
    # S = to_spectral(to_physical(S) * (1-_mask)) # not needed
    
    apply_boundary(S, u, v, dt, self.eta_penalty,_mask,_mask_v)

    return S
    
  def les(self, i, S, sol, dt, t, grid):
    _mask, _mask_v = self.solve_mask(i, sol, dt, t, grid)
    qh, _, uh, vh = compute_uv(sol, grid)
    q, u, v, e = to_physical_vars_les(qh, uh, vh, self.eta, self.da)
    qe = q + e
    uq, vq = compute_uvq(u, v, qe)
    update_spectral(S, uq, vq, grid, les=True)
    apply_sgs(S, self.sgs, self, i, sol, grid)
    apply_source(S, self.source, i, sol, dt, t, grid)
    S += self.linear_term * sol
    apply_boundary(S, u, v, dt, self.eta_penalty, _mask,_mask_v)
    return S

  def _linear_term(self, grid):
    Lc = -self.mu - self.nu * grid.krsq**self.nv - 1j * self.B * grid.kr * grid.irsq
    Lc[0, 0] = 0
    return Lc

  def solve_mask(self, i, sol, dt, t, grid):
    if hasattr(self, 'chi'):
      return self.base_chi, self.chi, self.solid_vel
    # shape xy,  2xy    
    basic_mask = self.mask(i, sol, dt, t, grid).to(device) # smooth out and add 1/2 at edges
    if basic_mask is None:
      chi = None
      solid_vel = None
    else:
      kernel = 1/16 * torch.tensor([[1, 2, 1], [2, 4, 2], [1, 2, 1]], dtype=torch.float64).to(device)[None,None,...]
      chi = F.conv2d(basic_mask[None,None,:,:], kernel, padding='same')[0,0]
    
    if self.mask_velocity:
      solid_vel = self.mask_velocity(i, sol, dt, t, grid)
      return chi, solid_vel 
    else:
      self.base_chi = basic_mask
      self.chi = chi
      solid_vel = torch.zeros([2,], dtype=torch.float64).to(device)
      self.solid_vel = solid_vel
      return chi, solid_vel
  
  # Flow with random gaussian energy only in the wavenumbers range
  def init_randn(self, energy, wavenumbers):
    K = torch.sqrt(self.grid.krsq) # Wavenumber of each point in frequency space
    k = self.grid.kr.repeat(self.grid.Ny, 1)

    qih = torch.randn(self.pde.sol.size(), dtype=torch.complex128).to(device)
    qih[K < wavenumbers[0]] = 0.0
    qih[K > wavenumbers[1]] = 0.0
    qih[k == 0.0] = 0.0

    E0 = energy
    Ei = 0.5 * (self.grid.int_sq(self.grid.kr * self.grid.irsq * qih) + self.grid.int_sq(self.grid.ky * self.grid.irsq * qih)) / (self.grid.Lx * self.grid.Ly)
    qih *= torch.sqrt(E0 / Ei)
    self.pde.sol = qih
    
  def init_randn_persist(self, args={}):
    if globals().get('IC_qih') is None:
      self.init_randn(*args)
      globals()['IC_qih'] = self.pde.sol.detach().clone()
    else:
      self.pde.sol = globals()['IC_qih'].detach().clone()
    
  def update(self):
    """
    Calculates streamfunction and velocities from vorticity
    """ 
    base_mask, mask, mask_v = self.solve_mask(0, self.pde.sol, self.pde.cur.dt, self.pde.cur.t, self.grid)
    qh, ph, uh, vh = compute_uv(self.pde.sol, self.grid)
    ph = -qh * self.grid.irsq
    # Potential vorticity
    q = to_physical(qh)
    # Streamfunction
    p = to_physical(ph)
    # x-axis velocity
    u = to_physical(uh)
    # y-axis velocity
    v = to_physical(vh)
    return q, p, u, v

  def J(self, grid, qh):
    qh, ph, uh, vh = compute_uv(qh, grid)
    q, u, v = to_physical_vars(qh, uh, vh)
    uq, vq = compute_uvq(u, v, q)
    uqh, vqh = to_spectral(uq), to_spectral(vq)
    return 1j * grid.kr * uqh + 1j * grid.ky * vqh

  def R(self, grid, scale):
    """
    R is the SGS parametrization
    """
    return self.R_field(grid, scale, self.pde.sol)

  def R_field(self, grid, scale, yh):
    """
    See eq (17) in Frezat et al., 22
    """
    return grid.div(torch.stack(self.R_flux(grid, scale, yh), dim=0))

  def R_flux(self, grid, scale, yh):
    # Calc. streamfn and velocities from vorticity
    qh, uh, vh = compute_uv(yh, grid)
    q, u, v = to_physical_vars(qh, uh, vh)

    # Calc. \overline{\vec{u} \omega}
    uq, vq = compute_uvq(u, v, q)
    uqh, vqh = to_spectral(uq), to_spectral(vq)
    uqh_, vqh_ = self.kernel(scale * self.grid.delta(), uqh), self.kernel(scale * self.grid.delta(), vqh)

    # Calc. \bar{\vec{u}} \bar\omega
    uh_, vh_, qh_ = self.kernel(scale * self.grid.delta(), uh), self.kernel(scale * self.grid.delta(), vh), self.kernel(scale * self.grid.delta(), qh)
    u_, v_, q_ = to_physical(uh_), to_physical(vh_), to_physical(qh_)
    u_q_, v_q_ = u_ * q_, v_ * q_
    u_q_h, v_q_h = to_spectral(u_q_), to_spectral(v_q_)
    tu, tv = u_q_h - uqh_, v_q_h - vqh_
    return grid.reduce(tu), grid.reduce(tv)

  # Returns filtered spectral variable, y. 
  def filter(self, grid, scale, y):
    yh = y.clone()
    return grid.reduce(self.kernel(scale * self.grid.delta(), yh))

  def filter_physical(self, grid, scale, y):
    yh = to_spectral(y)
    yl = grid.reduce(self.kernel(scale * self.grid.delta(), yh))
    yl = to_physical(yl)
    return yl

  def run(self, iters, visit, update=False, invisible=False):
    for it in tqdm.tqdm(range(iters), disable=invisible):
      self.pde.step(self)
      visit(self, self.pde.cur, it)
    if update:
      return self.update()

  # Diagnostics
  def energy(self, u, v):
    return 0.5 * torch.mean(u**2 + v**2)

  def enstrophy(self, q):
    return 0.5 * torch.mean(q**2)

  def cfl(self):
    _, _, u, v = self.update()
    return (u.abs().max() * self.pde.cur.dt) / self.grid.dx + (v.abs().max() * self.pde.cur.dt) / self.grid.dy

  def spectrum(self, y):
    K = torch.sqrt(self.grid.krsq)
    d = 0.5
    k = torch.arange(1, int(self.grid.kcut + 1))
    m = torch.zeros(k.size())

    e = [torch.zeros(k.size()) for _ in range(len(y))]
    for ik in range(len(k)):
      n = k[ik]
      i = torch.nonzero((K < (n + d)) & (K > (n - d)), as_tuple=True)
      m[ik] = i[0].numel()
      for j, yj in enumerate(y):
        e[j][ik] = torch.sum(yj[i]) * k[ik] * math.pi / (m[ik] - d)
    return k, e

  def invariants(self, qh):
    qh, uh, vh = compute_uv(qh, self.grid)
    e, z = torch.abs(uh)**2 + torch.abs(vh)**2, torch.abs(qh)**2
    k, [ek, zk] = self.spectrum([e, z])
    return k, ek, zk

  def fluxes(self, R, qh):
    # resolved rate
    sh = -torch.conj(qh) * self.J(self.grid, qh)
    # modeled rate
    lh =  torch.conj(qh) * R

    k, [sk, lk] = self.spectrum([torch.real(sh), torch.real(lh)])
    return k, sk, lk

  def zero_grad(self):
    self.stepper.zero_grad()

