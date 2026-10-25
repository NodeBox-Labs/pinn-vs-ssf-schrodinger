"""PDE residual, IC, and periodic-BC losses via autograd."""
import torch


def pde_residual(model, x, t, V_fn, g):
    """Return (Ru, Rv), the real/imag residuals of
    i ψ_t + ½ ψ_xx - V ψ - g|ψ|² ψ = 0.
    """
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u, v = model(x, t)

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v),
                              create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u),
                              create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v),
                              create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x),
                               create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x),
                               create_graph=True)[0]

    V = V_fn(x)
    mod2 = u ** 2 + v ** 2

    # i ψ_t = i(u_t + i v_t) = -v_t + i u_t
    # -½ ψ_xx + V ψ + g|ψ|² ψ
    # real: -v_t + ½ u_xx - V u - g mod2 u = 0
    # imag:  u_t + ½ v_xx - V v - g mod2 v = 0
    Ru = -v_t + 0.5 * u_xx - V * u - g * mod2 * u
    Rv = u_t + 0.5 * v_xx - V * v - g * mod2 * v
    return Ru, Rv


def ic_loss(model, x_ic, u0, v0):
    u, v = model(x_ic, torch.zeros_like(x_ic))
    return ((u - u0) ** 2 + (v - v0) ** 2).mean()


def bc_loss(model, t_bc, L):
    xL = torch.full_like(t_bc, -L / 2)
    xR = torch.full_like(t_bc, L / 2)
    uL, vL = model(xL, t_bc)
    uR, vR = model(xR, t_bc)
    return ((uL - uR) ** 2 + (vL - vR) ** 2).mean()