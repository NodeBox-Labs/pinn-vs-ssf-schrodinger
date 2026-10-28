import torch


def pde_residual(model, x, t, V_fn, g):
    x = x.clone().requires_grad_(True)
    t = t.clone().requires_grad_(True)
    u, v = model(x, t)

    u_t = torch.autograd.grad(u, t, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_t = torch.autograd.grad(v, t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, grad_outputs=torch.ones_like(u), create_graph=True)[0]
    v_x = torch.autograd.grad(v, x, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, grad_outputs=torch.ones_like(u_x), create_graph=True)[0]
    v_xx = torch.autograd.grad(v_x, x, grad_outputs=torch.ones_like(v_x), create_graph=True)[0]

    V = V_fn(x)
    mod2 = u ** 2 + v ** 2

    Ru = -v_t + 0.5 * u_xx - V * u - g * mod2 * u
    Rv =  u_t + 0.5 * v_xx - V * v - g * mod2 * v
    return Ru, Rv
