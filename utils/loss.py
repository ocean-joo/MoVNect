import torch

def teacher_heatmap_loss(heat_GT, x, x_GT) :
    return torch.norm(heat_GT * (x-x_GT), p=2)

def heatmap_loss(heat, heat_GT, heat_teach, alpha=0.5) :
    res1 = alpha * torch.norm(heat-heat_GT, p=2, dim=(2,3))
    res2 = (1-alpha) * torch.norm(heat-heat_teach, p=2, dim=(2,3))
    return torch.mean(res1+res2, axis=1)

def location_loss(loc, loc_GT, loc_teach, heat_GT, alpha=0.5) :
    res1 = alpha * torch.norm(heat_GT * (loc-loc_GT), p=2, dim=(2,3))
    res2 = (1-alpha) * torch.norm(heat_GT * (loc-loc_GT), p=2, dim=(2,3))
    return torch.mean(res1+res2, axis=1)
