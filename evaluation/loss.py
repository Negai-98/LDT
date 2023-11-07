import torch
try:
    import evaluation.ChamferDistancePytorch.chamfer3D.dist_chamfer_3D as dist_chamfer_3D
except:
    pass
from evaluation.emd import emdModule

def unmask(x, x_mask):
    # only applies for const-sized sets
    if x_mask is None:
        return x
    bsize = x.shape[0]
    n_points = (~x_mask).sum(-1)[0]
    assert ((~x_mask).sum(-1) == n_points).all()
    return x[~x_mask].reshape((bsize, n_points, -1))


def _chamfer_loss_v1(output_set, target_set):
    B, N, _ = output_set.shape
    _, M, _ = target_set.shape
    sizesA = [N, ] * B
    sizesB = [M, ] * B
    out = output_set.flatten(0, 1)  # [B * N, C]
    tgt = target_set.flatten(0, 1)  # [B * N, C]
    outs = out.split(sizesA, 0)
    tgts = tgt.split(sizesB, 0)
    cd_ot = list()
    cd_to = list()
    for o, t in zip(outs, tgts):  # [m, C]
        o_ = o.unsqueeze(1).repeat(1, t.size(0), 1)  # [m, m, C]
        t_ = t.unsqueeze(0).repeat(o.size(0), 1, 1)  # [m, m, C]
        l2 = (o_ - t_).pow(2).sum(dim=-1)  # [m, m]
        tdist = l2.min(0)[0].sum()  # min over outputs
        odist = l2.min(1)[0].sum()
        cd_ot.append(tdist)
        cd_to.append(odist)
    loss_ot = sum(cd_ot) / float(len(cd_ot))
    loss_to = sum(cd_to) / float(len(cd_to))
    # batch average
    return loss_ot + loss_to, loss_ot, loss_to


def _chamfer_loss_v2(a, b):
    # Memory inefficient! use v1
    x, y = a, b
    bs, x_num_points, points_dim = x.size()
    y_num_points = y.size(1)
    xx = torch.bmm(x, x.transpose(2, 1))  # [B, N, N]
    yy = torch.bmm(y, y.transpose(2, 1))  # [B, M, M]
    zz = torch.bmm(x, y.transpose(2, 1))  # [B, N, M]
    x_diag_ind = torch.arange(0, x_num_points).to(a).long()
    y_diag_ind = torch.arange(0, y_num_points).to(a).long()
    rx = xx[:, x_diag_ind, x_diag_ind].unsqueeze(-1).expand_as(zz)  # [B, N, M]
    ry = yy[:, y_diag_ind, y_diag_ind].unsqueeze(-1).expand_as(zz.transpose(2, 1))  # [B, M, N]
    P = (rx + ry.transpose(2, 1) - 2 * zz)  # [B, N, M]

    cd_ot = list()
    cd_to = list()
    for p in P:
        dl = p.min(0)[0].sum()  # [m,] -> scalar
        dr = p.min(1)[0].sum()  # [n,] -> scalar
        cd_ot.append(dl)
        cd_to.append(dr)
    loss_ot = sum(cd_ot) / float(len(cd_ot))
    loss_to = sum(cd_to) / float(len(cd_to))
    # batch average
    return loss_ot + loss_to, loss_ot, loss_to



def CD_loss(esti_shapes, shapes, type='l1'):
    cf_dist = dist_chamfer_3D.chamfer_3DDist()
    dist1, dist2, idx1, idx2 = cf_dist(esti_shapes, shapes)
    if type == 'l1':
        loss_cd = torch.mean(torch.sqrt(dist1)) + torch.mean(torch.sqrt(dist2))
    else:
        loss_cd = torch.mean(dist1) + torch.mean(dist2)
    return loss_cd


def EMD_loss(esti_shapes, shapes):
    emd_dist = emdModule()
    dist, assigment = emd_dist(esti_shapes, shapes, 0.005, 50)
    loss_emd = torch.sqrt(dist).mean()
    return loss_emd
