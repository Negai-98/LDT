import time

import torch
import numpy as np
import warnings

from scipy.stats import entropy
from sklearn.neighbors import NearestNeighbors
from numpy.linalg import norm
from scipy.optimize import linear_sum_assignment

# Borrow from https://github.com/ThibaultGROUEIX/AtlasNet
from tqdm import tqdm


try:
    from StructuralLosses.nn_distance import nn_distance
    def distChamferCUDA(x, y):
        return nn_distance(x, y)
except:
    print("distChamferCUDA not available; fall back to slower version.")

def distChamfer(a, b):
    x, y = a, b
    bs, num_points, points_dim = x.size()
    xx = torch.bmm(x, x.transpose(2, 1))
    yy = torch.bmm(y, y.transpose(2, 1))
    zz = torch.bmm(x, y.transpose(2, 1))
    diag_ind = torch.arange(0, num_points).to(a).long()
    rx = xx[:, diag_ind, diag_ind].unsqueeze(1).expand_as(xx)
    ry = yy[:, diag_ind, diag_ind].unsqueeze(1).expand_as(yy)
    P = (rx.transpose(2, 1) + ry - 2 * zz)
    return P.min(1)[0], P.min(2)[0]



try:
    from StructuralLosses.match_cost import match_cost


    def emd_approx_cuda(sample, ref):
        B, N, N_ref = sample.size(0), sample.size(1), ref.size(1)
        assert N == N_ref, "Not sure what would EMD do in this case"
        emd = match_cost(sample, ref)  # (B,)
        emd_norm = emd / float(N)  # (B,)
        return emd_norm
except:
    def emd_approx(x, y):
        bs, npts, mpts, dim = x.size(0), x.size(1), y.size(1), x.size(2)
        assert npts == mpts, "EMD only works if two point clouds are equal size"
        dim = x.shape[-1]
        x = x.reshape(bs, npts, 1, dim)
        y = y.reshape(bs, 1, mpts, dim)
        dist = (x - y).norm(dim=-1, keepdim=False)  # (bs, npts, mpts)

        emd_lst = []
        dist_np = dist.cpu().detach().numpy()
        for i in range(bs):
            d_i = dist_np[i]
            r_idx, c_idx = linear_sum_assignment(d_i)
            emd_i = d_i[r_idx, c_idx].mean()
            emd_lst.append(emd_i)
        emd = np.stack(emd_lst).reshape(-1)
        emd_torch = torch.from_numpy(emd).to(x)
        return emd_torch
    print("emd_approx_cuda not available. Fall back to slower version.")


def EMD_CD(sample_pcs, ref_pcs, batch_size, accelerated_cd=True, reduced=True,
           accelerated_emd=True):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    assert N_sample == N_ref, "REF:%d SMP:%d" % (N_ref, N_sample)

    cd_lst = []
    emd_lst = []
    iterator = range(0, N_sample, batch_size)

    for b_start in iterator:
        b_end = min(N_sample, b_start + batch_size)
        sample_batch = sample_pcs[b_start:b_end]
        ref_batch = ref_pcs[b_start:b_end]

        if accelerated_cd:
            dl, dr = distChamferCUDA(sample_batch, ref_batch)
        else:
            dl, dr = distChamfer(sample_batch, ref_batch)
        cd_lst.append(dl.mean(dim=1) + dr.mean(dim=1))

        if accelerated_emd:
            emd_batch = emd_approx_cuda(sample_batch, ref_batch)
        else:
            emd_batch = emd_approx(sample_batch, ref_batch)
        emd_lst.append(emd_batch)

    if reduced:
        cd = torch.cat(cd_lst).mean()
        emd = torch.cat(emd_lst).mean()
    else:
        cd = torch.cat(cd_lst)
        emd = torch.cat(emd_lst)

    results = {
        'mmd-CD': cd,
        'mmd-EMD': emd,
    }
    return results




def _pairwise_EMD_CD_(sample_pcs, ref_pcs, batch_size, accelerated_cd=True,
                      accelerated_emd=True):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    all_cd = []
    all_emd = []
    iterator = range(N_sample)
    time_a = 0.
    time_b = 0.
    tbar = tqdm(iterator, desc='Pairwise EMD-CD', nrows=220)
    for i, sample_b_start in enumerate(tbar):
        sample_batch = sample_pcs[sample_b_start]
        cd_lst = []
        emd_lst = []
        for ref_b_start in range(0, N_ref, batch_size):
            ref_b_end = min(N_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]

            batch_size_ref = ref_batch.size(0)
            sample_batch_exp = sample_batch.view(1, -1, 3).expand(batch_size_ref, -1, -1)
            sample_batch_exp = sample_batch_exp.contiguous()

            if accelerated_cd and distChamferCUDA is not None:
                t = time.time()
                dl, dr = distChamferCUDA(sample_batch_exp, ref_batch)
                time_a = time.time() - t

            else:
                dl, dr = distChamfer(sample_batch_exp, ref_batch)
            cd_lst.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1))

            if accelerated_emd:
                t = time.time()
                emd_batch = emd_approx_cuda(sample_batch_exp, ref_batch)
                time_b = time.time() - t

            else:
                emd_batch = emd_approx(sample_batch_exp, ref_batch)

            tbar.set_postfix({"t_cd": time_a, "t_emd": time_b})
            emd_lst.append(emd_batch.view(1, -1))

        cd_lst = torch.cat(cd_lst, dim=1)
        emd_lst = torch.cat(emd_lst, dim=1)
        all_cd.append(cd_lst)
        all_emd.append(emd_lst)

    all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref
    all_emd = torch.cat(all_emd, dim=0)  # N_sample, N_ref

    return all_cd, all_emd


def _pairwise_CD_(sample_pcs, ref_pcs, batch_size, verbose=True):
    N_sample = sample_pcs.shape[0]
    N_ref = ref_pcs.shape[0]
    all_cd = []
    iterator = range(N_sample)
    # if verbose:
    #     iterator = tqdm(iterator, desc='Pairwise EMD-CD')
    for sample_b_start in iterator:
        sample_batch = sample_pcs[sample_b_start]
        cd_lst = []
        sub_iterator = range(0, N_ref, batch_size)
        # if verbose:
        #     sub_iterator = tqdm(sub_iterator, leave=False)
        for ref_b_start in sub_iterator:
            ref_b_end = min(N_ref, ref_b_start + batch_size)
            ref_batch = ref_pcs[ref_b_start:ref_b_end]

            batch_size_ref = ref_batch.size(0)
            point_dim = ref_batch.size(2)
            sample_batch_exp = sample_batch.view(1, -1, point_dim).expand(
                batch_size_ref, -1, -1)
            sample_batch_exp = sample_batch_exp.contiguous()
            try:
                dl, dr = distChamferCUDA(sample_batch_exp, ref_batch)
            except:
                dl, dr = distChamfer(sample_batch_exp, ref_batch)
            cd_lst.append((dl.mean(dim=1) + dr.mean(dim=1)).view(1, -1))


        cd_lst = torch.cat(cd_lst, dim=1)
        all_cd.append(cd_lst)

    all_cd = torch.cat(all_cd, dim=0)  # N_sample, N_ref
    return all_cd


# Adapted from https://github.com/xuqiantong/GAN-Metrics/blob/master/framework/metric.py
def knn(Mxx, Mxy, Myy, k, sqrt=False):
    n0 = Mxx.size(0)
    n1 = Myy.size(0)
    label = torch.cat((torch.ones(n0), torch.zeros(n1))).to(Mxx)
    M = torch.cat((torch.cat((Mxx, Mxy), 1), torch.cat((Mxy.transpose(0, 1), Myy), 1)), 0)
    if sqrt:
        M = M.abs().sqrt()
    INFINITY = float('inf')
    val, idx = (M + torch.diag(INFINITY * torch.ones(n0 + n1).to(Mxx))).topk(k, 0, False)

    count = torch.zeros(n0 + n1).to(Mxx)
    for i in range(0, k):
        count = count + label.index_select(0, idx[i])
    pred = torch.ge(count, (float(k) / 2) * torch.ones(n0 + n1).to(Mxx)).float()

    s = {
        'tp': (pred * label).sum(),
        'fp': (pred * (1 - label)).sum(),
        'fn': ((1 - pred) * label).sum(),
        'tn': ((1 - pred) * (1 - label)).sum(),
    }

    s.update({
        'precision': s['tp'] / (s['tp'] + s['fp'] + 1e-10),
        'recall': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        # 'acc_t': s['tp'] / (s['tp'] + s['fn'] + 1e-10),
        # 'acc_f': s['tn'] / (s['tn'] + s['fp'] + 1e-10),
        'acc': torch.eq(label, pred).float().mean(),
    })
    return s


def lgan_mmd_cov(all_dist):
    N_sample, N_ref = all_dist.size(0), all_dist.size(1)
    min_val_fromsmp, min_idx = torch.min(all_dist, dim=1)
    min_val, _ = torch.min(all_dist, dim=0)
    mmd = min_val.mean()
    # mmd_smp = min_val_fromsmp.mean()
    cov = float(min_idx.unique().view(-1).size(0)) / float(N_ref)
    cov = torch.tensor(cov).to(all_dist)
    return {
        'mmd': mmd,
        'cov': cov,
        # 'mmd_smp': mmd_smp,
    }


def compute_all_metrics(sample_pcs, ref_pcs, batch_size, accelerated_cd=True, accelerated_emd=True):
    results = {}
    ref_pcs, sample_pcs = ref_pcs.cuda(), sample_pcs.cuda()
    M_rs_cd, M_rs_emd = _pairwise_EMD_CD_(ref_pcs, sample_pcs, batch_size, accelerated_cd=accelerated_cd, accelerated_emd=accelerated_emd)

    res_cd = lgan_mmd_cov(M_rs_cd.t())
    results.update({
        "%s-CD" % k: v for k, v in res_cd.items()
    })

    res_emd = lgan_mmd_cov(M_rs_emd.t())
    results.update({
        "%s-EMD" % k: v for k, v in res_emd.items()
    })
    for k, v in results.items():
        print('[%s] %.8f' % (k, v.item()))
    M_rr_cd, M_rr_emd = _pairwise_EMD_CD_(ref_pcs, ref_pcs, batch_size, accelerated_cd=accelerated_cd, accelerated_emd=accelerated_emd)
    M_ss_cd, M_ss_emd = _pairwise_EMD_CD_(sample_pcs, sample_pcs, batch_size, accelerated_cd=accelerated_cd, accelerated_emd=accelerated_emd)

    # 1-NN results
    one_nn_cd_res = knn(M_rr_cd, M_rs_cd, M_ss_cd, 1, sqrt=False)
    results.update({
        "1-NN-CD-%s" % k: v for k, v in one_nn_cd_res.items() if 'acc' in k
    })
    one_nn_emd_res = knn(M_rr_emd, M_rs_emd, M_ss_emd, 1, sqrt=False)
    results.update({
        "1-NN-EMD-%s" % k: v for k, v in one_nn_emd_res.items() if 'acc' in k
    })
    return results


def compute_MMD_metrics(sample_pcs, ref_pcs, batch_size, accelerated_cd=True, accelerated_emd=True):
    results = {}
    ref_pcs, sample_pcs = ref_pcs.cuda(), sample_pcs.cuda()
    M_rs_cd, M_rs_emd = _pairwise_EMD_CD_(ref_pcs, sample_pcs, batch_size, accelerated_cd=accelerated_cd, accelerated_emd=accelerated_emd)

    res_cd = lgan_mmd_cov(M_rs_cd.t())
    results.update({
        "%s-CD" % k: v for k, v in res_cd.items()
    })

    res_emd = lgan_mmd_cov(M_rs_emd.t())
    results.update({
        "%s-EMD" % k: v for k, v in res_emd.items()
    })
    for k, v in results.items():
        print('[%s] %.8f' % (k, v.item()))
    return results


def compute_CD_metrics(sample_pcs, ref_pcs, batch_size):
    results = {}

    print("Pairwise EMD CD")
    M_rs_cd = _pairwise_CD_(ref_pcs, sample_pcs, batch_size)
    ## CD
    res_cd = lgan_mmd_cov(M_rs_cd.t())
    results.update({
        "%s-CD" % k: v for k, v in res_cd.items()
    })
    for k, v in results.items():
        print('[%s] %.8f' % (k, v.item()))
    M_rr_cd = _pairwise_CD_(ref_pcs, ref_pcs, batch_size)
    M_ss_cd = _pairwise_CD_(sample_pcs, sample_pcs, batch_size)
    # 1-NN results
    one_nn_cd_res = knn(M_rr_cd, M_rs_cd, M_ss_cd, 1, sqrt=False)
    results.update({
        "1-NN-CD-%s" % k: v for k, v in one_nn_cd_res.items() if 'acc' in k
    })
    return results


#######################################################
# JSD : from https://github.com/optas/latent_3d_points
#######################################################
def unit_cube_grid_point_cloud(resolution, clip_sphere=False):
    """Returns the center coordinates of each cell of a 3D grid with resolution^3 cells,
    that is placed in the unit-cube.
    If clip_sphere it True it drops the "corner" cells that lie outside the unit-sphere.
    """
    grid = np.ndarray((resolution, resolution, resolution, 3), np.float32)
    spacing = 1.0 / float(resolution - 1)
    for i in range(resolution):
        for j in range(resolution):
            for k in range(resolution):
                grid[i, j, k, 0] = i * spacing - 0.5
                grid[i, j, k, 1] = j * spacing - 0.5
                grid[i, j, k, 2] = k * spacing - 0.5

    if clip_sphere:
        grid = grid.reshape(-1, 3)
        grid = grid[norm(grid, axis=1) <= 0.5]

    return grid, spacing


def jsd_between_point_cloud_sets(sample_pcs, ref_pcs, resolution=28):
    """Computes the JSD between two sets of point-clouds, as introduced in the paper
    ```Learning Representations And Generative Models For 3D Point Clouds```.
    Args:
        sample_pcs: (np.ndarray S1xR2x3) S1 point-clouds, each of R1 points.
        ref_pcs: (np.ndarray S2xR2x3) S2 point-clouds, each of R2 points.
        resolution: (int) grid-resolution. Affects granularity of measurements.
    """
    in_unit_sphere = True
    sample_grid_var = entropy_of_occupancy_grid(sample_pcs, resolution, in_unit_sphere)[1]
    ref_grid_var = entropy_of_occupancy_grid(ref_pcs, resolution, in_unit_sphere)[1]
    return jensen_shannon_divergence(sample_grid_var, ref_grid_var)


def entropy_of_occupancy_grid(pclouds, grid_resolution, in_sphere=False, verbose=False):
    """Given a collection of point-clouds, estimate the entropy of the random variables
    corresponding to occupancy-grid activation patterns.
    Inputs:
        pclouds: (numpy array) #point-clouds x points per point-cloud x 3
        grid_resolution (int) size of occupancy grid that will be used.
    """
    epsilon = 10e-4
    bound = 0.5 + epsilon
    if abs(np.max(pclouds)) > bound or abs(np.min(pclouds)) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit cube.')

    if in_sphere and np.max(np.sqrt(np.sum(pclouds ** 2, axis=2))) > bound:
        if verbose:
            warnings.warn('Point-clouds are not in unit sphere.')

    grid_coordinates, _ = unit_cube_grid_point_cloud(grid_resolution, in_sphere)
    grid_coordinates = grid_coordinates.reshape(-1, 3)
    grid_counters = np.zeros(len(grid_coordinates))
    grid_bernoulli_rvars = np.zeros(len(grid_coordinates))
    nn = NearestNeighbors(n_neighbors=1).fit(grid_coordinates)

    for pc in pclouds:
        _, indices = nn.kneighbors(pc)
        indices = np.squeeze(indices)
        for i in indices:
            grid_counters[i] += 1
        indices = np.unique(indices)
        for i in indices:
            grid_bernoulli_rvars[i] += 1

    acc_entropy = 0.0
    n = float(len(pclouds))
    for g in grid_bernoulli_rvars:
        if g > 0:
            p = float(g) / n
            acc_entropy += entropy([p, 1.0 - p])

    return acc_entropy / len(grid_counters), grid_counters


def jensen_shannon_divergence(P, Q):
    if np.any(P < 0) or np.any(Q < 0):
        raise ValueError('Negative values.')
    if len(P) != len(Q):
        raise ValueError('Non equal size.')

    P_ = P / np.sum(P)  # Ensure probabilities.
    Q_ = Q / np.sum(Q)

    e1 = entropy(P_, base=2)
    e2 = entropy(Q_, base=2)
    e_sum = entropy((P_ + Q_) / 2.0, base=2)
    res = e_sum - ((e1 + e2) / 2.0)

    res2 = _jsdiv(P_, Q_)

    if not np.allclose(res, res2, atol=10e-5, rtol=0):
        warnings.warn('Numerical values of two JSD methods don\'t agree.')

    return res


def _jsdiv(P, Q):
    """another way of computing JSD"""

    def _kldiv(A, B):
        a = A.copy()
        b = B.copy()
        idx = np.logical_and(a > 0, b > 0)
        a = a[idx]
        b = b[idx]
        return np.sum([v for v in a * np.log2(a / b)])

    P_ = P / np.sum(P)
    Q_ = Q / np.sum(Q)

    M = 0.5 * (P_ + Q_)

    return 0.5 * (_kldiv(P_, M) + _kldiv(Q_, M))


if __name__ == "__main__":
    B, N = 64, 512
    torch.random.manual_seed(1)
    part = 32
    smp = torch.ones(B-part, N , 3).cuda()
    smp = torch.cat((smp, torch.randn(part, N, 3).cuda()), dim=0)
    gt = torch.ones(B, N, 3).cuda()

    gen_res = compute_all_metrics(smp, gt, batch_size=256)
    # gen_res = EMD_CD(x, y, batch_size=256)
    all_res = {
        ("val/gen/%s" % k):
            (v if isinstance(v, float) else v.item())
        for k, v in gen_res.items()}
    print("Validation Sample (unit) ", gen_res)
