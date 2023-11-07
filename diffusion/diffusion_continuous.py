# ---------------------------------------------------------------
# Copyright (c) 2021, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the NVIDIA Source Code License
# for LSGM. To view a copy of this license, see the LICENSE file.
# ---------------------------------------------------------------


from abc import ABC, abstractmethod
import numpy as np
import torch
import gc
from torchdiffeq import odeint
from torch.cuda.amp import autocast
from timeit import default_timer as timer


def make_diffusion(args):
    """ simple diffusion factory function to return diffusion instances. Only use this to create continuous diffusions """
    if args.sde_type == 'geometric_sde':
        return DiffusionGeometric(args)
    elif args.sde_type == 'vpsde':
        return DiffusionVPSDE(args)
    elif args.sde_type == 'sub_vpsde':
        return DiffusionSubVPSDE(args)
    elif args.sde_type == 'vesde':
        return DiffusionVESDE(args)
    else:
        raise ValueError("Unrecognized sde type: {}".format(args.sde_type))


class DiffusionBase(ABC):
    """
    Abstract base class for all diffusion implementations.
    """

    def __init__(self, args):
        super().__init__()
        self.sigma2_0 = args.sigma2_0
        self.sde_type = args.sde_type
        self.time_eps = args.time_eps
        self.sample_time_eps = args.sample_time_eps

    # drift coefficient
    @abstractmethod
    def f(self, t):
        """ returns the drift coefficient at time t: f(t) """
        pass

    # diffusion coefficient
    @abstractmethod
    def g2(self, t):
        """ returns the squared diffusion coefficient at time t: g^2(t) """
        pass

    # var
    @abstractmethod
    def var(self, t):
        """ returns variance at time t, \sigma_t^2"""
        pass

    def std(self, t):
        return torch.sqrt(self.var(t))

    # mean in time t
    @abstractmethod
    def e2int_f(self, t):
        """ returns e^{\int_0^t f(s) ds} which corresponds to the coefficient of mean at time t. """
        pass

    # var^{-1}
    @abstractmethod
    def inv_var(self, var):
        """ inverse of the variance function at input variance var. """
        pass

    # sample q_{t}
    def sample_q(self, x_init, noise, var_t, m_t):
        """ returns a sample from diffusion process at time t """
        return m_t * x_init + torch.sqrt(var_t) * noise

    # const term of CE(q(z_{0}|x)||p(z_{0}))
    def cross_entropy_const(self, ode_eps):
        """ returns cross entropy factor with variance according to ode integration cutoff ode_eps """
        # _, c, h, w = x_init.shape
        return 0.5 * (1.0 + torch.log(2.0 * np.pi * self.var(t=torch.tensor(ode_eps, device='cuda'))))

    def sample_model_ode(self, score_fn, num_samples, shape, ode_eps, ode_solver_tol, enable_autocast, noise=None,
                         condition=None, label=None):
        """ generates samples using the ODE framework, assuming integration cutoff ode_eps """
        # ODE solver starts consuming the CPU memory without this on large models
        # https://github.com/scipy/scipy/issues/10070
        gc.collect()

        def ode_func(t, x):
            """ the ode function (sampling only, no NLL stuff) """
            global nfe_counter
            t = t.expand(x.shape[0])
            nfe_counter = nfe_counter + 1
            with autocast(enabled=enable_autocast):
                score, params = score_fn(t, x, label=label, condition=condition)
                dx_dt = self.f(t=t)[:, None, None] * x - 0.5 * self.g2(t=t)[:, None, None] * score
            return dx_dt

        # the initial noise
        if noise is None:
            noise = torch.randn(size=(num_samples,) + shape, device='cuda')

        if self.sde_type == 'vesde':
            noise_init = noise * np.sqrt(self.sigma2_max)
        else:
            noise_init = noise

        # NFE counter
        global nfe_counter
        nfe_counter = 0
        # solve the ODE
        start = timer()
        samples_out = odeint(
            ode_func,
            noise_init,
            torch.tensor([1.0, ode_eps], device='cuda'),
            atol=ode_solver_tol,
            rtol=ode_solver_tol,
            method="scipy_solver",
            options={"solver": 'RK45'},
        )
        end = timer()
        ode_solve_time = end - start

        return samples_out[-1], nfe_counter, ode_solve_time

    def sample_discrete(self, score_fn, num_samples, N, predictor, corrector, corrector_steps,
                        shape, time_eps, probability_flow, denoise, snr, device, condition=None,
                        label=None, print_steps=None):
        """
        Sampling according Reverse SDE
        """
        T = 1.0

        def ReverseDiffusion(x, t, condition=None):
            dt = torch.tensor((1 - time_eps) / N, device=t.device)
            f, g2 = self.f(t)[:, None, None] * x, self.g2(t)[:, None, None]
            score, params = score_fn(t, x, label=label, condition=condition)
            dx = (f - g2 * score * (0.5 if probability_flow else 1.)) * dt
            g = torch.zeros_like(g2) if probability_flow else torch.sqrt(g2)
            z = torch.randn_like(x)
            x_mean = x - dx
            x = x_mean + g * z * torch.sqrt(dt)
            return x, x_mean, params

        def Ancestral(x, t, condition=None):
            assert isinstance(self, DiffusionVPSDE)
            """The ancestral sampling predictor. Currently only supports VP SDEs."""
            T = 1.0
            idx = (t * (N - 1) / T).long()
            beta = self.betas[idx]
            score, params = score_fn(t, x, label=label, condition=condition)
            x_mean = (x + beta[:, None, None] * score) / torch.sqrt(1. - beta)[:, None, None]
            noise = torch.randn_like(x)
            x = x_mean + torch.sqrt(beta)[:, None, None] * noise
            return x, x_mean, params

        def DDIM(x, t, condition=None):
            assert isinstance(self, DiffusionVPSDE)
            T = 1.0
            idx = (t * (N - 1) / T).long()
            at = self.alphas_cump[idx][:, None, None]
            if idx[0] - 1 < 0:
                at_next = torch.ones_like(at)
            else:
                at_next = self.alphas_cump[idx - 1][:, None, None]
            _, params = score_fn(t, x, label=label, condition=condition)
            # sigma = torch.sqrt((1 - alpha0)/(1-alpha1)) * torch.sqrt(1-alpha1/alpha0)
            sigma = 0
            x_mean = at_next.sqrt() * (x - (1 - at).sqrt() * params) / at.sqrt() + (
                    1 - at_next - sigma ** 2).sqrt() * params
            noise = torch.randn_like(x)
            x = x_mean + sigma * noise
            return x, x_mean, params

        def EulerMaruyama(x, t, condition=None):
            dt = -1. / N
            z = torch.randn_like(x)
            f, g2 = self.f(t)[:, None, None] * x, self.g2(t)[:, None, None]
            score, params = score_fn(t, x, label=label, condition=condition)
            f = f - g2 * score * (0.5 if probability_flow else 1.)
            x_mean = x + f * dt
            g2 = torch.zeros(1).to(x) if probability_flow else g2
            x = x_mean + torch.sqrt(g2) * np.sqrt(-dt) * z
            return x, x_mean, params

        def LangevinCorrector(x, t, condition=None):
            target_snr = snr
            if self.__class__ in ["DiffusionVPSDE", "DiffusionSubVPSDE"]:
                timestep = (t * (N - 1) / T).long()
                alphas = 1 - torch.linspace(self.beta_start / N, self.beta_end / N, N).to(x)
                alpha = alphas[timestep]
            else:
                alpha = torch.ones_like(t)

            for i in range(corrector_steps):
                grad, params = score_fn(t, x, label=label, condition=condition)
                noise = torch.randn_like(x)
                grad_norm = torch.norm(grad.reshape(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.reshape(noise.shape[0], -1), dim=-1).mean()
                step_size = (target_snr * noise_norm / grad_norm) ** 2 * 2 * alpha
                x_mean = x + step_size[:, None] * grad
                x = x_mean + torch.sqrt(step_size * 2)[:, None] * noise
            return x, x_mean, params

        def AncestralCorrector(x, t, condition=None):
            assert isinstance(self, DiffusionVPSDE)
            """The ancestral sampling predictor. Currently only supports VP SDEs."""
            target_snr = snr
            if self.__class__ in ["DiffusionVPSDE", "DiffusionSubVPSDE"]:
                timestep = (t * (N - 1) / T).long()
                alphas = 1 - torch.linspace(self.beta_start / N, self.beta_end / N, N).to(x)
                alpha = alphas[timestep]
            else:
                alpha = torch.ones_like(t)
            std = self.std(t)
            for i in range(corrector_steps):
                grad, params = score_fn(t, x, label=label, condition=condition)
                noise = torch.randn_like(x)
                step_size = (target_snr * std) ** 2 * 2 * alpha
                x_mean = x + step_size[:, None, None] * grad
                x = x_mean + noise * torch.sqrt(step_size * 2)[:, None, None]
            return x, x_mean, params

        def pc_sampling(predictor, corrector, condition=None):
            with torch.no_grad():
                # Initial sample
                if self.__class__ in ["DiffusionVESDE"]:
                    x = torch.randn((num_samples,) + shape).to(device) * torch.sqrt(self.sigma2_max)
                else:
                    x = torch.randn((num_samples,) + shape).to(device)
                timesteps = torch.linspace(T, time_eps, N, device=device)
                if print_steps is not None:
                    out_list = [x]
                    steps = (N - 1) // (print_steps - 2)
                for i in range(N):
                    t = timesteps[i]
                    vec_t = torch.ones((num_samples,), device=t.device) * t
                    x_mean = x
                    if predictor is not None:
                        x, x_mean, params = predictor(x, vec_t, condition)
                    if corrector is not None:
                        x, x_mean, params = corrector(x, vec_t, condition)
                    # if self_condition:
                    #     condition = params
                    if print_steps is not None:
                        if (i + 1) % steps == 0:
                            out_list.append(x_mean)
                if print_steps is not None:
                    out_list.append(x_mean if denoise else x)
                    return out_list
                return x_mean if denoise else x

        def PNDM_Sampling(x, t, ets, alphas_cump, train_N):
            assert isinstance(self, DiffusionVPSDE)
            timesteps = torch.linspace(time_eps, 1.0, self.N * 2)

            def transfer(x, t, t_next, et):
                t = (train_N * (t - time_eps) + 1).long()
                t_next = (train_N * (t_next - time_eps) + 1).long()
                at = alphas_cump[t].view(-1, 1)
                at_next = alphas_cump[t_next].view(-1, 1)

                x_delta = (at_next - at) * ((1 / (at.sqrt() * (at.sqrt() + at_next.sqrt()))) * x - 1 / (at.sqrt() * (
                        ((1 - at_next) * at).sqrt() + ((1 - at) * at_next).sqrt())) * et)

                x_next = x + x_delta
                return x_next

            def runge_kutta(x, t_list, score_fn, ets):
                t_in1 = timesteps[t_list[0] * 2 - 1].view(-1).expand(x.shape[0]).to(x)
                t_in2 = timesteps[int(t_list[1] * 2) - 1].view(-1).expand(x.shape[0]).to(x)
                t_in3 = timesteps[int(t_list[2] * 2) - 1].view(-1).expand(x.shape[0]).to(x)
                _, e_1 = score_fn(t_in1, x, condition=condition, label=label)
                ets.append(e_1)
                x_2 = transfer(x, t_in1, t_in2, e_1)
                _, e_2 = score_fn(t_in2, x_2, condition=condition, label=label)
                x_3 = transfer(x, t_in1, t_in2, e_2)
                _, e_3 = score_fn(t_in2, x_3, condition=condition, label=label)
                x_4 = transfer(x, t_in1, t_in3, e_3)
                _, e_4 = score_fn(t_in3, x_4, condition=condition, label=label)
                et = (1 / 6) * (e_1 + 2 * e_2 + 2 * e_3 + e_4)
                return et

            t_next = t - 1
            t_list = [t, (t + t_next) / 2, t_next]
            if len(ets) > 2:
                t_in = timesteps[t * 2 - 1].view(-1).expand(x.shape[0]).to(x)
                _, noise_ = score_fn(t_in, x, condition=condition, label=label)
                ets.append(noise_)
                noise = (1 / 24) * (55 * ets[-1] - 59 * ets[-2] + 37 * ets[-3] - 9 * ets[-4])
            else:
                noise = runge_kutta(x, t_list, score_fn, ets)
            t = timesteps[t * 2 - 1].view(-1).expand(x.shape[0]).to(x)
            t_next = timesteps[t_next * 2 - 1].view(-1).expand(x.shape[0]).to(x)
            x_next = transfer(x, t, t_next, noise)
            return x_next, ets

        if predictor == "pndm":
            sample = torch.randn((num_samples,) + shape).to(device)
            ets = []
            betas = torch.from_numpy(
                np.linspace(self.beta_start / self.train_N, self.beta_end / self.train_N, self.train_N,
                            dtype=np.float64)).to(
                self.delta_beta_half)
            alphas_cump = (1.0 - betas).cumprod(dim=0)
            alphas_cump = torch.cat((torch.ones(1).to("cuda"), alphas_cump))  # 1001
            for idx in range(self.N, 0, -1):
                sample, ets = PNDM_Sampling(sample, idx, ets, alphas_cump, train_N=self.train_N)
            return sample

        if predictor is not None:
            if predictor == "reversediffusion":
                predictor = ReverseDiffusion
            elif predictor == "ancestral":
                predictor = Ancestral
            elif predictor == "eulermaruyama":
                predictor = EulerMaruyama
            elif predictor == "ddim":
                predictor = DDIM
            else:
                raise NotImplementedError("preditor not Implemented")
        if corrector is not None:
            if corrector == "langevin":
                corrector = LangevinCorrector
            elif corrector == "ancestral":
                corrector = AncestralCorrector
            else:
                raise NotImplementedError("corrector not Implemented")
        sample = pc_sampling(predictor, corrector, condition)

        return sample

    def iw_quantities(self, size, time_eps, iw_sample_mode, iw_subvp_like_vp_sde):
        if self.sde_type in ['geometric_sde', 'vpsde']:
            return self._iw_quantities_vpsdelike(size, time_eps, iw_sample_mode)
        elif self.sde_type in ['sub_vpsde']:
            return self._iw_quantities_subvpsdelike(size, time_eps, iw_sample_mode, iw_subvp_like_vp_sde)
        elif self.sde_type in ['vesde']:
            return self._iw_quantities_vesde(size, time_eps, iw_sample_mode)
        else:
            raise NotImplementedError

    # sample methods for w(t) see equation(8)/(9)
    def _iw_quantities_vpsdelike(self, size, time_eps, iw_sample_mode):
        """
        For all SDEs where the underlying SDE is of the form dz = -0.5 * beta(t) * z * dt + sqrt{beta(t)} * dw, like
        for the VPSDE.
        """
        rho = torch.rand(size=[size], device='cuda')

        # In the following, obj_weight_t corresponds to the weight in front of the l2 loss for the given iw_sample_mode.
        # obj_weight_t_ll corresponds to the weight that converts the weighting scheme in iw_sample_mode to likelihood
        # weighting.

        if iw_sample_mode == 'll_uniform':
            # uniform t sampling - likelihood obj. for both q and p
            t = rho * (1. - time_eps) + time_eps
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = obj_weight_t_ll = g2_t / (2.0 * var_t)

        elif iw_sample_mode == 'll_iw':
            # importance sampling for likelihood obj. - likelihood obj. for both q and p
            ones = torch.ones_like(rho, device='cuda')
            sigma2_1, sigma2_eps = self.var(ones), self.var(time_eps * ones)
            log_sigma2_1, log_sigma2_eps = torch.log(sigma2_1), torch.log(sigma2_eps)
            var_t = torch.exp(rho * log_sigma2_1 + (1 - rho) * log_sigma2_eps)
            t = self.inv_var(var_t)
            m_t, g2_t = self.e2int_f(t), self.g2(t)
            obj_weight_t = obj_weight_t_ll = 0.5 * (log_sigma2_1 - log_sigma2_eps) / (1.0 - var_t)

        elif iw_sample_mode == 'drop_all_uniform':
            # uniform t sampling - likelihood obj. for q, all-prefactors-dropped obj. for p
            t = rho * (1. - time_eps) + time_eps
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = torch.ones(1, device='cuda')
            obj_weight_t_ll = g2_t / (2.0 * var_t)

        elif iw_sample_mode == 'drop_all_iw':
            # importance sampling for all-pref.-dropped obj. - likelihood obj. for q, all-pref.-dropped obj. for p
            assert self.sde_type == 'vpsde', 'Importance sampling for fully unweighted objective is currently only ' \
                                             'implemented for the regular VPSDE.'
            t = torch.sqrt(1.0 / self.delta_beta_half) * torch.erfinv(
                rho * self.const_norm_2 + self.const_erf) - self.beta_frac
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = self.const_norm / (1.0 - var_t)
            obj_weight_t_ll = obj_weight_t * g2_t / (2.0 * var_t)

        elif iw_sample_mode == 'drop_sigma2t_iw':
            # importance sampling for inv_sigma2_t-dropped obj. - likelihood obj. for q, inv_sigma2_t-dropped obj. for p
            ones = torch.ones_like(rho, device='cuda')
            sigma2_1, sigma2_eps = self.var(ones), self.var(time_eps * ones)
            var_t = rho * sigma2_1 + (1 - rho) * sigma2_eps
            t = self.inv_var(var_t)
            m_t, g2_t = self.e2int_f(t), self.g2(t)
            obj_weight_t = 0.5 * (sigma2_1 - sigma2_eps) / (1.0 - var_t)
            obj_weight_t_ll = obj_weight_t / var_t

        elif iw_sample_mode == 'drop_sigma2t_uniform':
            # uniform sampling for inv_sigma2_t-dropped obj. - likelihood obj. for q, inv_sigma2_t-dropped obj. for p
            t = rho * (1. - time_eps) + time_eps
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = g2_t / 2.0
            obj_weight_t_ll = g2_t / (2.0 * var_t)

        elif iw_sample_mode == 'rescale_iw':
            # importance sampling for 1/(1-sigma2_t) resc. obj. - likelihood obj. for q, 1/(1-sigma2_t) resc. obj. for p
            t = rho * (1. - time_eps) + time_eps
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = 0.5 / (1.0 - var_t)
            obj_weight_t_ll = g2_t / (2.0 * var_t)

        else:
            raise ValueError("Unrecognized importance sampling type: {}".format(iw_sample_mode))

        return t, var_t.view(-1, 1), m_t.view(-1, 1), obj_weight_t.view(-1, 1), \
               obj_weight_t_ll.view(-1, 1), g2_t.view(-1, 1)

    def _iw_quantities_subvpsdelike(self, size, time_eps, iw_sample_mode, iw_subvp_like_vp_sde):
        """
        For all SDEs where the underlying SDE is of the form
        dz = -0.5 * beta(t) * z * dt + sqrt{beta(t) * (1 - exp[-2 * betaintegral])} * dw, like for the Sub-VPSDE.
        When iw_subvp_like_vp_sde is True, then we define the importance sampling distributions based on an analogous
        VPSDE, while stile using the Sub-VPSDE. The motivation is that deriving the correct importance sampling
        distributions for the Sub-VPSDE itself is hard, but the importance sampling distributions from analogous VPSDEs
        probably already significantly reduce the variance also for the Sub-VPSDE.
        """
        rho = torch.rand(size=[size], device='cuda')

        # In the following, obj_weight_t corresponds to the weight in front of the l2 loss for the given iw_sample_mode.
        # obj_weight_t_ll corresponds to the weight that converts the weighting scheme in iw_sample_mode to likelihood
        # weighting.
        if iw_sample_mode == 'll_uniform':
            # uniform t sampling - likelihood obj. for both q and p
            t = rho * (1. - time_eps) + time_eps
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = obj_weight_t_ll = g2_t / (2.0 * var_t)

        elif iw_sample_mode == 'll_iw':
            if iw_subvp_like_vp_sde:
                # importance sampling for vpsde likelihood obj. - sub-vpsde likelihood obj. for both q and p
                ones = torch.ones_like(rho, device='cuda')
                sigma2_1, sigma2_eps = self.var_vpsde(ones), self.var_vpsde(time_eps * ones)
                log_sigma2_1, log_sigma2_eps = torch.log(sigma2_1), torch.log(sigma2_eps)
                var_t_vpsde = torch.exp(rho * log_sigma2_1 + (1 - rho) * log_sigma2_eps)
                t = self.inv_var_vpsde(var_t_vpsde)
                var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
                obj_weight_t = obj_weight_t_ll = g2_t / (2.0 * var_t) * \
                                                 (log_sigma2_1 - log_sigma2_eps) * var_t_vpsde / (
                                                         1 - var_t_vpsde) / self.beta(t)
            else:
                raise NotImplementedError

        elif iw_sample_mode == 'drop_all_uniform':
            # uniform t sampling - likelihood obj. for q, all-prefactors-dropped obj. for p
            t = rho * (1. - time_eps) + time_eps
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = torch.ones(1, device='cuda')
            obj_weight_t_ll = g2_t / (2.0 * var_t)

        elif iw_sample_mode == 'drop_all_iw':
            if iw_subvp_like_vp_sde:
                # importance sampling for all-pref.-dropped obj. - likelihood obj. for q, all-pref.-dropped obj. for p
                assert self.sde_type == 'sub_vpsde', 'Importance sampling for fully unweighted objective is ' \
                                                     'currently only implemented for the Sub-VPSDE.'
                t = torch.sqrt(1.0 / self.delta_beta_half) * torch.erfinv(
                    rho * self.const_norm_2 + self.const_erf) - self.beta_frac
                var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
                obj_weight_t = self.const_norm / (1.0 - self.var_vpsde(t))
                obj_weight_t_ll = obj_weight_t * g2_t / (2.0 * var_t)
            else:
                raise NotImplementedError

        elif iw_sample_mode == 'drop_sigma2t_iw':
            if iw_subvp_like_vp_sde:
                # importance sampling for inv_sigma2_t-dropped obj. - likelihood obj. for q, inv_sigma2_t-dropped obj. for p
                ones = torch.ones_like(rho, device='cuda')
                sigma2_1, sigma2_eps = self.var_vpsde(ones), self.var_vpsde(time_eps * ones)
                var_t_vpsde = rho * sigma2_1 + (1 - rho) * sigma2_eps
                t = self.inv_var_vpsde(var_t_vpsde)
                var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
                obj_weight_t = 0.5 * g2_t / self.beta(t) * (sigma2_1 - sigma2_eps) / (1.0 - var_t_vpsde)
                obj_weight_t_ll = obj_weight_t / var_t
            else:
                raise NotImplementedError

        elif iw_sample_mode == 'drop_sigma2t_uniform':
            # uniform sampling for inv_sigma2_t-dropped obj. - likelihood obj. for q, inv_sigma2_t-dropped obj. for p
            t = rho * (1. - time_eps) + time_eps
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = g2_t / 2.0
            obj_weight_t_ll = g2_t / (2.0 * var_t)

        elif iw_sample_mode == 'rescale_iw':
            # importance sampling for 1/(1-sigma2_t) resc. obj. - likelihood obj. for q, 1/(1-sigma2_t) resc. obj. for p
            # Note that we use the sub-vpsde variance to scale the p objective! It's not clear what's optimal here!
            t = rho * (1. - time_eps) + time_eps
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = 0.5 / (1.0 - var_t)
            obj_weight_t_ll = g2_t / (2.0 * var_t)

        else:
            raise ValueError("Unrecognized importance sampling type: {}".format(iw_sample_mode))

        return t, var_t.view(-1, 1), m_t.view(-1, 1), obj_weight_t.view(-1, 1), \
               obj_weight_t_ll.view(-1, 1), g2_t.view(-1, 1)

    def _iw_quantities_vesde(self, size, time_eps, iw_sample_mode):
        """
        For the VESDE.
        """
        rho = torch.rand(size=[size], device='cuda')

        # In the following, obj_weight_t corresponds to the weight in front of the l2 loss for the given iw_sample_mode.
        # obj_weight_t_ll corresponds to the weight that converts the weighting scheme in iw_sample_mode to likelihood
        # weighting.
        if iw_sample_mode == 'll_uniform':
            # uniform t sampling - likelihood obj. for both q and p
            t = rho * (1. - time_eps) + time_eps
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = obj_weight_t_ll = g2_t / (2.0 * var_t)

        elif iw_sample_mode == 'll_iw':
            # importance sampling for likelihood obj. - likelihood obj. for both q and p
            ones = torch.ones_like(rho, device='cuda')
            nsigma2_1, nsigma2_eps, sigma2_eps = self.var_N(ones), self.var_N(time_eps * ones), self.var(
                time_eps * ones)
            log_frac_sigma2_1, log_frac_sigma2_eps = torch.log(self.sigma2_max / nsigma2_1), torch.log(
                nsigma2_eps / sigma2_eps)
            var_N_t = (1.0 - self.sigma2_min) / (
                    1.0 - torch.exp(rho * (log_frac_sigma2_1 + log_frac_sigma2_eps) - log_frac_sigma2_eps))
            t = self.inv_var_N(var_N_t)
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = obj_weight_t_ll = 0.5 * (log_frac_sigma2_1 + log_frac_sigma2_eps) * self.var_N(t) / (
                    1.0 - self.sigma2_min)

        elif iw_sample_mode == 'drop_all_uniform':
            # uniform t sampling - likelihood obj. for q, all-prefactors-dropped obj. for p
            t = rho * (1. - time_eps) + time_eps
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = torch.ones(1, device='cuda')
            obj_weight_t_ll = g2_t / (2.0 * var_t)

        elif iw_sample_mode == 'drop_all_iw':
            # importance sampling for all-pref.-dropped obj. - likelihood obj. for q, all-pref.-dropped obj. for p
            ones = torch.ones_like(rho, device='cuda')
            nsigma2_1, nsigma2_eps, sigma2_eps = self.var_N(ones), self.var_N(time_eps * ones), self.var(
                time_eps * ones)
            log_frac_sigma2_1, log_frac_sigma2_eps = torch.log(self.sigma2_max / nsigma2_1), torch.log(
                nsigma2_eps / sigma2_eps)
            var_N_t = (1.0 - self.sigma2_min) / (
                    1.0 - torch.exp(rho * (log_frac_sigma2_1 + log_frac_sigma2_eps) - log_frac_sigma2_eps))
            t = self.inv_var_N(var_N_t)
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t_ll = 0.5 * (log_frac_sigma2_1 + log_frac_sigma2_eps) * self.var_N(t) / (1.0 - self.sigma2_min)
            obj_weight_t = 2.0 * obj_weight_t_ll / np.log(self.sigma2_max / self.sigma2_min)

        elif iw_sample_mode == 'drop_sigma2t_iw':
            # importance sampling for inv_sigma2_t-dropped obj. - likelihood obj. for q, inv_sigma2_t-dropped obj. for p
            ones = torch.ones_like(rho, device='cuda')
            nsigma2_1, nsigma2_eps = self.var_N(ones), self.var_N(time_eps * ones)
            var_N_t = torch.exp(rho * torch.log(nsigma2_1) + (1 - rho) * torch.log(nsigma2_eps))
            t = self.inv_var_N(var_N_t)
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = 0.5 * torch.log(nsigma2_1 / nsigma2_eps) * self.var_N(t)
            obj_weight_t_ll = obj_weight_t / var_t

        elif iw_sample_mode == 'drop_sigma2t_uniform':
            # uniform sampling for inv_sigma2_t-dropped obj. - likelihood obj. for q, inv_sigma2_t-dropped obj. for p
            t = rho * (1. - time_eps) + time_eps
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = g2_t / 2.0
            obj_weight_t_ll = g2_t / (2.0 * var_t)

        elif iw_sample_mode == 'rescale_iw':
            # uniform sampling for 1/(1-sigma2_t) resc. obj. - likelihood obj. for q, 1/(1-sigma2_t) resc. obj. for p
            t = rho * (1. - time_eps) + time_eps
            var_t, m_t, g2_t = self.var(t), self.e2int_f(t), self.g2(t)
            obj_weight_t = 0.5 / (1.0 - var_t)
            obj_weight_t_ll = g2_t / (2.0 * var_t)

        else:
            raise ValueError("Unrecognized importance sampling type: {}".format(iw_sample_mode))

        return t, var_t.view(-1, 1), m_t.view(-1, 1), obj_weight_t.view(-1, 1), \
               obj_weight_t_ll.view(-1, 1), g2_t.view(-1, 1)


class DiffusionGeometric(DiffusionBase):
    """
    Diffusion implementation with dz = -0.5 * beta(t) * z * dt + sqrt(beta(t)) * dW SDE and geometric progression of
    variance. This is our new diffusion.
    """

    def __init__(self, args):
        super().__init__(args)
        self.sigma2_min = args.sigma2_min
        self.sigma2_max = args.sigma2_max

    def f(self, t):
        return -0.5 * self.g2(t)

    def g2(self, t):
        sigma2_geom = self.sigma2_min * ((self.sigma2_max / self.sigma2_min) ** t)
        log_term = np.log(self.sigma2_max / self.sigma2_min)
        return sigma2_geom * log_term / (1.0 - self.sigma2_0 + self.sigma2_min - sigma2_geom)

    def var(self, t):
        return self.sigma2_min * ((self.sigma2_max / self.sigma2_min) ** t) - self.sigma2_min + self.sigma2_0

    def e2int_f(self, t):
        return torch.sqrt(
            1.0 + self.sigma2_min * (1.0 - (self.sigma2_max / self.sigma2_min) ** t) / (1.0 - self.sigma2_0))

    def inv_var(self, var):
        return torch.log((var + self.sigma2_min - self.sigma2_0) / self.sigma2_min) / np.log(
            self.sigma2_max / self.sigma2_min)


class DiffusionVPSDE(DiffusionBase):
    """
    Diffusion implementation of the VPSDE. This uses the same SDE like DiffusionGeometric but with linear beta(t).
    Note that we need to scale beta_start and beta_end by 1000 relative to JH's DDPM values, since our t is in [0,1].
    """

    def __init__(self, args):
        super().__init__(args)
        self.beta_start = args.beta_start
        self.beta_end = args.beta_end

        # auxiliary constants
        self.delta_beta_half = torch.tensor(0.5 * (self.beta_end - self.beta_start), device='cuda')
        self.beta_frac = torch.tensor(self.beta_start / (self.beta_end - self.beta_start), device='cuda')
        self.const_aq = (1.0 - self.sigma2_0) * torch.exp(0.5 * self.beta_frac) * torch.sqrt(
            0.25 * np.pi / self.delta_beta_half)
        self.const_erf = torch.erf(torch.sqrt(self.delta_beta_half) * (self.time_eps + self.beta_frac))
        self.const_norm = self.const_aq * (
                torch.erf(torch.sqrt(self.delta_beta_half) * (1.0 + self.beta_frac)) - self.const_erf)
        self.const_norm_2 = torch.erf(torch.sqrt(self.delta_beta_half) * (1.0 + self.beta_frac)) - self.const_erf
        self.train_N = args.train_N
        if args.sample_mode == "discrete":
            self.N = args.sample_N
            self.betas = torch.from_numpy(
                np.linspace(self.beta_start / self.N, self.beta_end / self.N, self.N, dtype=np.float64)).to(
                self.delta_beta_half)
            self.alpha = 1.0 - self.betas
            self.alphas_cump = self.alpha.cumprod(dim=0)

    def f(self, t):
        return -0.5 * self.g2(t)

    def g2(self, t):
        return self.beta_start + (self.beta_end - self.beta_start) * t

    def discrete(self, idx):
        return self.betas.index_select(0, idx), self.alpha.index_select(0, idx)

    def var(self, t):
        return 1.0 - (1.0 - self.sigma2_0) * torch.exp(
            -self.beta_start * t - 0.5 * (self.beta_end - self.beta_start) * t * t)

    def std(self, t):
        return torch.sqrt(self.var(t))

    def e2int_f(self, t):
        return torch.exp(-0.5 * self.beta_start * t - 0.25 * (self.beta_end - self.beta_start) * t * t)

    def inv_var(self, var):
        c = torch.log((1 - var) / (1 - self.sigma2_0))
        a = self.beta_end - self.beta_start
        t = (-self.beta_start + torch.sqrt(np.square(self.beta_start) - 2 * a * c)) / a
        return t


class DiffusionSubVPSDE(DiffusionBase):
    """
    Diffusion implementation of the sub-VPSDE. Note that this uses a different SDE compared to the above two diffusions.
    """

    def __init__(self, args):
        super().__init__(args)
        self.beta_start = args.beta_start
        self.beta_end = args.beta_end

        # auxiliary constants (assumes regular VPSDE)
        self.delta_beta_half = torch.tensor(0.5 * (self.beta_end - self.beta_start), device='cuda')
        self.beta_frac = torch.tensor(self.beta_start / (self.beta_end - self.beta_start), device='cuda')
        self.const_aq = (1.0 - self.sigma2_0) * torch.exp(0.5 * self.beta_frac) * torch.sqrt(
            0.25 * np.pi / self.delta_beta_half)
        self.const_erf = torch.erf(torch.sqrt(self.delta_beta_half) * (self.time_eps + self.beta_frac))
        self.const_norm = self.const_aq * (
                torch.erf(torch.sqrt(self.delta_beta_half) * (1.0 + self.beta_frac)) - self.const_erf)
        self.const_norm_2 = torch.erf(torch.sqrt(self.delta_beta_half) * (1.0 + self.beta_frac)) - self.const_erf

    def f(self, t):
        return -0.5 * self.beta(t)

    def g2(self, t):
        return self.beta(t) * (1.0 - torch.exp(-2.0 * self.beta_start * t - (self.beta_end - self.beta_start) * t * t))

    def var(self, t):
        int_term = torch.exp(-self.beta_start * t - 0.5 * (self.beta_end - self.beta_start) * t * t)
        return torch.square(1.0 - int_term) + self.sigma2_0 * int_term

    def e2int_f(self, t):
        return torch.exp(-0.5 * self.beta_start * t - 0.25 * (self.beta_end - self.beta_start) * t * t)

    def beta(self, t):
        """ auxiliary beta function """
        return self.beta_start + (self.beta_end - self.beta_start) * t

    def inv_var(self, var):
        raise NotImplementedError

    def var_vpsde(self, t):
        return 1.0 - (1.0 - self.sigma2_0) * torch.exp(
            -self.beta_start * t - 0.5 * (self.beta_end - self.beta_start) * t * t)

    def inv_var_vpsde(self, var):
        c = torch.log((1 - var) / (1 - self.sigma2_0))
        a = self.beta_end - self.beta_start
        t = (-self.beta_start + torch.sqrt(np.square(self.beta_start) - 2 * a * c)) / a
        return t


class DiffusionVESDE(DiffusionBase):
    """
    Diffusion implementation of the VESDE with dz = sqrt(beta(t)) * dW
    """

    def __init__(self, args):
        super().__init__(args)
        self.sigma2_min = args.sigma2_min
        self.sigma2_max = args.sigma2_max
        assert self.sigma2_min == self.sigma2_0, "VESDE was proposed implicitly assuming sigma2_min = sigma2_0!"

    def f(self, t):
        return torch.zeros_like(t, device='cuda')

    def g2(self, t):
        return self.sigma2_min * np.log(self.sigma2_max / self.sigma2_min) * ((self.sigma2_max / self.sigma2_min) ** t)

    def var(self, t):
        return self.sigma2_min * ((self.sigma2_max / self.sigma2_min) ** t) - self.sigma2_min + self.sigma2_0

    def e2int_f(self, t):
        return torch.ones_like(t, device='cuda')

    def inv_var(self, var):
        return torch.log((var + self.sigma2_min - self.sigma2_0) / self.sigma2_min) / np.log(
            self.sigma2_max / self.sigma2_min)

    def var_N(self, t):
        return 1.0 - self.sigma2_min + self.sigma2_min * ((self.sigma2_max / self.sigma2_min) ** t)

    def inv_var_N(self, var):
        return torch.log((var + self.sigma2_min - 1.0) / self.sigma2_min) / np.log(self.sigma2_max / self.sigma2_min)
