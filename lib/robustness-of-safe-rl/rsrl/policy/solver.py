import numpy as np
import torch
import torch.nn as nn
from scipy.optimize import minimize
from torch.distributions.uniform import Uniform
from torch.distributions import MultivariateNormal
from torch.optim import Adam
from torch.nn.utils import clip_grad_norm_

from rsrl.util.torch_util import (to_ndarray, to_tensor)


class ScipySolver:
    def __init__(self, kl_thres, qc_thres, eta_init=0.1, lam_init=0.1) -> None:
        super().__init__()
        self.kl_thres = kl_thres
        self.qc_thres = qc_thres
        self.eta = eta_init
        self.lam = lam_init

        # scipy optimizer  config
        self.bounds = [(1e-6, 20), (1e-6, 20)]
        self.options = {"ftol": 1e-3, "maxiter": 20}
        self.method = 'SLSQP'
        self.tol = 1e-3

    def solve(self, q, qc):
        '''
        @param q, [tensor], (sample actions, batch_size)
        @param qc, [tensor], (sample actions, batch_size)
        '''
        q = to_ndarray(q).T  # (batch, sample actions)
        qc = to_ndarray(qc).T  # (batch, sample actions)

        def dual_loss(x):
            """
            dual function of the non-parametric variational
            """
            η, lam = x
            target_q_np_comb = q - lam * qc
            max_q = np.max(target_q_np_comb, 1)
            return η * self.kl_thres + lam * self.qc_thres + np.mean(max_q) \
                + η * np.mean(np.log(np.mean(np.exp((target_q_np_comb - max_q[:, None]) / η), axis=1)))

        res = minimize(dual_loss,
                       np.array([self.eta, self.lam]),
                       method='SLSQP',
                       bounds=self.bounds,
                       tol=self.tol,
                       options=self.options)

        self.eta, self.lam = res.x
        return self.eta, self.lam, 1


class TorchSolver:
    def __init__(self,
                 kl_thres,
                 qc_thres,
                 eta_init=1,
                 lam_init=1,
                 lr=0.1,
                 bounds=[1e-6, 20],
                 tol=1e-3,
                 max_iterations=20,
                 use_prev_init=False,
                 logsumexp=True) -> None:
        super().__init__()
        self.kl_thres = kl_thres
        self.qc_thres = qc_thres
        self.eta_init = eta_init
        self.lam_init = lam_init
        self.eta = torch.nn.Parameter(to_tensor([eta_init]), requires_grad=True)
        self.lam = torch.nn.Parameter(to_tensor([lam_init]), requires_grad=True)
        self.optimizer = Adam([self.eta, self.lam], lr=lr)

        # optimizer config
        self.bounds = bounds
        self.tol = tol
        self.max_iterations = max_iterations
        self.use_prev_init = use_prev_init
        self.logsumexp = logsumexp

    def solve(self, q, qc, kl_thres=None, qc_thres=None):
        '''
        @param q, [tensor], (sample actions, batch_size)
        @param qc, [tensor], (sample actions, batch_size)
        '''
        if kl_thres is not None:
            self.kl_thres = kl_thres
        if qc_thres is not None:
            self.qc_thres = qc_thres

        q = q.T.detach()
        qc = qc.T.detach()  # (batch_size, sample actions)

        if not self.use_prev_init:
            self.eta.data = to_tensor([self.eta_init])
            self.lam.data = to_tensor([self.lam_init])

        K = qc.shape[1]

        def dual_loss(eta, lam):
            """
            dual function of the non-parametric variational
            """
            target_q_comb = q - lam * qc
            max_q, _ = torch.max(target_q_comb, dim=1, keepdim=True)  # (batch, 1)
            return eta * self.kl_thres + lam * self.qc_thres + torch.mean(max_q) \
                + eta * torch.mean(
                                torch.log(
                                    torch.mean(
                                        torch.exp((target_q_comb - max_q) / eta), dim=1,
                                        )
                                    )
                                )

        def dual_loss2(eta, lam):
            target_q_comb = (q - lam * qc) / eta  # (batch_size, sample actions)
            return eta * self.kl_thres + lam * self.qc_thres + eta * torch.mean(
                torch.logsumexp(target_q_comb, dim=1) - np.log(K))

        iteration = 0
        loss_prev = 9999999
        while iteration < self.max_iterations:
            self.optimizer.zero_grad()
            if self.logsumexp:
                loss = dual_loss2(self.eta, self.lam)
            else:
                loss = dual_loss(self.eta, self.lam)
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                self.eta.clamp_(self.bounds[0], self.bounds[1])
                self.lam.clamp_(self.bounds[0], self.bounds[1])

            loss = loss.item()
            # print(iteration, " loss diff: ", np.abs(loss - loss_prev))
            # early stop
            if np.abs(loss - loss_prev) < self.tol:
                break
            iteration += 1
            loss_prev = loss

        return self.eta.item(), self.lam.item(), iteration


class TorchSolverV2:
    def __init__(self,
                 kl_thres,
                 qc_thres,
                 eta_init=1,
                 lam_init=1,
                 lr=0.1,
                 bounds=[1e-6, 20],
                 tol=1e-3,
                 max_iterations=20,
                 use_prev_init=False,
                 logsumexp=True) -> None:
        super().__init__()
        self.kl_thres = kl_thres
        self.qc_thres = qc_thres
        self.eta_init = eta_init
        self.lam_init = lam_init
        self.eta = torch.nn.Parameter(to_tensor([eta_init]), requires_grad=True)
        self.lam = torch.nn.Parameter(to_tensor([lam_init]), requires_grad=True)
        self.optimizer = Adam([self.eta, self.lam], lr=lr)

        # optimizer config
        self.bounds = bounds
        self.tol = tol
        self.max_iterations = max_iterations
        self.use_prev_init = use_prev_init
        self.logsumexp = logsumexp

    def solve(self, q, qc, pi, kl_thres=None, qc_thres=None):
        '''
        @param q, [tensor], (sample actions, batch_size)
        @param qc, [tensor], (sample actions, batch_size)
        '''
        if kl_thres is not None:
            self.kl_thres = kl_thres
        if qc_thres is not None:
            self.qc_thres = qc_thres

        q = q.T.detach()
        qc = qc.T.detach()  # (batch_size, sample actions)

        if not self.use_prev_init:
            self.eta.data = to_tensor([self.eta_init])
            self.lam.data = to_tensor([self.lam_init])

        K = qc.shape[1]

        def dual_loss(eta, lam):
            """
            dual function of the non-parametric variational
            """
            target_q_comb = q - lam * qc
            max_q, _ = torch.max(target_q_comb, dim=1, keepdim=True)  # (batch, 1)
            return eta * self.kl_thres + lam * self.qc_thres + torch.mean(max_q) \
                + eta * torch.mean(
                                torch.log(
                                    torch.mean(
                                        torch.exp((target_q_comb - max_q) / eta), dim=1,
                                        )
                                    )
                                )

        def dual_loss2(eta, lam):
            target_q_comb = (q - lam * qc) / eta  # (batch_size, sample actions)
            return eta * self.kl_thres + lam * self.qc_thres + eta * torch.mean(
                torch.logsumexp(target_q_comb, dim=1) - np.log(K))

        iteration = 0
        loss_prev = 9999999
        while iteration < self.max_iterations:
            self.optimizer.zero_grad()
            if self.logsumexp:
                loss = dual_loss2(self.eta, self.lam)
            else:
                loss = dual_loss(self.eta, self.lam)
            loss.backward()
            self.optimizer.step()
            with torch.no_grad():
                self.eta.clamp_(self.bounds[0], self.bounds[1])
                self.lam.clamp_(self.bounds[0], self.bounds[1])

            loss = loss.item()
            # print(iteration, " loss diff: ", np.abs(loss - loss_prev))
            # early stop
            if np.abs(loss - loss_prev) < self.tol:
                break
            iteration += 1
            loss_prev = loss

        return self.eta.item(), self.lam.item(), iteration