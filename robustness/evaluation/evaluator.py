import os
import time

import imageio
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from statsmodels.stats.proportion import proportion_confint

from robustness.agents import Agent
from robustness.analysis import Problem, Solver, TraceEvaluator
from robustness.analysis.utils import normalize, scale


class Evaluator:
    def __init__(self, problem: Problem, solver: Solver):
        self.problem = problem
        self.solver = solver
        print('Robustness falsification options:')
        print(solver.options())
        print('System evaluation options:')
        print(solver.sys_evaluator.options())
    
    def any_violation(self, boundary=None):
        return self.solver.any_unsafe_deviation(self.problem, boundary)
    
    def min_violation(self, boundary=None):
        return self.solver.min_unsafe_deviation(self.problem, boundary)
    
    def certified_min_violation(self, n=100, alpha=0.05):
        certificated = False
        min_delta, min_dist, min_x0 = None, np.inf, None
        while not certificated:
            delta, dist, x0 = self.min_violation()
            print('CMA search min deviation:', delta, dist)

            if dist < min_dist:
                min_delta, min_dist, min_x0 = delta, dist, x0

            lower_bound, violation, violated_x0 = self.certify(min_dist, n, alpha)
            print('Certify:', lower_bound, violation)

            if violation is None:
                certificated = True
            else:
                violated_dist = self.problem.dist.eval_dist(violation)
                self.solver.sigma = violated_dist / 2
                min_delta, min_dist, min_x0 = violation, violated_dist, violated_x0

        return min_delta, min_dist, min_x0

    def visualize_violation(self, delta, x0=None, gif=None, **kwargs):
        env, _ = self.problem.env.instantiate(delta, **kwargs)
        visualizer = EpisodeVisualizer(env, self.problem.agent, self.problem.phi)
        if x0 is not None:
            visualizer.visual_episode(
                self.solver.sys_evaluator.options()['episode_len'],
                x0,
                gif=gif,
            )
        else:
            v, x0 = self.solver.sys_evaluator.eval_sys(delta, self.problem)
            if v < 0:
                v = np.inf
                for _ in range(100):
                    v = visualizer.visual_episode(
                        self.solver.sys_evaluator.options()['episode_len'],
                        x0,
                        gif=gif,
                    )
                    if v < 0:
                        break
                if v > 0:
                    print("WARNING: Unable to reproduce the counterexample.")
            else:
                visualizer.visual_episode(
                    self.solver.sys_evaluator.options()['episode_len'],
                    x0,
                    gif=gif,
                )
        env.close()
    
    def grid_data(self, x_bound, y_bound, n_x, n_y, override=False, out_dir='data'):
        if not os.path.exists(f'{out_dir}/Z.csv') or override:
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
                
            X = np.linspace(x_bound[0], x_bound[1], n_x)
            Y = np.linspace(y_bound[0], y_bound[1], n_y)
            X, Y = np.meshgrid(X, Y, indexing='ij')

            Z = np.zeros((n_x, n_y))
            for i in range(n_x):
                for j in range(n_y):
                    # treat xv[i,j], yv[i,j]
                    x, y = X[i, j], Y[i, j]
                    v, _ = self.solver.sys_evaluator.eval_sys([x, y], self.problem)
                    Z[i, j] = v

            np.savetxt(f"{out_dir}/X.csv", X, delimiter=",")
            np.savetxt(f"{out_dir}/Y.csv", Y, delimiter=",")
            np.savetxt(f"{out_dir}/Z.csv", Z, delimiter=",")
        else:
            X = np.loadtxt(f"{out_dir}/X.csv", delimiter=",")
            Y = np.loadtxt(f"{out_dir}/Y.csv", delimiter=",")
            Z = np.loadtxt(f"{out_dir}/Z.csv", delimiter=",")
        
        return X, Y, Z
    
    def gridplot(self, x_bound, y_bound, n_x, n_y, x_name="X", y_name="Y", z_name="Z",
                  override=False, out_dir='data', boundary=None, **kwargs):
        X, Y, Z = self.grid_data(x_bound, y_bound, n_x, n_y, override, out_dir)

        _, ax = plt.subplots(subplot_kw={"projection": "3d"}, figsize=(14, 14))
        
        if boundary is not None:
            mask = np.asarray([
                [self.problem.dist.eval_dist([x, y]) > boundary for x, y in zip(xs, ys)]
                for xs, ys in zip(X, Y)
            ])
            ax.plot_surface(X, Y, np.ma.masked_where(mask, Z), cmap=cm.coolwarm, **kwargs)
            ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, alpha=0.25, **kwargs)
        else:
            ax.plot_surface(X, Y, Z, cmap=cm.coolwarm, **kwargs)
        
        ax.set_zlabel(z_name, fontsize=13)
        ax.set_xlabel(x_name, fontsize=13)
        ax.set_ylabel(y_name, fontsize=13)

        return ax, X, Y, Z
    
    def heatmap(self, x_bound, y_bound, n_x, n_y, x_name="X", y_name="Y", z_name="Z",
                override=False, out_dir='data', boundary=None, **kwargs):
        X, Y, Z = self.grid_data(x_bound, y_bound, n_x, n_y, override, out_dir)
        
        _, ax = plt.subplots()
        im = ax.imshow(Z.T, cmap=cm.coolwarm, **kwargs)
        
        ax.set_xticks(np.arange(0, len(X[:, 0]), 3), labels=['{:.2f}'.format(x) for x in X[:, 0][::3]])
        ax.set_yticks(np.arange(0, len(Y[0]), 3), labels=['{:.2f}'.format(y) for y in Y[0][::3]])
        cbar = ax.figure.colorbar(im)
        cbar.ax.set_ylabel(z_name, rotation=-90, va="bottom")

        ax.set_xlabel(x_name)
        ax.set_ylabel(y_name)

        if boundary is not None:
            center = normalize(self.problem.env.get_delta_0(), self.problem.env.get_dev_bounds())
            c_X = np.linspace(center[0] - boundary, center[0] + boundary, 100)
            c_Y1 = center[1] + np.sqrt(np.clip(boundary**2 - (c_X - center[0])**2, 0, None))
            c_Y2 = center[1] - np.sqrt(np.clip(boundary**2 - (c_X - center[0])**2, 0, None))
            
            c_X = np.clip(c_X, 0.0, None)
            c_Y2 = np.clip(c_Y2, 0.0, None)

            ax.scatter(center[0] * n_x, center[1] * n_y, color='black')
            ax.plot(c_X * n_x, c_Y1 * n_y, color='black')
            ax.plot(c_X * n_x, c_Y2 * n_y, color='black')

        plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
        return ax, X, Y, Z
    
    def certify(self, dist, n=100, alpha=0.05):
        '''
        Return <lower bound?, violated delta?, violated x0?>.
        '''
        bounds = self.problem.env.get_dev_bounds()
        center = normalize(self.problem.env.get_delta_0(), bounds)
        low, high = np.clip(center - dist, 0.0, 1.0), np.clip(center + dist, 0.0, 1.0)

        samples = []
        values = []
        x0s = []
        c = 0
        while c < n:
            delta = scale(np.random.uniform(low, high), bounds)
            if self.problem.dist.eval_dist(delta) > dist:
                continue

            c += 1
            samples.append(delta)
            v, x0 = self.solver.sys_evaluator.eval_sys(delta, self.problem)
            values.append(v)
            x0s.append(x0)
        
        values = np.asarray(values)
        idx = values.argmin()
        violation, violated_x0 = (samples[idx], x0s[idx]) if values[idx] < 0 else (None, None)
        k = np.sum(values >= 0)
        return proportion_confint(k, n, alpha * 2, method='beta')[0], violation, violated_x0


class EpisodeVisualizer:
    def __init__(self, env, agent: Agent, phi: TraceEvaluator) -> None:
        self.env = env
        self.agent = agent
        self.phi = phi
        self.img = None
        self.fig = None
        self.ax = None

    def _init_fig(self):
        self.fig, self.ax = plt.subplots()
        self.ax.set_title(
            f"{self.env.spec.id}\n" +
            f"Step: 0 | Reward: 0.000 | Done: False\n" +
            f"{self.phi}: 0.000",
            {'fontsize': 13}
        )
        self.ax.set_axis_off()
        self.img = self.ax.imshow(self.env.render(mode='rgb_array'))
    
    def _update_fig(self, step, reward, done, phi_val):
        title = self.ax.set_title(
            f"{self.env.spec.id}\n" +
            f"Step: {step} | Reward: {reward:.3f} | Done: {done}\n" +
            f"{self.phi}: {phi_val:.3f}",
            {'color': 'r' if done or phi_val < 0 else 'k', 'fontsize': 13}
        )
        self.img.set_data(self.env.render(mode='rgb_array'))
        self.fig.canvas.draw()
        # display.display(fig)
        # display.clear_output(wait=True)
        return np.array(self.fig.canvas.buffer_rgba())

    def visual_episode(self, episode_len, x0=None, sleep=0.01, gif=None):
        if gif is not None:
            os.makedirs(os.path.dirname(gif), exist_ok=True)

        if x0 is not None:
            obs = self.env.reset_to(x0)
        else:
            obs = self.env.reset()
        self.agent.reset()
        if gif is not None:
            gif_data = []
            self._init_fig()
        else:
            self.env.render()
        space = self.env.observation_space
        total_reward = 0.0
        obs_record = [obs]
        reward_record = [0]

        for step in range(1, episode_len+1):
            action = self.agent.next_action(obs)
            obs, reward, done, _ = self.env.step(action)
            total_reward += reward
            obs_record.append(np.clip(obs, space.low, space.high))
            reward_record.append(reward)

            if gif is not None:
                fig_data = self._update_fig(
                    step, total_reward, done,
                    self.phi.eval_trace(np.array(obs_record), np.array(reward_record))
                )
                gif_data.append(fig_data)
            else:
                self.env.render()
                if sleep > 0.0:
                    time.sleep(sleep)

        if gif is not None:
            imageio.mimsave(gif, [data for data in gif_data], fps=10)

        return self.phi.eval_trace(np.array(obs_record), np.array(reward_record))
