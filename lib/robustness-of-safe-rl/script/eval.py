import os.path as osp
import os
# from rsrl import evaluator

from rsrl.evaluator import Evaluator
from rsrl.util.run_util import load_config, setup_eval_configs


def gen_data_dir_name(load_dir, noise=False):
    load_dirs = []
    for dir in os.listdir(load_dir):
        env_dir = osp.join(load_dir, dir)
        if not osp.isdir(env_dir):
            continue
        if "AntCircle" in env_dir:
            continue
        for subdir in os.listdir(env_dir):
            if noise and "vanilla" not in subdir:
                continue
            exp_dir = osp.join(env_dir, subdir)
            for seeddir in os.listdir(exp_dir):
                exp = osp.join(exp_dir, seeddir)
                load_dirs.append(exp)
    print(load_dirs)
    return load_dirs


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', '-e', type=str, default='SafetyAntRun-v0')
    parser.add_argument('--policy', '-p', type=str, default='robust_cvpo')
    parser.add_argument('--pretrain_dir', '-pre', type=str, default=None)

    parser.add_argument('--load_dir', '-d', type=str, default=None)
    parser.add_argument('--noise', action="store_true")
    parser.add_argument('--optimal', action="store_true")

    parser.add_argument('--mode', '-m', type=str, default='train')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--device', type=str, default="cpu")
    parser.add_argument('--exp_name', type=str, default=None)
    parser.add_argument('--suffix', '--id', type=str, default=None)
    parser.add_argument('--no_render', action="store_true")
    parser.add_argument('--sleep', type=float, default=0.003)
    args = parser.parse_args()

    load_dir = args.load_dir
    itr = None
    if not args.optimal:
        itr = 0

    load_dirs = [load_dir]

    assert load_dir is not None, "The load_path parameter has not been specified!!!"
    model_path, config = setup_eval_configs(load_dir, itr)

    # update config
    config["evaluate_episode_num"] = 1
    config["epochs"] = args.epochs
    config["data_dir"] += "_eval"
    config["eval_attackers"] = ["amad", "mad", "max_cost", "max_reward", "random"]

    evaluator = Evaluator(**config, config_dict=config)
    evaluator.eval(model_path)