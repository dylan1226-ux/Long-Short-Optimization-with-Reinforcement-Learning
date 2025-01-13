import json
import os
from typing import Optional, Tuple
from datetime import datetime
import fire

import numpy as np
from sb3_contrib.ppo_mask import MaskablePPO
from stable_baselines3.common.callbacks import BaseCallback
from alphagen.data.calculator import AlphaCalculator

from alphagen.data.expression import *
# from alphagen.models.alpha_pool import AlphaPool, AlphaPoolBase
from alphagen.models.sharpe_pool import AlphaPoolSharpe, SharpePoolBase
from alphagen.rl.env.wrapper import AlphaEnv
from alphagen.rl.policy import LSTMSharedNet
from alphagen.utils.random import reseed_everything
from alphagen.rl.env.core import AlphaEnvCore
# from alphagen_qlib.calculator import QLibStockDataCalculator
from alphagen_qlib.calculator_sharp import SharpeCalculator
from alphagen_qlib.my_data import *
import torch


class CustomCallback(BaseCallback):
    def __init__(self,
                 save_freq: int,
                 show_freq: int,
                 save_path: str,
                 test_calculator: AlphaCalculator,
                 name_prefix: str = 'rl_model',
                 timestamp: Optional[str] = None,
                 verbose: int = 0):
        super().__init__(verbose)
        self.save_freq = save_freq
        self.show_freq = show_freq
        self.save_path = save_path
        self.name_prefix = name_prefix


        self.test_calculator = test_calculator

        if timestamp is None:
            self.timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        else:
            self.timestamp = timestamp

    def _init_callback(self) -> None:
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_end(self) -> None:
        assert self.logger is not None
        self.logger.record('pool/size', self.pool.size)
        self.logger.record('pool/significant', (np.abs(self.pool.weights[:self.pool.size]) > 1e-4).sum())
        self.logger.record('pool/best_sharpe_ret', self.pool.best_sharpe_ret)
        self.logger.record('pool/eval_cnt', self.pool.eval_cnt)
        sharpe = self.pool.test_ensemble(self.test_calculator)
        self.logger.record('test/sharpe', sharpe)
        # self.logger.record('test/rank_ic', rank_ic_test)
        self.save_checkpoint()

    def save_checkpoint(self):
        path = os.path.join(self.save_path, f'{self.name_prefix}_{self.timestamp}', f'{self.num_timesteps}_steps')
        self.model.save(path)   # type: ignore
        if self.verbose > 1:
            print(f'Saving model checkpoint to {path}')
        with open(f'{path}_pool.json', 'w') as f:
            json.dump(self.pool.to_dict(), f)

    def show_pool_state(self):
        state = self.pool.state
        n = len(state['exprs'])
        print('---------------------------------------------')
        for i in range(n):
            weight = state['weights'][i]
            expr_str = str(state['exprs'][i])
            ic_ret = state['ics_ret'][i]
            print(f'> Alpha #{i}: {weight}, {expr_str}, {ic_ret}')
        print(f'>> Ensemble ic_ret: {state["best_ic_ret"]}')
        print('---------------------------------------------')

    @property
    def pool(self) -> SharpePoolBase:
        return self.env_core.pool

    @property
    def env_core(self) -> AlphaEnvCore:
        return self.training_env.envs[0].unwrapped  # type: ignore


def main(
    seed: int = 1,
    instruments: str = "all",
    pool_capacity: int = 10,
    steps: int = 200_000
):
    reseed_everything(seed)

    device = torch.device('cuda')


    # Define and create directories for checkpoints and TensorBoard logs
    checkpoint_directory = '/tmp/pycharm_project_877/checkpoints/v1'
    tensorboard_log_path = '/tmp/pycharm_project_877/tb_logs/v1'

    # Create directories if they do not exist
    os.makedirs(checkpoint_directory, exist_ok=True)
    os.makedirs(tensorboard_log_path, exist_ok=True)

    # if csv file-input
    csv_path_file = '/tmp/pycharm_project_877/company_data/processed_data/in_sample_18_22.joblib'

    # You can re-implement AlphaCalculator instead of using QLibStockDataCalculator.
    data_train = StockData2(csv_path=csv_path_file,
                            instrument='all',
                           start_time='2018-01-01',
                           end_time='2021-12-31')

    data_test = StockData2(csv_path=csv_path_file,
                            instrument='all',
                          start_time='2022-01-02',
                          end_time='2022-12-31')

    calculator_train = SharpeCalculator(data_train)
    calculator_test = SharpeCalculator(data_test)

    pool = AlphaPoolSharpe(
        capacity=pool_capacity,
        calculator=calculator_train,
        l1_alpha=5e-3
    )
    env = AlphaEnv(pool=pool, device=device, print_expr=True)

    name_prefix = f"new_{instruments}_{pool_capacity}_{seed}"
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')


    checkpoint_callback = CustomCallback(
        save_freq=10000,
        show_freq=10000,
        save_path=checkpoint_directory,
        test_calculator=calculator_test,
        name_prefix=name_prefix,
        timestamp=timestamp,
        verbose=1,
    )


    model = MaskablePPO(
        'MlpPolicy',
        env,
        policy_kwargs=dict(
            features_extractor_class=LSTMSharedNet,
            features_extractor_kwargs=dict(
                n_layers=2,
                d_model=128,
                dropout=0.1,
                device=device,
            ),
        ),
        gamma=1.,
        ent_coef=0.01,
        batch_size=128,
        tensorboard_log=tensorboard_log_path,
        device=device,
        verbose=1,
    )

    model.learn(
        total_timesteps=steps,
        callback=checkpoint_callback,
        tb_log_name=f'{name_prefix}_{timestamp}',
    )


def fire_helper(
    seed: Union[int, Tuple[int]],
    code: str,
    pool: int,
    step: int = None
):
    if isinstance(seed, int):
        seed = (seed, )
    default_steps = {
        10: 250_000,
        20: 300_000,
        50: 350_000,
        100: 400_000
    }
    for _seed in seed:
        main(_seed,
             code,
             pool,
             default_steps[int(pool)] if step is None else int(step)
             )


if __name__ == '__main__':
    fire.Fire(fire_helper)

# if __name__ == '__main__':
#     # Call main with specific parameters
#     main(
#         seed=1,               # Specify the seed value
#         instruments="all",    # Specify the instruments
#         pool_capacity=10,     # Set pool capacity
#         steps=8000  # Define the total number of training steps
#     )

