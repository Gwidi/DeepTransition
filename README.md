





# Deep Transition Repository #
This repository builds upon an implementation (Simulation code) of the CPG-RL framework in the following papers:
&nbsp;

```
@article{shafiee2024Viability,
  title={Viability leads to the emergence of gait transitions in learning agile quadrupedal locomotion on challenging terrains},
  author={Shafiee, Milad and Bellegarda, Guillaume and Ijspeert, Auke },
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={3073},
  year={2024},
  publisher={Nature Publishing Group UK London}
}
@article{bellegarda_allgaits_2024,
	title = {{AllGaits}: Learning All Quadruped Gaits and Transitions},
	url = {http://arxiv.org/abs/2411.04787},
	doi = {10.48550/arXiv.2411.04787},
	shorttitle = {{AllGaits}},
	number = {{arXiv}:2411.04787},
	publisher = {{arXiv}},
	author = {Bellegarda, Guillaume and Shafiee, Milad and Ijspeert, Auke},
	urldate = {2025-12-02},
	date = {2024-11-07},
}
```

##  :hammer: System requirements and Installation (GPU > 8GB and CUDA needed) ##
 We need GPU and Cuda installed for running the Isaac Gym simulator. To train in the default configuration, we recommend a GPU with at least 8GB memory.
  The code tested with GPU RTX-3070 (cuda-11.4) and  GPU RTX-4090 (cuda-11.7). Installation of Nvidia drivers and Cuda may take couple of hours depending on the hardware.
  After having cuda installed and GPU ready, installation ideally will take less than 15 minutes:

:floppy_disk: Create a new python virtual env with python 3.6, 3.7 or 3.8 (3.8 recommended)
   
   - `conda create -n DeepTransitionENV python=3.8`
   - `conda activate DeepTransitionENV`
     
:page_facing_up:  Clone this repository and install dependencies:

- `pip install -r requirements.txt`

   
-  Install pytorch 1.10 with cuda-11.3 or (pytorch 1.11 and cuda-11.4) (tested with GPU RTX-3070 on Ubuntu 18.04 and Ubuntu 20.04)
   - `pip3 install torch==1.10.0+cu113 torchvision==0.11.1+cu113 torchaudio==0.10.0+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html`
   
   or alternatively, Install pytorch 1.13 with cuda-11.7 (tested with GPU RTX-4090 on Ubuntu 20.04)   
   - `pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0 --extra-index-url https://download.pytorch.org/whl/cu117`
   
:cartwheeling:  Install Isaac Gym
   - Install Isaac Gym Preview 3 
   - `cd isaacgym/python && pip install -e .`
   
:chart_with_upwards_trend:  Install rsl_rl (PPO)
   -  `cd rsl_rl  && pip install -e .` 
   
:mechanical_leg:  Install legged_gym
   -  `cd legged_gym && pip install -e .`


## :school: CODE STRUCTURE  ##
The training environment is defined by an env file (`quadruped_with_spine.py`) and a config file ( either `silver_badger_rigid_spine.py` or `silver_badger_active_spine`). Config file include body names, default_joint_positions and PD gains, reward weights,etc. You need to modify the reward weights in quadruped_config.py and train the policy to reproduce the result of the paper.
The config file contains two classes: one containing all the environment parameters (LeggedRobotCfg) and one for the training parameters (LeggedRobotCfgPPo).
 Central Pattern Generator (CPG) is defined by `CPG.py`.
 
## :weight_lifting:  Instructions for training  ##

- Isaac Gym requires an NVIDIA GPU. To train in the default configuration, we recommend a GPU with at least 8GB memory. The code can run on a smaller GPU if you decrease the number of parallel environments (`Cfg.env.num_envs`) or reducing the precision of  terrain meshes. Training will be slower with fewer environments.
 An example for training (training will take less than two hours with the suggest GPUs):
 
  `python legged_gym/scripts/train.py --task=silver_badger_rigid_spine  --max_iterations=3000 --num_envs=4096`
  
 - To run on CPU add following arguments: `--sim_device=cpu`, `--rl_device=cpu` (sim on CPU and rl on GPU is possible). Running on CPU is not encouraged and may lead to different results.
    
 -  To run without no rendering add `--headless`.
    
 -  To improve performance, once the training starts (and if your are not running headless) make sure to press `v` to stop the rendering. You can then enable it later to check the progress.
    
   - The trained policy is saved in `legged_gym/logs/<experiment_name>/<date_time>_<run_name>/model_<iteration>.pt`. Where `<experiment_name>` and `<run_name>` are defined in the train config.
    
   -  The following command line arguments override the values set in the config files:
    
   - `--task` TASK: Task name.
     
   - `--num_envs` NUM_ENVS:  Number of environments to create.
     
   - `--max_iterations` MAX_ITERATIONS:  Maximum number of training iterations.

## Acknowledgement  ##
```
This environment builds on the amazing repository [legged gym environment](https://leggedrobotics.github.io/legged_gym/) by Nikita
Rudin, Robotic Systems Lab, ETH Zurich (Paper: https://arxiv.org/abs/2109.11978) and the Isaac Gym simulator from 
NVIDIA (Paper: https://arxiv.org/abs/2108.10470). Training code builds on the [rsl_rl](https://github.com/leggedrobotics/rsl_rl) 
repository, also by Nikita Rudin, Robotic Systems Lab, ETH Zurich. 
All redistributed code retains its original [license](LICENSES/legged_gym/LICENSE).
```

