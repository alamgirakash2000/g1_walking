# G1 Walking

## Commands to run this project:

```
# To clone this repo
git clone https://github.com/alamgirakash2000/g1_walking

cd g1_walking

# To install the shell and libraries
./isaaclab.sh -i

# To train the model 
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/train.py --task=Isaac-Velocity-Rough-G1-v0 --headless

# To Test the model
./isaaclab.sh -p scripts/reinforcement_learning/rsl_rl/play.py --task=Isaac-Velocity-Rough-G1-v0 --num_envs=8
```




## For setting up the environment

```
conda create -n env_isaaclab python=3.11
conda activate env_isaaclab


pip install --upgrade pip

pip install torch==2.7.0 torchvision==0.22.0 --index-url https://download.pytorch.org/whl/cu128

pip install "isaacsim[all,extscache]==5.0.0" --extra-index-url https://pypi.nvidia.com


# To verify IsaacSim
isaacsim


sudo apt install cmake build-essential
```