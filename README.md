IDQL Implementation REPO!
Make sure you setup your Wandb key and pip install requirements


## Reproducing Results

[Offline.](launcher/examples/train_ddpm_iql_offline.py)

Run
```
python3 launcher/examples/train_ddpm_iql_offline.py --variant 0...N
```

[Finetune.](launcher/examples/train_ddpm_iql_finetune.py)

Run 
```
python3 launcher/examples/train_ddpm_iql_finetune.py --variant 0...N
```
Specific File Paths that are Important

[Main run script were variant dictionary is passed.](/examples/states/train_diffusion_offline.py)

[DDPM Implementation.](/jaxrl5/networks/diffusion.py)

[LN_Resnet.](/jaxrl5/networks/resnet.py)

[DDPM IQL Learner.](/jaxrl5/agents/ddpm_iql/ddpm_iql_learner.py)

[![CircleCI](https://dl.circleci.com/status-badge/img/gh/ikostrikov/jaxrl5/tree/main.svg?style=svg&circle-token=668374ebe0f27c7ee70edbdfbbd1dd928725c01a)](https://dl.circleci.com/status-badge/redirect/gh/ikostrikov/jaxrl5/tree/main) [![codecov](https://codecov.io/gh/ikostrikov/jaxrl5/branch/main/graph/badge.svg?token=Q5QMIDZNZ3)](https://codecov.io/gh/ikostrikov/jaxrl5)

## Installation

Run
```bash
pip install --upgrade pip

pip install -e .
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

See instructions for other versions of CUDA [here](https://github.com/google/jax#pip-installation-gpu-cuda).
```

Based from a re-implementation of https://github.com/ikostrikov/jaxrl 
