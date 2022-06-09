# Adversarial Mixture Density Networks

This is the repo for paper Adversarial Mixture Density Networks: Learning to Drive Safely from Collision Data. 

Trains a mixture density networks on two datasets, 
one with positive examples (demonstrated safe driving)
and one with negative examples (trajectories leading to collision), training an action distribution for both sets. 

The model maximises likelihood of actions of closer to
the safe action distribution and minimises likelihood of actions closer to unsafe action distribution, leading to safer driving policies.
Trained and tested in a vehicle following setting.

For further details see the paper: https://arxiv.org/abs/2107.04485


## Installation
Clone the repo

```bash
git clone https://github.com/sampo-kuutti/adversarial-mixture-density-networks
```

install requirements:
```bash
pip install -r requirements.txt
```

## Training the model


To train the AMDN model run `train_amdn.py`.

## Citing the Repo

If you find the code useful in your research or wish to cite it, please use the following BibTeX entry.

```text
@inproceedings{kuutti2021adversarial,
  title={Adversarial Mixture Density Networks: Learning to Drive Safely from Collision Data},
  author={Kuutti, Sampo and Fallah, Saber and Bowden, Richard},
  booktitle={2021 IEEE International Intelligent Transportation Systems Conference (ITSC)},
  pages={705--711},
  year={2021},
  organization={IEEE}
}
```