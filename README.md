# Follow-the-Ridge

This repository contains the code to reproduce the Follow-the-Ridge results from the paper [On Solving Minimax Optimization Locally: A Follow-the-Ridge Approach](https://openreview.net/forum?id=Hkx7_1rKwS).

Particularly, you can reproduce our results of GAN on 1-D and 2-D mixture of Gaussians. 

```
# for 1-D MOG
$ python follow_ridge_1D.py --follow_ridge --adapt_damping

# for 2-D MOG
$ python follow_ridge_2D.py --follow_ridge --adapt_damping
```

## Requirements
Python 3.6, Tensorflow 1.14.0

## Citation
To cite this work, please use
```
@inproceedings{
    wang2020on,
    title={On Solving Minimax Optimization Locally: A Follow-the-Ridge Approach},
    author={Yuanhao Wang and Guodong Zhang and Jimmy Ba},
    booktitle={International Conference on Learning Representations},
    year={2020},
    url={https://openreview.net/forum?id=Hkx7_1rKwS}
}
```
