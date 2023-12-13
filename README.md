# nlp_fast_unlearning

This repository implements a fast machine unlearning algorithm that utilizes error-maximizing noisy samples to unlearn target classes of a text classifier trained with DBPedia in just one epoch. Unlike Tarun et al. [1], my experiments show that a repair epoch isn't always necessary for retaining the performance on the classes that are not to be forgotten.

## Installation

Please run `pip install git+https://github.com/shouborno/nlp_fast_unlearning.git#egg=nlp_fast_unlearning`. If you wish to modify the modules on the go, you can install with `pip install -e .` instead.

## References
1. Tarun, A.K., Chundawat, V.S., Mandal, M. and Kankanhalli, M., 2023. Fast yet effective machine unlearning. IEEE Transactions on Neural Networks and Learning Systems.
