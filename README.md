# DeepRP: Bottleneck Theory Guided Relay Placement for 6G Mesh Backhaul Augmentation

This repository provides simplified implementation for a paper that is in submission.

More specifically, some files are highlighted:
* In `./utils/bottleneck.py`, the algorithms of bottleneck structure construction (`BSC`) and clique gradient computation (`CliqueGrad`) are implemented in details based on the the data structure of heap queue.
* In `./ppo_agent.py` and `./drl_train.py`, an actor-critic agent is trained by [proximal policy optimization](https://arxiv.org/abs/1707.06347).
* In `./heuristic_main.py` and `DeepRP/random_main.py`, two baseline relay placement methods are implemented.
* In `./config.py`, a toy example on network configuration is provided.

A more complete and detailed implementation accompanied with real-world datasets will be released upon the acceptance of the paper.


