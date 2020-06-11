# EPIC-Kitchens
This project is the result of an attempt in EPIC-Kitchens Challenge. We choose TSN (Temporal Segment Networks) as a backbone for initial classification. Further I try to improve the result using various ways (listed below in Algorithms part), though didn't succeed much, but learnt alot! 

Here I want to share my repo which comprise of from scratch implementation (including some referred code) with fully integrated module for action classification workflow.

## Backbone
    Temporal Segment Networks (TSN) [1]

## Algorithms [options to choose in config] on top of backbone:
    1. "ERM" : Vanila TSN
    2. "FSL" : Few Shot Learning [2]
    3. "IRM" : Invariant Risk Minimization [3]
    4. "MTGA": Multi-Task Learning as Multi-Objective Optimization [4]

## Dataset : EPIC-KITCHENS Challenge [5]
    Dataset is downloaded in the form of frames. 
    [options to edit the annotation and raw files path in config]
    
## How to run?
    python -W ignore main.py --config config.yaml
  
    (Edit config.yaml and config.py for trying various specifications)


[1] Wang, L., Xiong, Y., Wang, Z., Qiao, Y., Lin, D., Tang, X. and Van Gool, L., 2016, October. Temporal segment networks: Towards good practices for deep action recognition. In European conference on computer vision (pp. 20-36). Springer, Cham.
    
[2] https://github.com/oscarknagg/few-shot

[3] Arjovsky, M., Bottou, L., Gulrajani, I. and Lopez-Paz, D., 2019. Invariant risk minimization. arXiv preprint arXiv:1907.02893.
    
[4] Sener, O. and Koltun, V., 2018. Multi-task learning as multi-objective optimization. In Advances in Neural Information Processing Systems (pp. 527-538).

[5] https://epic-kitchens.github.io/2020
