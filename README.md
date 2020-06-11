# EPIC-Kitchens
This project is the result of an attempt in EPIC-Kitchens Challenge. We choose TSN (Temporal Segment Networks) as a backbone for initial classification. Further I try to improve the result using various ways (listed below in Algorithms part), though didn't succeed much, but learnt alot! 

Here I want to share my repo which comprise of from scratch implementation (including some referred code) with fully integrated module for action classification workflow.

Backbone : Temporal Segment Networks (TSN) [1]

Algorithms [options to choose in config] on top of backbone:
    1. "ERM" : Vanila TSN
    2. "FSL" : Few Shot Learning [2]
    3. "IRM" : Invariant Risk Minimization [3]
    4. "MTGA": Multi-Task Learning as Multi-Objective Optimization [4]

Dataset : EPIC-KITCHENS Challenge [5]
    Dataset is downloaded in the form of frames. 
    [options to edit the annotation and raw files path in config]
    
How to run?
  python -W ignore main.py --config config.yaml
  
  (Edit config.yaml and config.py for trying various specifications)


[1] @misc{wang2016temporal,
    title={Temporal Segment Networks: Towards Good Practices for Deep Action Recognition},
    author={Limin Wang and Yuanjun Xiong and Zhe Wang and Yu Qiao and Dahua Lin and Xiaoou Tang and Luc Van Gool},
    year={2016},
    eprint={1608.00859},
    archivePrefix={arXiv},
    primaryClass={cs.CV}
    }
    
[2] https://github.com/oscarknagg/few-shot

[3] @misc{arjovsky2019invariant,
    title={Invariant Risk Minimization},
    author={Martin Arjovsky and Léon Bottou and Ishaan Gulrajani and David Lopez-Paz},
    year={2019},
    eprint={1907.02893},
    archivePrefix={arXiv},
    primaryClass={stat.ML}
    }
    
[4] @incollection{NeurIPS2018_Sener_Koltun,
    title = {Multi-Task Learning as Multi-Objective Optimization},
    author = {Sener, Ozan and Koltun, Vladlen},
    booktitle = {Advances in Neural Information Processing Systems 31},
    editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
    pages = {525--536},
    year = {2018},
    publisher = {Curran Associates, Inc.},
    url = {http://papers.nips.cc/paper/7334-multi-task-learning-as-multi-objective-optimization.pdf}
    }

[5] https://epic-kitchens.github.io/2020
