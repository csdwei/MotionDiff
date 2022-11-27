# MotionDiff
Code for AAAI2023 paper "Human Joint Kinematics Diffusion-Refinement for Stochastic Motion Prediction"

By Dong Wei, Huaijiang Sun, Bin Li, Jianfeng Lu, Weiqing Li, Xiaoning Sun and Shengxiang Hu

> Stochastic human motion prediction aims to forecast multiple plausible future motions given a single pose sequence from the past. Most previous works focus on designing elaborate losses to improve the accuracy, while the diversity is typically characterized by randomly sampling a set of latent variables from the latent prior, which is then decoded into possible motions. This joint training of sampling and decoding, however, suffers from posterior collapse as the learned latent variables tend to be ignored by a strong decoder, leading to limited diversity. Alternatively, inspired by the diffusion process in nonequilibrium thermodynamics, we propose MotionDiff, a diffusion probabilistic model to treat the kinematics of human joints as heated particles, which will diffuse from original states to a noise distribution. This process not only offers a natural way to obtain the ``whitened'' latents without any trainable parameters, but also introduces a new noise in each diffusion step, both of which facilitate more diverse motions. Human motion prediction is then regarded as the reverse diffusion process that converts the noise distribution into realistic future motions conditioned on the observed sequence. Specifically, MotionDiff consists of two parts: a spatial-temporal transformer-based diffusion network to generate diverse yet plausible motions, and a flexible refinement network to further enable geometric losses and align with the ground truth. Experimental results on two datasets demonstrate that our model yields the competitive performance in terms of both diversity and accuracy. 


# Code

## Environment
    PyTorch == 1.7.1
    CUDA > 10.1


## Training & Evaluation 

### Step 1: Modify or create your own config file in ```/configs``` 

You can revise parameters and seeds in config file as you like and change the network architecture of the diffusion model in ```models/Diffusion.py```

### Step 2: Train the Diffusion Network

 ```python main.py --config configs/baseline.yaml --mode train_diff``` 

 Logs and checkpoints will be automatically saved.

### Step 3: Generate the motions by Diffusion Network

 ```python main.py --config configs/baseline.yaml --mode generate_diff``` 

 Since the sampling process of the diffusion network may take a long time, we generate future motions in advance. The obtained motions will be saved in ```./results/generated_diff```

### Step 4: Train the Refinement Network

```python main.py --config configs/baseline.yaml --mode train_refine```

Logs and checkpoints will be automatically saved.

### Step 5: Evaluation  

```python main.py --config configs/baseline.yaml --mode test```

Evaluation for diffusion network and diffusion-refinement architecture, including statistics (APD, ADE, FDE, MMADE, MMFDE) and visualizations.

### Citation
```
    @article{wei2022human,
    title={Human Joint Kinematics Diffusion-Refinement for Stochastic Motion Prediction},
    author={Wei, Dong and Sun, Huaijiang and Li, Bin and Lu, Jianfeng and Li, Weiqing and Sun, Xiaoning and Hu, Shengxiang},
    journal={arXiv preprint arXiv:2210.05976},
    year={2022}
    }
```

### Acknowledgements 

Part of our code is borrowed from [DLow](https://github.com/Khrylx/DLow), [PoseFormer](https://github.com/zczcwh/PoseFormer), [LTD-GCN](https://github.com/wei-mao-2019/LearnTrajDep) and [MID](https://github.com/gutianpei/MID). We thank the authors for releasing the codes.