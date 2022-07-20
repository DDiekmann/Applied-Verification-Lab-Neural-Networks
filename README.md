# Tutorials for Neural Network Verification Tools

*This is a work in progress repository. Currently worked on by a course at the Carl von Ossietzky University Oldenburg.* 

## Why do we need to verify neural networks? 

Nowadays, neural networks are used in safety-critical areas, among others. As a result, there is a need to formally prove some of the network's behaviours (especially under malicious inputs).
One of those behaviours is the so-called **Robustness** of the network. Robustness means that small perturbations to the inputs should not lead to changes in the output of the neural network.

<p align="center">
  <img src="https://openai.com/content/images/2017/02/adversarial_img_1.png" />
</p>

The illustration shows, for example, image recognition for an animal. On the left side, the neural network recognises a panda with a probability of 57.7%. By adding noise, a gibbon is recognised in the following with a probability of 99.3%. In this example, this wrong decision is probably not safety-critical, but other use cases (for example in the field of autonomous driving) are conceivable where such a wrong decision could have serious consequences.

## Tutorials

### SMT Based Verification

If you are **new to SMT based verification** of neural network we recommend you starting with the Z3 Tutorial because it is the most basic one. If you are additionally interested in the theory behind this we recommend this [book on neural network verification](https://arxiv.org/pdf/2109.10317.pdf). 

| Tool             | Colab Link | Tool Codebase                                        | Progress                |
|------------------|------------|------------------------------------------------------|-------------------------|
| Z3               | <a href="https://colab.research.google.com/github/DDiekmann/Applied-Verification-Lab-Neural-Networks/blob/main/Tutorials/Tutorial_for_SMT_based_Verification_with_Z3.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | https://github.com/Z3Prover/z3                       | :heavy_check_mark: DONE |
| Marabou          | <a href="https://colab.research.google.com/github/DDiekmann/Applied-Verification-Lab-Neural-Networks/blob/main/Tutorials/Tutorial_for_Neural_Network_Verification_with_Marabou.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | https://github.com/NeuralNetworkVerification/Marabou | :heavy_check_mark: DONE |
| Planet           | <a href="https://colab.research.google.com/github/DDiekmann/Applied-Verification-Lab-Neural-Networks/blob/main/Tutorials/Planet.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | https://github.com/progirep/planet                   | :heavy_check_mark: DONE        |

### Abstraction Based Verification

| Tool             | Colab Link | Tool Codebase                                        | Progress                |
|------------------|------------|------------------------------------------------------|-------------------------|
| ERAN               | <a href="https://colab.research.google.com/github/DDiekmann/Applied-Verification-Lab-Neural-Networks/blob/main/Tutorials/Verification_by_abstraction_with_eran.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | https://github.com/eth-sri/eran                       | :heavy_check_mark: DONE |

### Others
| Tool             | Colab Link | Tool Codebase                                        | Progress                |
|------------------|------------|------------------------------------------------------|-------------------------|
| α,β-CROWN | <a href="https://colab.research.google.com/github/DDiekmann/Applied-Verification-Lab-Neural-Networks/blob/main/Tutorials/Alpha_Beta_Crown.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a> | https://github.com/huanzhang12/alpha-beta-CROWN      | :heavy_check_mark: DONE |

