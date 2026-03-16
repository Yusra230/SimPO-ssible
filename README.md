# SimPO-ssible: SimPO Implementation

## Team Members
- Member 1: Yusra - AI-22010
- Member 2: Muhammad Umer - AI-22035

## Paper
**SimPO: Simple Preference Optimization with a Reference-Free Reward**  
- Authors: Yu Meng, Mengzhou Xia, Danqi Chen  
- Venue: NeurIPS 2024  
- Paper Link: https://arxiv.org/abs/2405.14734  
- Official Repo: https://github.com/princeton-nlp/SimPO

## What is SimPO?
SimPO is an improved version of DPO (Direct Preference Optimization) 
that trains language models to prefer good responses over bad ones.

Key improvements over DPO:
- No reference model needed (saves memory)
- Uses average log probability as reward (fixes training/generation mismatch)
- Adds a target margin gamma to improve learning confidence

## Repo Structure
```
SimPO-ssible/
│
├── notebook/
│   └── SimPO_Implementation.ipynb  
│
├── results/
│   └── training_curve.png          
│
├── README.md                       
└── requirements.txt                
```

## How to Run
1. Open `notebook/SimPO_Implementation.ipynb` in Google Colab
2. Go to Runtime → Change runtime type → Select **T4 GPU**
3. Run all cells in order from top to bottom
4. Training takes approximately 20-30 minutes on free Colab GPU
5. Training curve will be saved automatically in the results/ folder

## Requirements
See `requirements.txt` for full list. Main libraries:
- torch
- transformers
- trl
- datasets
- matplotlib

## Results
| Metric | Our Result (GPT-2) | Paper Result (Llama-3) |
|--------|-------------------|----------------------|
| Final Training Loss |  | ~0.65 |
| AlpacaEval 2 Score | N/A (GPT-2 too small) | 64.8 |

Note: We used GPT-2 due to hardware limitations (free Google Colab).
The paper used Llama-3-8B. Our goal was to verify the training 
process and loss trends, not match the exact benchmark numbers.

## Key Hyperparameters
| Parameter | Value | Meaning |
|-----------|-------|---------|
| gamma (γ) | 0.5 | Target margin between good and bad responses |
| beta (β) | 0.1 | Controls deviation from base model |
| Learning rate | 1e-5 | How fast the model updates |
| Batch size | 2 | Number of examples per update step |
| Epochs | 1 | Full passes through the dataset |

## Assignment Info
- Course: CT-469 Reinforcement Learning
- Semester: Spring 2026
- Assignment: Research Paper Analysis & Implementation

---
