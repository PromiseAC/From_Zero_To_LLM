# Week 4 - MiniMind Pretraining

## Goals

- Read MiniMind source code
- Map Transformer knowledge to a real LLM implementation
- Understand Dataset / DataLoader / Causal LM Loss
- Run CUDA pretraining
- Benchmark GPU memory and throughput
- Validate Checkpoint / Resume Training
- Perform fixed-prompt checkpoint evaluation

## Structure

- `minimind/` — MiniMind source and runnable code
- `notes/` — learning notes
- `experiments/` — experiment records

## Environment

Development workflow:

`Mac` → `VS Code Remote SSH` → `WSL Ubuntu` → `RTX 3060 Laptop GPU`

Python environment:

`/home/promisea/Environments/llm`

## Git Workflow

`main` → task branch → experiment / code changes → commit → push → PR → review → merge

## Data Policy

Datasets, checkpoints, model weights, and logs live only on the remote machine and are not committed to Git.
