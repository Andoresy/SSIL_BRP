# SSIL_BRP
Self-Supervised Imitation Learning for Blocks Relocation Problem

## Scripts

The project contains three main scripts:

1. `train_IL.py`: Trains the model using Imitation Learning.
2. `train_SSIL.py`: Trains the model using Self-Supervised Imitation Learning.
3. `test.py`: Tests the trained model using pre_trained model.

For a full list of available parameters and their descriptions, please refer to the `params.py`

## Usage

### Training with Imitation Learning (IL)

Use `train_IL.py` to train the model with Imitation Learning:

```bash
python rBRP_Imitation/train_IL.py [options]
python uBRP_Imitation/train_IL.py [options]
```

#### Example

```bash
python rBRP_Imitation/train_IL.py --seed 2024 --embed_dim 256 --n_heads 8 --n_encode_layers 3 --ff_hidden_dim 512 --batch 256 --lr 0.001 --epoch 50
```

### Training with Self-Supervised Imitation Learning (SSIL)

Use `train_SSIL.py` to further train with the Self-Supervised Imitation Learning:

```bash
python rBRP_Imitation/train_SSIL.py [options]
python uBRP_Imitation/train_SSIL.py [options]
```
#### Example

```bash
python uBRP_Imitation/train_SSIL.py --model_path "uBRP_IL.pt" --max_stacks 10 --max_tiers 12 --batch 256 --lr 0.0001 --epoch 200 --problem_num 1024 --sampling_num 512 --commit_alpha 0.05 --rollback_alpha 0.1 
```
### Testing the Model

Use `test.py` to test the trained model:

```bash
python rBRP_Imitation/test.py [options]
python uBRP_Imitation/test.py [options]
```
#### Examples
1. Greedy Decoding
```bash
python uBRP_Imitation/test.py --model_path "uBRP_IL_SSIL.pt" --decode_type greedy --max_stacks 5 --max_tiers 7
```
2. Sampling (ESS) Decoding
```bash
python uBRP_Imitation/test.py --model_path "uBRP_IL_SSIL.pt" --decode_type ESS --temp 2 --batch 2560 --sampling_num 2560 --max_stacks 5 --max_tiers 7
```
3. Testing all configurations:
```bash
python uBRP_Imitation/test.py --test_all --model_path "uBRP_IL_SSIL.pt" --decode_type ESS --batch 2560 --sampling_num 2560
```
