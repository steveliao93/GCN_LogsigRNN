# GCN_LogsigRNN

This repository holds the codebase for the paper:
> Logsig-RNN: a novel network for robust and efficient skeleton-based action recognition - Shujian Liao, Terry Lyons, Weixin Yang, Kevin Schlegel, and Hao Ni, BMVC 2021
### Datasets
We provide configureations for two datasets:

-Chalearn 2013 skeleton
-NTU RGB+D 120 skeleton

#### Directory Structure

Put downloaded data into the following directory structure:

```
- data/
  - chalearn/
  - nturgbd_raw/
    - nturgb+d_skeletons/     # from `nturgbd_skeletons_s001_to_s017.zip`
      ...
    - nturgb+d_skeletons120/  # from `nturgbd_skeletons_s018_to_s032.zip`
      ...
    - NTU_RGBD_samples_with_missing_skeletons.txt
    - NTU_RGBD120_samples_with_missing_skeletons.txt
```

#### Generating Data

1. NTU RGB+D 120
    - `cd data_gen`
    - `python3 ntu120_gendata.py`

## Training & Testing

- To train a new GCN-LogsigRNN model run:
```
python3 main.py
  --config <config file>
  --work-dir <place to keep things (weights, checkpoints, logs)>
  --device <GPU IDs to use>
```

- To test a trained model:
```
python3 main.py
  --config <config file>
  --work-dir <place to keep things>
  --device <GPU IDs to use>
  --weights <path to model weights>
```

- Examples
  - Train on Chalearn 2013
    - `python3 main.py --config ./config/chalearn/train_joint.yaml `
  - Train on NTU 120 XSub Joint on device 0
    - `python3 main.py --config ./config/ntu_sub/train_joint.yaml --device 0`
  - The model used is in `model/gcn_logsigRNN.py`

- Resume training from checkpoint
```
python3 main.py
  ...  # Same params as before
  --start-epoch <0 indexed epoch>
  --weights <weights in work_dir>
  --checkpoint <checkpoint in work_dir>
```

## Acknowledgements

We want to thank the authors of the following papers and repositories, their work formed the basis for this repository
  - [MS-G3D](https://github.com/kenziyuliu/MS-G3D)
  - [2s-AGCN](https://github.com/lshiwjx/2s-AGCN)
  - [ST-GCN](https://github.com/yysijie/st-gcn)

## Citation

Please cite this work if you find it useful:




  
