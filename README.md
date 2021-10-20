# GCN_LogsigRNN

This repository holds the codebase for the paper:
**Logsig-RNN: a novel network for robust and efficient skeleton-based action recognition** Shujian Liao, Terry Lyons, Weixin Yang, Kevin Schlegel, and Hao Ni, BMVC 2021
### Data Preprocessing

#### Directory Structure

Put downloaded data into the following directory structure:

```
- data/
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

To train a new GCN-LogsigRNN model run:
```
python3 main.py --config ./config/ntu_sub/train_joint.yaml --device 0
```
- The model used is in `model/gcn_logsigRNN.py`
