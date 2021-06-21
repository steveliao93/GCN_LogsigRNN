# gcn_logsigrnn
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

1. NTU RGB+D
    - `cd data_gen`
    - `python3 ntu120_gendata.py`
    
## Training & Testing

For varied length case:
- The general training template command
- `python3 main_var.py --config ./config/ntu_sub/train_joint_logsig_gcn_var.yaml --device 0`
