#### Train on the StabStitch-D dataset
Modify the train_path and test_path in train_ssd.py and run:
```
python train_ssd.py
```
#### Train on TraditionalDataset
Modify the train_path and test_path in train_tra.py and run:
```
python train_tra.py
```
#### Generate spatial warps and save them for the warp smoothing model
Modify the test_path in test_ssd.py or test_tra.py and run:
```
python test_ssd.py
```
or
```
python test_tra.py
```
#### Finetune on the driving pairs
From the repository root, starting from `pretrained/spatial_warp.pth`. Each sample is camera 5 with camera 4, or camera 5 with camera 3. Scenes and frames are shuffled. Epochs are set in `finetune_drive.py`: nuScenes 1, Lyft 1920+1224 together 3, WideDrive3 3. WideDriveTrain sits next to WideDriveVal and is listed in `train.txt`.

```
python finetune_drive.py --dataset nuscenes --gpu 0
python finetune_drive.py --dataset lyft --gpu 0
python finetune_drive.py --dataset widedrive3 --gpu 0
```

Weights are written to `SpatialWarp/model_drive/<dataset>/`.

#### NOTE
1. We only generate the spatial warps for the training set.
2. The traditional dataset is not split into training and testing sets. We merely validate the robustness of StabStitch++ on it.
