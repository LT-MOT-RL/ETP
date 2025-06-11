# ETP
Long Term Video Object Tracking

#  OTETrack 

Thanks to the contributors of [OTETrack](https://github.com/OrigamiSL/OTETrack).  
Please place the file `OTETrack_all.pth.tar` into the `./OTETrack/test_checkpoint/` directory.
For more details, please refer to [OTETrack](https://github.com/OrigamiSL/OTETrack).


Weight sources (OTETrack_256_full):

You can download the model weights from [Google Drive](https://drive.google.com/file/d/1-9CceF4HwsudLi9pt5ylDEhYtrgGDhsz/view?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1lJz4RlgCE8XW7lV3sXbcBw?pwd=25ur) (extracted code: 25ur).

Put the model weights you download in `./test_checkpoint.` The file tree shall look like this:
```
   ${PROJECT_ROOT}
    |-- test_checkpoint
    |   |-- OTETrack_all.pth.tar
```




#  Unicorn

Thanks to the contributors of [Unicorn](https://github.com/MasterBin-IIAU/Unicorn). 
Deploying Unicorn can be difficult. Please see [Unicorn](https://github.com/MasterBin-IIAU/Unicorn) for guidance.


#  The File tree of ETP
The file tree of ETP shall look like this:
```
    -- ${DATASETS_PATH}
        |-- LaSOT/airplane
        ...
    -- ${OTETRACK_PATH}
        |-- experiments/otetrack_256_full.yaml
        |-- lib
        |-- test_checkpoint/test_checkpoint/OTETrack_all.pth.tar
        ...
    -- ${UNICORN_PATH}
        |-- Unicorn_outputs/unicorn_track_tiny_sot_only/latest_ckpt.pth
        |-- external_2
        ...
    -- ${SOT_STGCNN_PATH}
        |-- checkpoint/sot-stgcnn/sot-stgcnn-lasot_1_300_nr_6_4/val_best.pkl
    -- ${TOOLS_PATH}
        |-- test_lasot.py
```

#  Install the environment


CUDA 11.3
Partial paramount site-packages requirements are listed below:
- `python == 3.9.7` 
- `pytorch == 1.11.0`
- `torchvision == 0.12.0`
- `matplotlib == 3.5.1`
- `numpy == 1.21.2`
- `pandas == 1.4.1`
- `pyyaml == 6.0`
- `scipy == 1.7.3`
- `scikit-learn == 1.0.2`
- `tqdm == 4.63.0`
- `yaml == 0.2.5`
- `opencv-python == 4.5.5.64`


#  Test
```
python tools/test_lasot.py
```

# Others 

Coming soon~~