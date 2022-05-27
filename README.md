# MVPNet: Multi-view fusion Prediction Network for LiDAR Point Cloud Prediction based on Spatiotemporal Feature Learning

open source project for the paper: MVPNet: Multi-view fusion Prediction Network for LiDAR Point Cloud Prediction based on Spatiotemporal Feature Learning



### Export Environment Variables to dataset
We process the data in advance to speed up training. The preprocessing is automatically done if ```GENERATE_FILES``` is set to true in ```config/parameters.yaml```. The environment variable ```PATH_TO_ORIGIN_DATA``` points to the directory containing the train/val/test sequences specified in the config file. It can be set with

```bash
export PCF_DATA_RAW=/path/to/kitti-odometry/dataset/sequences
```

and the destination of the processed files ```PATH_TO_PROCESSED_DATA``` is set with

```bash
export PCF_DATA_PROCESSED=/path/to/processed/data/
```