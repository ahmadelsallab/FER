# Fusion approaches for spatiotemporal Face Expression Recognition (FER)
This work is based on S. Zhang et al.: Learning Affective Video Features for FER via Hybrid Deep Learning, we will follow their architecture, with the some modifications:



# Features
1- Replace DBN with normal Dense (Fully Connected) layers
2- Use normal softmax with categorical cross entropy loss as the classification layer
3- Smaller model

__TBD__
- Experiment different fusion layers (flatten, GlobalAvgPooling (GAP)) and operations (concat, sum, avg)
- Pre-train on other datasets/tasks
- Add recurrence (ConvLSTM)
- Add frame stacking

# Requirements
TBD

# How ro run
## Data preparation
- Download and unzip BAUM 1 dataset in a home_dir
- In FER_data_preparation.ipynb set baum_dir to your BAUM directory
- Run FER_data_preparation.ipynb. This will generate csv file (data.csv) and two folders: imgs_spatial and imgs_flows as described in the data preparation section


 
## Training
- Set baum_dir in FER_fusion.ipynb to the data directory in the previous step
- Run FER_fusion.ipynb. This will save a model models/Model_fusion.h5

## Inference
- Set baum_dir in FER_BAUM Baseline.ipynb 
- Set model_path = os.path.join(baum_dir, 'models', 'Model_fusion.h5')
- Run FER_BAUM Baseline.ipynb 



