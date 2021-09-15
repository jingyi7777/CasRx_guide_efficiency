# CasRx_guide_efficiency
Supplementary code for the analyse of CasRx tiling screen data and prediction of high efficiency CasRx guides.

Please refer to ... for more information. If you find our code useful, please cite our work ...

We also provide a user-friendly, web-based tool with precomputed human and mouse transcriptome at ...

## Contents
* `data/`: CasRx tiling screen data
* `scripts/`: Scripts for preprocessing CasRx screen data, analyzing feature importance and interpreting models.  
* `models/`: All models involved in the paper, including linear models (Logistic regression), ensemble models (Random forest and Gradient-boosted tree) and deep learning models (CNN and BiLSTM).
* `prediction_results/`: Model prediction results.
* `database/`: BLAST database and Refseq gene information. 
* `model.sh`: Wrapper script for deep learning model training and testing. 

## Requirements
Installing Requirements
```
pip3 install -r requirements.txt
```