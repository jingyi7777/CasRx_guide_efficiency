# CasRx_guide_efficiency
Supplementary code for the analyse of CasRx tiling screen data and prediction of high efficiency CasRx guides.

Please refer to <a href="https://www.biorxiv.org/content/10.1101/2021.09.14.460134v1" target="_blank">"Deep learning of Cas13 guide activity from high-throughput gene essentiality screening"</a> for more information.\
If you find our code useful, please cite our work!

We also provide a user-friendly, web-based tool with precomputed model organism transcriptome and custom sequences at 
<a href="https://www.rnatargeting.org" target="_blank">rnatargeting.org</a>.

## Contents
* `data/`: CasRx tiling screen data
* `scripts/`: Scripts for preprocessing CasRx screen data, analyzing feature importance and interpreting models.  
* `models/`: All models involved in the paper, including linear models (Logistic regression), ensemble models (Random forest and Gradient-boosted tree) and deep learning models (CNN and BiLSTM).
* `prediction_results/`: Model prediction results.
* `database/`: BLAST database and Refseq gene information. 
* `model.sh`: Wrapper script for deep learning model training and testing. 
