## Getting Started (Graph Generation)
### Task-specific FCs
To run the graph generation baseline using task-specific FCs, run the script:

`python run_easg.py annts_in_new_format data your_output_folder`
                                                                                                                                                                                                        The `data` folder should contain
* [Verb features](https://iplab.dmi.unict.it/sharing/EASG/verb_features.pt)
* [RoI object features for train set](https://iplab.dmi.unict.it/sharing/EASG/roi_feats_train.pkl)
* [RoI object features for validation set](https://iplab.dmi.unict.it/sharing/EASG/roi_feats_val.pkl)

This script also relies on the annotations in the format adjusted for SGG purposes stored in `easg-generation/annts_in_new_format/`.

The code for extracting object and verb features is provided in `easg-generation/utils/`. Verb features are based on [Ego4D SlowFast features](https://ego4d-data.org/docs/data/features/). Object features are based on detections of [Faster-RCNN model](https://iplab.dmi.unict.it/sharing/EASG/model_final.pth), trained on our Ego4D-EASG dataset

### STTran
This baseline builds on the original [STTran repository](https://github.com/yrcong/STTran). Please follow the instructions for requirements or other preparation.

First, download the [Faster-RCNN checkpoint that is pre-trained on EASG](https://iplab.dmi.unict.it/sharing/EASG/faster_rcnn_1_11_952.pth) and put it under `fasterRCNN/models/res101/easg`. 
Then, run the graph generation baseline with STTran as follows:

`python train_with_EASG.py -mode $MODE -data_path $DATAPATH`

MODE: edgecls, sgcls, or easgcls
