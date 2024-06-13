# Action Scene Graphs for Long-Form Understanding of Egocentric Videos

![image](https://github.com/fpv-iplab/EASG/assets/17033647/e38d266b-ba22-4fff-98b7-29046ec2611d)


This repository hosts the code related to the paper

Ivan Rodin*, Antonino Furnari*, Kyle Min*, Subarna Tripathi, and Giovanni Maria Farinella. "Action Scene Graphs for Long-Form Understanding of Egocentric Videos." Computer Vision and Pattern Recognition Conference (CVPR). 2024.

[arXiv pre-print](https://arxiv.org/pdf/2312.03391.pdf)


## Overview
This repository provides the following components:
 * The Ego-EASG dataset, containing graph annotations with object groundings and PRE, PNR, POST frames: [Download](https://iplab.dmi.unict.it/sharing/EASG/EASG.zip)
 * Code to prepare the dataset for the annotation procedure
 * Code for the EASG Annotation Tool
 * Code for the EASG Generation baseline

#### + Example video with dynamic graphs and object bounding boxes
[![video](http://markdown-videos-api.jorgenkh.no/youtube/Qx3UHbl08K4?width=640&height=360)](https://youtu.be/Qx3UHbl08K4)

## EASG Labeling system

Steps to reproduce the EASG labeling system:

#### 1. Input preparation
1.1. First, download the original annotations and the clips from [Ego4D dataset](https://ego4d-data.org/docs/start-here/).

1.2 Run the `input-preparation/create-json-annots.ipynb` notebook to obtain the set of folders initializing the graphs.

1.3 Run the `input-preparation/finalize-preparation.ipynb` to extract the relevant PRE, PNR, POST frames from video clips, to create necessary mapping files, manifest, and job list for AWS SageMaker

#### 2. Setting-up AWS SageMaker

To create the custom labeling workflow in AWS SageMaker, please refer to [official documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/sms-custom-templates.html).

2.1 To set up the annotation procedure (first stage annotation), it is necessary to provide:
* Pre-condition Lambda function, which creates the Human Intelligence Task (HIT)
`easg-labeling-system/easg-annotation/PRE-EASG`
* Consolidation Lambda function, which consolidates annotations from workers
`easg-labeling-system/easg-annotation/ACS-EASG`
* Custom web-page template and interface logic
`easg-labeling-system/easg-annotation/index.liquid.html`
* Ensure that the `easg-labeling-system/web/static` files are present in AWS S3, as they are needed for the interface

2.2. The Lambda functions and template for the second stage are provided in `easg-labeling-system/easg-validation/`


## Graph Generation baseline

To run the graph generation baseline, run the script:

`python run_easg.py annts_in_new_format data your_output_folder`

The `data` folder should contain
* [Verb features](https://iplab.dmi.unict.it/sharing/EASG/verb_features.pt)
* [RoI object features for train set](https://iplab.dmi.unict.it/sharing/EASG/roi_feats_train.pkl)
* [RoI object features for validation set](https://iplab.dmi.unict.it/sharing/EASG/roi_feats_val.pkl)

This script also relies on the annotations in the format adjusted for SGG purposes stored in `easg-generation/annts_in_new_format/`.

The code for extracting object and verb features is provided in `easg-generation/utils/`. Verb features are based on [Ego4D SlowFast features](https://ego4d-data.org/docs/data/features/). Object features are based on detections of [Faster-RCNN model](https://github.com/facebookresearch/detectron2/blob/main/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml), trained on our Ego4D-EASG dataset

### Note
This is the version 1.0 of our dataset, and it contains the graphs obtained from Ego4D-SCOD benchmark for videos collected by University of Catania (UNICT).
Currently, we are working on extending the dataset to include annotations for videos recorded by other institutions.

If you use the code/models hosted in this repository, please cite the following paper:

```bibtex
@article{rodin2023action,
  title={Action Scene Graphs for Long-Form Understanding of Egocentric Videos},
  author={Rodin, Ivan and Furnari, Antonino and Min, Kyle and Tripathi, Subarna and Farinella, Giovanni Maria},
  journal={arXiv preprint arXiv:2312.03391},
  year={2023}
}
```

Please, refer to the paper for more technical details.
