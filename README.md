# An Integrated Framework for Multi-Granular Explanation of Video Summarization

## PyTorch implementation [[Paper](https://arxiv.org/abs/2405.10082)] [[Cite](#citation)]
- From **"An Integrated Framework for Multi-Granular Explanation of Video Summarization"**
- Written by Konstantinos Tsigos, Evlampios Apostolidis and Vasileios Mezaris.
- This software can be used to produce explanations for the outcome of a video summarization model. Our framework integrates methods for generating explanations at the fragment level (indicating which video fragments influenced the most the decisions of the summarizer), and the more fine-grained object level (highlighting which visual objects were the most influential for the summarizer on a specific video fragment). For fragment-level explanations, we employ the model-specific attention-based approach proposed in [Apostolidis et al. (2022)](https://ieeexplore.ieee.org/document/10019643), and introduce a new model-agnostic method that does not require any knowledge about the summarization model. The fragments of the aformentioned explanations, alongside the fragments selected by the summarizer to be included in the summary, are then processed by a state-of-the-art video panoptic segmentation framework and combined with an adaptation of a perturbation-based approach, to generate object-level explanations.

## Main dependencies
The code was developed, checked and verified on an `Ubuntu 20.04.6` PC with an `NVIDIA RTX 4090` GPU and an `i5-12600K` CPU. All dependencies can be found inside the [requirements.txt](requirements.txt) file, which can be used to set up the necessary virtual enviroment.

To run the Video K-Net method for video panoptic segmentation, use the code from the [official Github repository](https://github.com/lxtGH/Video-K-Net) and set-up the necesary environment following the instructions in the aforementioned repository, and the steps bellow: 
- The utilized trained model, called `video_k_net_swinb_vip_seg.pth` and found [here](https://github.com/lxtGH/Video-K-Net?tab=readme-ov-file#pretrained-ckpts-and-trained-models), should be placed within the root directory of the video K-Net project.
- The `test_step.py` script located [here](https://github.com/lxtGH/Video-K-Net/tree/main/tools), needs to be replaced by the provided [test_step.py](/k-Net/test_step.py) script.
- The `data` folder within the root directory of the video k-Net project, should be created manually and have the following structure:
```Text
/data
    /VIPSeg
        /images
            /fragment
        /panomasks
            /fragment
        val.txt
```
- The `val.txt` file found [here](), should be placed within the /VIPSeg directory, as shown above.

Regarding the temporal segmentation of the videos, the utilized fragments in our experiments are available in the [data](https://github.com/IDT-ITI/XAI-Video-Summaries/tree/main/data) folder. As stated in our paper, these fragments were produced by the TransNetV2 shot segmentation method (for multi-shot videos) and the motion-driven method for sub-shot segmentation (for single-shot videos), described in [Apostolidis et al. (2018)](https://link.springer.com/chapter/10.1007/978-3-319-73603-7_3). In case there is a need to re-run shot segmentation, please use the code from the [official Github repository](https://github.com/soCzech/TransNetV2) and set-up the necesary environment following the instructions in the aforementioned repository. In case there is a need to also re-run sub-shot segmentation, please contact us for providing access to the utilized method.

The paths of the Video K-Net and TransNetV2 projects, along with their corresponding virtual environments can be set in the [video_segmentation.py](segmentation/video_segmentation.py#L7:L10) and [frame_segmentation.py](segmentation/frame_segmentation.py#L12:L15) files, accordingly. Please note that the paths for the projects are given relatively to the parent directory of this project, while the paths of the virtual environments are given relatively to the root directory of the corresponding project.

If there is a need to use the default paths:
- Set the name of the root directory of the projects to *TransNetV2* and *K-Net* and place them in the parent directory of this project.
- Set the name of the virtual environment of each project to *.venv* and place it inside the root directory of the corresponding project.
This will result in the following project structure:
```Text
/Parent Directory
    /K-Net
        /.venv
            ...
        ...
    /TransNetV2
        /.venv
            ...
        ...
    /XAI-Video-Summaries
        ...
```

## Data
<div align="justify">

Original videos for each dataset are available in the dataset providers' webpages: 
- <a href="https://github.com/yalesong/tvsum" target="_blank"><img align="center" src="https://img.shields.io/badge/Dataset-TVSum-green"/></a> <a href="https://gyglim.github.io/me/vsum/index.html#benchmark" target="_blank"><img align="center" src="https://img.shields.io/badge/Dataset-SumMe-blue"/></a>

These videos have to be placed into the `SumMe` and `TVSum' directories of the [data](data) folder.

The extracted deep features for the SumMe and TVSum videos are already available into aforementioned directories. In case there is a need to extract these deep features from scratch (and store them into h5 files), please run the [feature_extraction.py](features/feature_extraction.py) script. Otherwise, an h5 file will be produced automatically for each video and stored into the relevant directory of the [data](data) folder.

The produced h5 files have the following structure:
```Text
/key
    /features                 2D-array with shape (n_steps, feature-dimension)
    /n_frames                 number of frames in original video
```
</div>

The utilized pre-trained models of the [CA-SUM](https://github.com/e-apostolidis/CA-SUM) method, are available within the [models](/explanation/models) directory. Their performance, as well as some other training details, are reported below.
Model| F1 score | Epoch | Split | Reg. Factor
| --- | --- | --- | --- | --- |
summe.pkl | 59.138 | 383 | 4 | 0.5
tvsum.pkl | 63.462 | 44 | 4 | 0.5

## Producing explanations
<div align="justify">

To produce explanations for a video of the SumMe and TVSum datasets, please execute the following command:
```
python explanation/explain.py --model MODEL_PATH --video VIDEO_PATH --fragments NUM_OF_FRAGMENTS (optional, default=3)
```
where, `MODEL_PATH` refers to the path of the trained summarization model, `VIDEO_PATH` refers to the path of the video, and `NUM_OF_FRAGMENTS` refers to the number of utilized video fragments for generating the explanations.

After executing the above command, a new folder is created (if it does not already exist), with the same name and in the same directory as the video we want to explain. There, the deep features (if the file containing the deep features for all the videos of the dataset, does not exist in the dataset folder) and the video shots (unless we already have the optical flow subshots) are extracted in h5 and txt format files, accordingly. Another folder is also created called explanation, containing two txt files that hold the fragment-level explanation (the top `NUM_OF_FRAGMENTS` from Attention and Positive LIME are the explanation) and the evaluation scores, one csv file with the indexes of the explanation fragments and three folders that have the object level explanations, using the top fragments obtained from the afformentioned fragment-level explanations, as well as the ones selected to be included in the summary by the summarizer. Each folder contains 4 explanation images for each fragment, highlighting the most positive, most negative and both positive and negative segments of the keyframe, as well as providing the mask of the latter. Finally, the evaluation scores of each object level explanation can be found inside a csv file, with each row corresponding to the metrics of the top fragments in descending order (first row: top 1 fragment, second row: top 2 fragment etc).

Alternatively, to run the experiment on all of the videos of both datasets, execute the [explain.sh](/explanation/explain.sh) bash script.

## Evaluation results
<div align="justify">

To get the evaluation results, averaged on all of the videos of each dataset, run the [final_scores.py](explanation/final_scores.py) script. The final scores are saved into the `final_scores` folder, created inside the [explanation](/explanation) path. To specify a specific dataset or videos for each dataset, set the [dataset](explanation/final_scores.py#L9:L10) and [videos](explanation/final_scores.py#L11:L12) variables appropriately.

## Citation
<div align="justify">
    
If you find our work, code or pretrained models, useful in your work, please cite the following publication:

K. Tsigos, E. Apostolidis, V. Mezaris, "<b>An Integrated Framework for Multi-Granular Explanation of Video Summarization</b>", arXiv preprint arXiv:2405.10082, 2024.
</div>

BibTeX:

```
@misc{tsigos2024integrated,
      title={An Integrated Framework for Multi-Granular Explanation of Video Summarization}, 
      author={Konstantinos Tsigos and Evlampios Apostolidis and Vasileios Mezaris},
      year={2024},
      eprint={2405.10082},
      archivePrefix={arXiv},
      primaryClass={cs.CV}
}
```

## License
<div align="justify">
    
Copyright (c) 2024, Konstantinos Tsigos, Evlampios Apostolidis, Vasileios Mezaris / CERTH-ITI. All rights reserved. This code is provided for academic, non-commercial use only. Redistribution and use in source and binary forms, with or without modification, are permitted for academic non-commercial use provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice, this list of conditions and the following disclaimer in the documentation provided with the distribution.

This software is provided by the authors "as is" and any express or implied warranties, including, but not limited to, the implied warranties of merchantability and fitness for a particular purpose are disclaimed. In no event shall the authors be liable for any direct, indirect, incidental, special, exemplary, or consequential damages (including, but not limited to, procurement of substitute goods or services; loss of use, data, or profits; or business interruption) however caused and on any theory of liability, whether in contract, strict liability, or tort (including negligence or otherwise) arising in any way out of the use of this software, even if advised of the possibility of such damage.
</div>

## Acknowledgement
<div align="justify"> This work was supported by the EU Horizon 2020 programme under grant agreement 951911 AI4Media. </div>
