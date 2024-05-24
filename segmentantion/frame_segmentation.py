import os
import pickle
import warnings
import sys
import cv2
sys.path.append('../')
import numpy as np
import subprocess
warnings.simplefilter(action='ignore', category=FutureWarning)


#Path of the root directory of the video k-Net project relative to the parent directory of the current project
knet_root_dir_path="./K-Net"
#Path of the virtual environment directory of the video K-Net project relative to the root directory of the K-Net project
knet_venv_dir_path="./.venv"


#Empty the folder containing the images that were passed into video K-Net to segment
def clearImages():

    files = os.listdir("../../"+knet_root_dir_path.strip("/")+"/data/VIPSeg/images/fragment")
    for file in files:
        file_path = os.path.join("../../"+knet_root_dir_path.strip("/")+"/data/VIPSeg/images/fragment", file)
        os.remove(file_path)

    files = os.listdir("../../"+knet_root_dir_path.strip("/")+"/data/VIPSeg/panomasks/fragment")
    for file in files:
        file_path = os.path.join("../../"+knet_root_dir_path+"/data/VIPSeg/panomasks/fragment", file)
        os.remove(file_path)

def getSegmentation(images):

    clearImages()

    #For each image
    for i in range(len(images)):
        #Resize it to the required dimensions needed by video K-Net
        temp=cv2.resize(images[i],(1280,720))
        #Set the color space and the indexing
        temp=cv2.cvtColor(temp, cv2.COLOR_RGB2BGR)
        numbering="0"*(8-len(str(i+1)))+str(i+1)
        cv2.imwrite("../../"+knet_root_dir_path.strip("/")+"/data/VIPSeg/images/fragment/"+numbering+".jpg", temp)
        cv2.imwrite("../../"+knet_root_dir_path.strip("/")+"/data/VIPSeg/panomasks/fragment/"+numbering+".png", temp)

    #Set the required paths to call the subprocess to run video K-Net
    python_executable = "./"+knet_venv_dir_path.strip("/")+"/bin/python"
    command_and_args = [
        python_executable,
        "./tools/test_step.py",
        "./configs/det/video_knet_vipseg/video_knet_s3_swin_b_rpn_vipseg_mask_embed_link_ffn_joint_train_8e.py",
        "./video_k_net_swinb_vip_seg.pth",
        "--show-dir",
        "./out",
    ]

    root_dir = '/'.join(os.path.abspath(__file__).split('/')[:-3]) + "/" + knet_root_dir_path.strip("/")
    working_dir = root_dir
    #Set PYTHONPATH to include the Python package path
    env = os.environ.copy()
    env["PYTHONPATH"] = root_dir

    #Call the video K-Net subprocess
    p = subprocess.Popen(command_and_args, stdout=subprocess.PIPE, stderr=subprocess.PIPE, cwd=working_dir, env=env)
    p.wait()
    stdout, stderr = p.communicate()

    #Open and read the file returned from video K-Net containing the segmentation masks
    file = open("../../"+knet_root_dir_path.strip("/")+"/out/segments", 'rb')
    segments = pickle.load(file)
    file.close()

    clearImages()

    #Return the segmentation masks
    return segments

#Segment the frames of the fragment into visual objects using video K-Net
def segmentFrames(frames, keyframe_index):

    #Get the video panoptic segmenation for the sampled frames of the fragment
    segments = getSegmentation(frames)

    target_shape = (frames[0].shape[1],frames[0].shape[0])
    #Resize the segmentation masks to the desired shape
    for i in range(len(segments)):
        segments[i] = cv2.resize(segments[i], target_shape, interpolation=cv2.INTER_NEAREST).astype(int)

    #Get the keyframe segments and superimpose them to the matching segments of the other frames
    #(Ignore segments that don't belong to the keyframe)
    segments_keyframe = segments[keyframe_index]
    segments_new = np.zeros_like(segments)
    for i, seg in enumerate(np.unique(segments_keyframe)):
        segments_new[np.where(np.array(segments) == seg)] = i + 1
    segments = list(segments_new)

    #Return the emerging segmentation masks
    return segments