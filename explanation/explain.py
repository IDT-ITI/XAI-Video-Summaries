import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import numpy as np
import warnings
warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)
import sys
sys.path.append("../")
import argparse
import pandas as pd
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from lime_explanation import *
from explanation.utils import load_model,load_data,predict
from segmentantion.video_segmentation import getShots
from segmentantion.frame_segmentation import *
from metrics import MetricsSpatialCalculator, MetricsFragmentCalculator
from attention_explanation import *

def getFragments(video_path, scores=None, num_of_fragments=3, top=False):

    #Extract the shots of the video
    shots = np.array(getShots(video_path))
    total_frames=shots[-1][-1]

    #Get the frame indexes of the sampled frames
    sampled_frames_indexes = []
    for i in range(0,total_frames,15):
        sampled_frames_indexes.append(i+1)

    #If we want to get the top scoring fragments
    if(top):
        #Match each sampled frame to the corresponding shot it belongs to compute the scores of the shots
        curr_shot = 0
        shot_scores = [[] for _ in range(len(shots))]
        for i in range(len(sampled_frames_indexes)):
            for j in range(curr_shot, len(shots)):
                if (sampled_frames_indexes[i] >= shots[j][0] and sampled_frames_indexes[i] <= shots[j][1]):
                    shot_scores[j].append(scores[i])
                    curr_shot = j
                    break

        count=0
        #For each shot
        for i in range(len(shot_scores)):
            #If at least two sampled frame belongs to the shot
            if (len(shot_scores[i]) > 1):
                #Compute the score of the shot by taking the average of the sampled frame scores
                shot_scores[i] = (sum(shot_scores[i]) / len(shot_scores[i]))
            else:
                #Otherwise mark the shot
                shot_scores[i] = -1
                count += 1

        #Get the top fragments based on the shot scores (ignore the marked shots which are at the beginning of the list due to the negative values of their scores)
        top_fragments = shots[np.argsort(shot_scores)[count:][-num_of_fragments:]][::-1]
        fragments=top_fragments

    else:
        #Otherwise the fragments are the shots of the video
        fragments=shots

    fragment_sampled_frames = []
    fragment_sampled_frames_index = []
    #For each top fragment
    for fragment in fragments:
        #Get the sampled frames belonging to the corresponding fragment (true for the frames belonging to the fragment else false)
        fragment_frames_selection = (fragment[0] <= sampled_frames_indexes) & (sampled_frames_indexes <= fragment[1])
        #Get the indexes of the sampled frames belonging to the corresponding fragment
        fragment_frames_index = np.argwhere(fragment_frames_selection == True).reshape(-1,)
        #Get the numbers of the sampled frames belonging to the corresponding fragment
        fragment_frames=np.array(sampled_frames_indexes)[fragment_frames_index]
        #Append the indexes and the numbers of the sampled frames of the fragment into lists
        fragment_sampled_frames.append(list(fragment_frames))
        fragment_sampled_frames_index.append(list(fragment_frames_index))

    fragment_sampled_frames=list(fragment_sampled_frames)
    fragment_sampled_frames_index=list(fragment_sampled_frames_index)

    #If we want to get all of the fragments
    if(not top):
        #Check if any fragment contains zero sampled frames and mark it
        delete_fragments=[]
        for i in range(len(fragments)):
            if (not len(fragment_sampled_frames[i])>0):
                delete_fragments.append(i)

        #Start by deleting the last marked fragment in order for to maintain the indexes of the following unaltered
        #(If we deleted for example the one in the first position then if we wanted to delete the forth one it would be now on the third position)
        delete_fragments=delete_fragments[::-1]
        for i in delete_fragments:
            fragments=np.delete(fragments, i, axis=0)
            del fragment_sampled_frames[i]
            del fragment_sampled_frames_index[i]

    #Return the fragment ranges, the numbers of the sampled frames of each fragment and their corresponding indexes used by the summarizer
    return fragments,fragment_sampled_frames,fragment_sampled_frames_index

def loadFrame(video_path,index):
    #Open a video capture for the specific video
    vidcap = cv2.VideoCapture(video_path)
    #Set the current frame index
    vidcap.set(cv2.CAP_PROP_POS_FRAMES, index)
    #Extract the frame at that index
    _, frame = vidcap.read()
    #Set the colorspace
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #Close the video capture
    vidcap.release()
    #Return the frame
    return frame


def explain_fragments(features, model, video_path, num_of_fragments, explanation_path, explanation_method):

    #Create the necessary path to save all of the explanations if it doesn't already exists
    if (not (os.path.exists(explanation_path))):
        os.mkdir(explanation_path)

    #Get the fragment ranges, the numbers of the sampled frames of each fragment and their corresponding indexes used by the summarizer
    fragments, fragment_frames, fragment_frames_index = getFragments(video_path)

    #Check if the fragment-level explanation and evaluation has already been computed
    if (os.path.isfile(explanation_path + "indexes.csv") and os.path.isfile(explanation_path + "fragments_explanation.txt")
        and os.path.isfile(explanation_path + "fragments_explanation_evaluation_metrics.csv")):

        #If true, then load the saved indexes of the explanation fragments for the Attention and LIME fragment-level explanations
        df = pd.read_csv(explanation_path + "indexes.csv")
        attention_explanation = df["Attention"].values
        lime_explanation_positive = df["Lime Positive"].values
        lime_explanation_positive = lime_explanation_positive[np.where(lime_explanation_positive != -1)]

    else:
        #Otherwise compute the explanations
        ##Compute the fragment-level explanation for Attention
        attention_explanation = explain_with_attention(features, model, fragment_frames_index)

        #Create a fragment-level LIME explanator
        explainer = LimeFragmentExplainer()
        #Compute the fragment-level explanation for LIME
        explanation = explainer.explain_instances(20000, features, fragment_frames_index, model, len(fragments))
        #Top positive scoring fragments
        lime_explanation_positive = np.array([x[0] for x in explanation if x[1] > 0])
        #Top negative scoring fragments
        lime_explanation_negative = np.array([x[0] for x in explanation if x[1] < 0])
        #(unlike Attention where the first and the last fragments of the results are the top- and bottom-scoring respectively,
        #LIME returns seperate results for fragments that contribute positively or negatively to the explanation)

        #Create a dataframe to save the indexes of the explanation fragments for each XAI method
        df = pd.DataFrame({'Attention': attention_explanation, 'Lime Positive': -1, 'Lime Negative': -1})
        df.loc[:len(lime_explanation_positive) - 1, 'Lime Positive'] = lime_explanation_positive
        df.loc[:len(lime_explanation_negative) - 1, 'Lime Negative'] = lime_explanation_negative
        #Save the dataframe to a csv file
        df.to_csv(explanation_path + "indexes.csv", index=False)

        #Save the ranges of the top- and bottom-scoring explanation fragments to txt file for each XAI method
        with open(explanation_path + "fragments_explanation.txt", "w") as output:
            output.write("Attention:\n")
            output.write("\n".join([','.join(str(y) for y in p) for p in fragments[attention_explanation]]))
            output.write("\n\nLIME:\n")
            output.write("Positive:\n")
            if(lime_explanation_positive.shape[0]>0):
                output.write("\n".join([','.join(str(y) for y in p) for p in fragments[lime_explanation_positive]]))
                output.write("\n")
            output.write("Negative:\n")
            if (lime_explanation_negative.shape[0] > 0):
                output.write("\n".join([','.join(str(y) for y in p) for p in fragments[lime_explanation_negative]]))
                output.write("\n")

        #Create a metric calculator object
        metrics_calculator = MetricsFragmentCalculator(model, features, fragment_frames_index)
        #Compute the disc plus, disc minus and sanity violation scores for the Attention explanation
        discoverability_plus_attention = metrics_calculator.compute_discoverability(attention_explanation, 3)
        #Conduct the evaluation on the top three and bottom three scoring fragments returned by Attention
        discoverability_minus_attention = metrics_calculator.compute_discoverability(attention_explanation[::-1], 3)
        sanity_violation_attention = metrics_calculator.compute_sanity_violation(discoverability_plus_attention, discoverability_minus_attention)

        #Do the same for the LIME explanation
        metrics_calculator = MetricsFragmentCalculator(model, features, fragment_frames_index)
        discoverability_plus_lime = metrics_calculator.compute_discoverability(lime_explanation_positive, 3)
        #Conduct the evaluation on the top three positive and top three negative scoring fragments returned by LIME
        discoverability_minus_lime = metrics_calculator.compute_discoverability(lime_explanation_negative, 3)
        sanity_violation_lime = metrics_calculator.compute_sanity_violation(discoverability_plus_lime, discoverability_minus_lime)

        #Create a dataframe to save the scores
        df = pd.DataFrame()

        #Fill the dataframe with the scores
        df['Attention Disc Plus One By One'] = discoverability_plus_attention[0] + [np.nan] * (3 - len(discoverability_plus_attention[0]))
        df['Attention Disc Plus Sequentially'] = discoverability_plus_attention[1] + [np.nan] * (3 - len(discoverability_plus_attention[1]))
        df['Attention Disc Minus One By One'] = discoverability_minus_attention[0]  + [np.nan] * (3 - len(discoverability_minus_attention[0]))
        df['Attention Disc Minus Sequentially'] = discoverability_minus_attention[1] + [np.nan] * (3 - len(discoverability_minus_attention[1]))
        df['Attention Sanity Violation One By One'] = [sanity_violation_attention[0]] + [np.nan] * 2
        df['Attention Sanity Violation Sequentially'] = [sanity_violation_attention[1]] + [np.nan] * 2
        df['Lime Disc Plus One By One'] = discoverability_plus_lime[0] + [np.nan] * (3 - len(discoverability_plus_lime[0]))
        df['Lime Disc Plus Sequentially'] = discoverability_plus_lime[1] + [np.nan] * (3 - len(discoverability_plus_lime[1]))
        df['Lime Disc Minus One By One'] = discoverability_minus_lime[0] + [np.nan] * (3 - len(discoverability_minus_lime[0]))
        df['Lime Disc Minus Sequentially'] = discoverability_minus_lime[1] + [np.nan] * (3 - len(discoverability_minus_lime[1]))
        df['Lime Sanity Violation One By One'] = [sanity_violation_lime[0]] + [np.nan] * 2
        df['Lime Sanity Violation Sequentially'] = [sanity_violation_lime[1]] + [np.nan] * 2

        df[''] = ['Top 1', 'Top 2', 'Top 3']
        df.set_index('', inplace=True)
        #Save the dataframe to a csv file
        df.to_csv(explanation_path + "fragments_explanation_evaluation_metrics.csv")

    #For each of the different fragment explanation method
    if(explanation_method=="Attention"):
        #Get the explanation fragments ranges from the indexes
        explanation_fragments=fragments[attention_explanation[:num_of_fragments]]
        #Do the same for the numbers of the sampled frames of each explanation fragment
        explanation_fragment_frames=list(np.array(fragment_frames)[attention_explanation[:num_of_fragments]])
        #and their corresponding indexes used by the summarizer
        explanation_fragment_frames_index=list(np.array(fragment_frames_index)[attention_explanation[:num_of_fragments]])
    elif(explanation_method=="LIME"):
        #Do the same for the LIME explanation method, by first checking if there are any fragments to be returned as explanation
        if(lime_explanation_positive.shape[0]>0):
            explanation_fragments=fragments[lime_explanation_positive[:num_of_fragments]]
            explanation_fragment_frames=list(np.array(fragment_frames)[lime_explanation_positive[:num_of_fragments]])
            explanation_fragment_frames_index=list(np.array(fragment_frames_index)[lime_explanation_positive[:num_of_fragments]])
        #If there aren't, then create empty lists
        else:
            explanation_fragments=np.array([]).astype(int)
            explanation_fragment_frames=[]
            explanation_fragment_frames_index=[]
    else:
        print("Invalid Explanation Method!")
        sys.exit(0)

    #Return the fragments, the numbers of the sampled frames and their indexes
    return [explanation_fragments,explanation_fragment_frames,explanation_fragment_frames_index]

def explain_fragments_spatially(original_result, features, model, video_path, fragments, explanation_path, num_of_fragments,summary_fragments=False):

    #Create the necessary path to save all of the explanations if it doesn't already exists
    if (not (os.path.exists(explanation_path))):
        os.mkdir(explanation_path)

    #LIME perturbations and number of top features to highlight as explanation in the keyframe
    perturbations = 2000; num_features = 1

    #Set the fragment ranges, the numbers of the sampled frames of each fragment and their corresponding indexes used by the summarizer
    top_fragments,top_fragment_frames,top_fragment_frames_index=fragments

    #Remove the previous files from the explanation path
    files = os.listdir(explanation_path)
    for file in files:
        file_path = os.path.join(explanation_path, file)
        if os.path.isfile(file_path):
            os.remove(file_path)

    evaluation_results=[]
    #For each fragment
    for i in range(len(top_fragments)):

        fragment=top_fragments[i]
        fragment_frames=top_fragment_frames[i]
        fragment_frames_index=top_fragment_frames_index[i]

        frames = []
        #Extract and save the sampled fragment frames
        for frame_index in fragment_frames:
            frames.append(loadFrame(video_path,frame_index-1))

        #Get the index of the keyframe, corresponding to the sampled fragment frame with the biggest importance score
        keyframe_index=np.argmax(np.array(original_result)[fragment_frames_index])
        #Get the keyframe from the saved sampled fragment frames using the index
        keyframe=frames[keyframe_index]

        #Get the segmentation masks for the sampled fragment frames using Video K-Net
        segments=segmentFrames(frames, keyframe_index)

        #Create a object-level LIME explanator
        explainer = LimeSpatialExplainer()
        #Compute the object-level explanation
        explanation = explainer.explain_instances(perturbations, frames, features, fragment_frames_index, model, segments, keyframe_index)

        #Get the image and mask with the top 1 positive segment of the keyframe for explaining the fragment
        temp, mask = explanation.get_image_and_mask(keyframe, num_features, positive_only=True, hide_rest=True)
        #Make the segments that are not in the explanation half-bright for better visualization
        temp = np.uint8(0.5 * (keyframe) + 0.5 * (temp))
        #Save the explanation image with the superimposed importance areas
        plt.imsave(explanation_path + str(i+1) +'_explanation_' + str(fragment[0]) + '_' + str(fragment[1]) + '_pos.jpg', mark_boundaries(temp, mask))

        #Get the image and mask with the top 1 positive and negative segments of the keyframe for explaining the fragment
        temp, mask = explanation.get_image_and_mask(keyframe, num_features, positive_only=False, hide_rest=False)
        temp = np.uint8(0.5 * (keyframe) + 0.5 * (temp))
        plt.imsave(explanation_path + str(i+1) +'_explanation_' + str(fragment[0]) + '_' + str(fragment[1]) + '_pos_neg.jpg', mark_boundaries(temp, mask))

        #Create a visualization of the explanation mask with the top 1 positive and negative segments of the keyframe for explaining the fragment
        pos_neg_mask=np.ones(temp.shape)
        #Positive segment is green color
        pos_neg_mask[mask==1] = (0,1,0)
        #Negative segment is red color
        pos_neg_mask[mask==-1] = (1,0,0)
        plt.imsave(explanation_path + str(i + 1) + '_explanation_' + str(fragment[0]) + '_' + str(fragment[1]) + '_mask.jpg', pos_neg_mask)

        #Get the image and mask with the top 1 negative segment of the keyframe for explaining the fragment
        temp, mask = explanation.get_image_and_mask(keyframe, num_features, positive_only=False, negative_only=True, hide_rest=True)
        temp = np.uint8(0.5 * (keyframe) + 0.5 * (temp))
        plt.imsave(explanation_path + str(i+1) +'_explanation_' + str(fragment[0]) + '_' + str(fragment[1]) + '_neg.jpg', mark_boundaries(temp, mask))

        num_of_segs=3
        #Create a metric calculator object
        metrics_calculator = MetricsSpatialCalculator(model,features,frames,segments,fragment_frames_index)
        #Compute the disc plus, disc minus and sanity violation scores
        ranked_positive_segments = [x[0] for x in explanation.local_exp[0] if x[1] > 0]
        ranked_negative_segments = [x[0] for x in explanation.local_exp[0] if x[1] < 0 and abs(x[1]) > 0]
        discoverability_plus = metrics_calculator.compute_discoverability(ranked_positive_segments, num_of_segs, summary_fragments)
        discoverability_minus = metrics_calculator.compute_discoverability(ranked_negative_segments, num_of_segs, summary_fragments)
        sanity_violation = metrics_calculator.compute_sanity_violation(discoverability_plus,discoverability_minus)

        #If the number of segments is less than the desired number
        if (len(discoverability_plus[0]) < num_of_segs):
            #Append nan values in the evaluation scores for the missing segments
            while (len(discoverability_plus[0]) < num_of_segs):
                discoverability_plus[0].append(np.nan)

        if (len(discoverability_plus[1]) < num_of_segs):
            while (len(discoverability_plus[1]) < num_of_segs):
                discoverability_plus[1].append(np.nan)

        if (len(discoverability_minus[0]) < num_of_segs):
            while (len(discoverability_minus[0]) < num_of_segs):
                discoverability_minus[0].append(np.nan)

        if (len(discoverability_minus[1]) < num_of_segs):
            while (len(discoverability_minus[1]) < num_of_segs):
                discoverability_minus[1].append(np.nan)

        #Collect all of the evaluation scores
        evaluation_results.append([discoverability_plus[0][0],discoverability_plus[0][1],discoverability_plus[0][2],
                                   discoverability_plus[1][0], discoverability_plus[1][1], discoverability_plus[1][2],
                                   discoverability_minus[0][0], discoverability_minus[0][1], discoverability_minus[0][2],
                                   discoverability_minus[1][0], discoverability_minus[1][1], discoverability_minus[1][2],
                                   sanity_violation[0],sanity_violation[1]])

    #Define the column names of the dataframe
    columns_names=["Discoverability Plus One By One Top 1", "Discoverability Plus One By One Top 2", "Discoverability Plus One By One Top 3",
                   "Discoverability Plus Sequentially Top 1", "Discoverability Plus Sequentially Top 2", "Discoverability Plus Sequentially Top 3",
                   "Discoverability Minus One By One Top 1", "Discoverability Minus One By One Top 2", "Discoverability Minus One By One Top 3",
                   "Discoverability Minus Sequentially Top 1", "Discoverability Minus Sequentially Top 2", "Discoverability Minus Sequentially Top 3",
                   "Sanity Violation One By One", "Sanity Violation Sequentially"]

    #If the number of fragments is less than the desired number
    if (len(evaluation_results) < num_of_fragments):
        #Append lists with nan values in the evaluation scores for the missing fragments
        while (len(evaluation_results) < num_of_fragments):
            evaluation_results.append([np.nan]*14)

    #Create the dataframe with the scores
    df = pd.DataFrame(evaluation_results,columns=columns_names)
    #Save the dataframe to a csv file
    df.to_csv(explanation_path + "fragments_spatial_explanation_evaluation_metrics.csv", index=False)


def explain(original_result, features, model, video_path, num_of_fragments, explanation_path):

    #Set the object-level explanation method to Attention
    explanation_method = "Attention"
    #Run the fragment-level explanation and get the explanation fragments
    fragments = explain_fragments(features, model, video_path, num_of_fragments, explanation_path, explanation_method)
    #Run the object-level explanation on these fragments
    explain_fragments_spatially(original_result, features, model, video_path, fragments, explanation_path+explanation_method+'/', num_of_fragments)

    #Do the same for LIME
    explanation_method = "LIME"
    fragments = explain_fragments(features, model, video_path, num_of_fragments, explanation_path, explanation_method)
    explain_fragments_spatially(original_result, features, model, video_path, fragments, explanation_path+explanation_method+'/', num_of_fragments)

    #Get the top fragments included in the summary by the summarizer
    fragments = getFragments(video_path, original_result, num_of_fragments, True)
    #Run the object-level explanation on these fragments
    explain_fragments_spatially(original_result, features, model, video_path, fragments, explanation_path+'Top Fragments/', num_of_fragments, True)

if __name__=='__main__':

    #Parse the running parameters
    parser = argparse.ArgumentParser(description = "Usage")
    #Specify the path of the video summarizaton model
    parser.add_argument("-m", "--model", help = "Path of the video summarization model", required = True)
    #Specify the path of the video to explain
    parser.add_argument("-v", "--video", help = "Path of the video to explain", required = True)
    #Specify number of temporal fragments to explain, default values is 3
    parser.add_argument("-f", "--fragments", help="Number of temporal fragments to explain", required=False, default=3)
    argument = parser.parse_args()

    #Create the necessary path to save all of the needed files to explain the video if it doesn't already exists
    save_path = argument.video[:-4]
    if (not (os.path.exists(save_path))):
        os.mkdir(save_path)

    #Load the model from the path
    model = load_model(argument.model)
    #Load the deep features of the sampled frames of the video
    data = load_data(argument.video,model)
    #Get the original video summarization frames importance scores
    result = predict(data, model)

    #Explain the video
    explain(result, data, model, argument.video, int(argument.fragments), save_path+'/explanation/')