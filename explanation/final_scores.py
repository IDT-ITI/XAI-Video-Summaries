import os
import warnings
import pandas as pd
import numpy as np
EPSILON = 1e-10



#Select datasets to compute the final scores
dataset=["SumMe","TVSum"]
#Select the videos of each dataset used to compute the average
videos=[list(range(1, 26)),list(range(1, 51))]


def compute_sanity_violation(disc_plus, disc_minus):
    #Initialize the sv score
    sv=0
    #Compare the disc plus and disc minus scores and set the sv value
    for i in range(len(disc_plus)):
        if (disc_minus[i] < disc_plus[i]):
            sv += 1

    #Compute the return the mean sv score
    return sv / len(disc_plus)


#Compute the fragment-level evaluation scores separately for the top 1,2,3 fragments
def fragments_explanation_scores(video_path):

    #Read the csv file of the video containing the evaluation scores and place them into a dataframe
    df = pd.read_csv(video_path + "fragments_explanation_evaluation_metrics.csv")
    df.drop(columns=df.columns[0], axis=1, inplace=True)

    #Get the video scores from the dataframe
    video_scores = df.to_numpy()
    #Add or subtract a insignificant epsilon value from elements that are 1 or -1 respectively, in order to be able to compute the arctan of the scores
    video_scores[video_scores == 1.0] = 1.0 - EPSILON
    video_scores[video_scores == -1.0] = -1.0 + EPSILON

    #For the top 1 disc plus and bottom 1 disc minus scores (if they exist)
    if(not np.isnan(video_scores[0][0]) and not np.isnan(video_scores[0][2]) and not np.isnan(video_scores[0][6]) and not np.isnan(video_scores[0][8])):
        #Copy the top/bottom 1 scores
        temp=np.copy(video_scores[:1])

        #Compute and set the sanity violation scores for the top/bottom 1 fragments for Attention
        temp[0, 4] = compute_sanity_violation(temp[0:1, 0], temp[:, 2])
        temp[0, 5] = compute_sanity_violation(temp[0:1, 1], temp[:, 3])

        #Compute and set the sanity violation scores for the top/bottom 1 fragments for LIME
        temp[0, 10] = compute_sanity_violation(temp[0:1, 6], temp[:, 8])
        temp[0, 11] = compute_sanity_violation(temp[0:1, 7], temp[:, 9])

        #Compute the arctan values of the disc plus and disc minus scores (in order to average them as they can have values between -1 and 1)
        temp[:, 0:4] = np.arctanh(temp[:, 0:4])
        temp[:, 6:10] = np.arctanh(temp[:, 6:10])

        #Append the scores to the corresponding list
        fragment_scores_1.append(temp)

    #Do the same for the top 2 disc plus and bottom 2 disc minus scores (if they exist)
    if(not np.isnan(video_scores[1][0]) and not np.isnan(video_scores[1][2]) and not np.isnan(video_scores[1][6]) and not np.isnan(video_scores[1][8])):
        temp=np.copy(video_scores[:2])

        temp[0, 4] = compute_sanity_violation(temp[0:1, 0], temp[0:1, 2])
        temp[0, 5] = compute_sanity_violation(temp[0:1, 1], temp[0:1, 3])

        temp[0, 10] = compute_sanity_violation(temp[0:1, 6], temp[0:1, 8])
        temp[0, 11] = compute_sanity_violation(temp[0:1, 7], temp[0:1, 9])

        #Compute and set the sanity violation scores for the top/bottom 2 fragments for Attention
        temp[1, 4] = compute_sanity_violation(temp[1:2, 0], temp[1:2, 2])
        temp[1, 5] = compute_sanity_violation(temp[1:2, 1], temp[1:2, 3])

        #Compute and set the sanity violation scores for the top/bottom 2 fragments for LIME
        temp[1, 10] = compute_sanity_violation(temp[1:2, 6], temp[1:2, 8])
        temp[1, 11] = compute_sanity_violation(temp[1:2, 7], temp[1:2, 9])

        temp[:, 0:4] = np.arctanh(temp[:, 0:4])
        temp[:, 6:10] = np.arctanh(temp[:, 6:10])
        fragment_scores_2.append(temp)

    #Do the same for the top 3 disc plus and bottom 3 disc minus scores(if they exist)
    if (not np.isnan(video_scores[2][0]) and not np.isnan(video_scores[2][2]) and not np.isnan(video_scores[2][6]) and not np.isnan(video_scores[2][8])):
        temp = np.copy(video_scores[:3])

        temp[0, 4] = compute_sanity_violation(temp[0:1, 0], temp[0:1, 2])
        temp[0, 5] = compute_sanity_violation(temp[0:1, 1], temp[0:1, 3])

        temp[0, 10] = compute_sanity_violation(temp[0:1, 6], temp[0:1, 8])
        temp[0, 11] = compute_sanity_violation(temp[0:1, 7], temp[0:1, 9])

        temp[1, 4] = compute_sanity_violation(temp[1:2, 0], temp[1:2, 2])
        temp[1, 5] = compute_sanity_violation(temp[1:2, 1], temp[1:2, 3])

        temp[1, 10] = compute_sanity_violation(temp[1:2, 6], temp[1:2, 8])
        temp[1, 11] = compute_sanity_violation(temp[1:2, 7], temp[1:2, 9])

        #Compute and set the sanity violation scores for the top/bottom 3 fragments for Attention
        temp[2, 4] = compute_sanity_violation(temp[2:3, 0], temp[2:3, 2])
        temp[2, 5] = compute_sanity_violation(temp[2:3, 1], temp[2:3, 3])

        #Compute and set the sanity violation scores for the top/bottom 3 fragments for LIME
        temp[2, 10] = compute_sanity_violation(temp[2:3, 6], temp[2:3, 8])
        temp[2, 11] = compute_sanity_violation(temp[2:3, 7], temp[2:3, 9])

        temp[:, 0:4] = np.arctanh(temp[:, 0:4])
        temp[:, 6:10] = np.arctanh(temp[:, 6:10])
        fragment_scores_3.append(temp)

#Compute the object-level evaluation scores separately for the top 1,2,3 visual objects on the fragments returned by the fragment-level explanation for Attention and LIME
def fragments_spatial_explanation_scores(video_path):
    #Read the csv file of the video containing the evaluation Attention and LIME fragment scores and place them into a dataframe
    df_attention = pd.read_csv(video_path + "Attention/fragments_spatial_explanation_evaluation_metrics.csv")
    df_lime = pd.read_csv(video_path + "LIME/fragments_spatial_explanation_evaluation_metrics.csv")

    #Get the video scores from the dataframe for the desired number of fragments
    video_scores_attention = df_attention.to_numpy()[:3,:]
    video_scores_lime = df_lime.to_numpy()[:3,:]
    #Add or subtract a insignificant epsilon value from elements that are 1 or -1 respectively, in order to be able to compute the arctan of the scores
    video_scores_attention[video_scores_attention == 1.0] = 1.0 - EPSILON
    video_scores_lime[video_scores_lime == -1.0] = -1.0 + EPSILON

    t1_attention=[]; t2_attention=[]; t3_attention=[]
    t1_lime=[]; t2_lime=[]; t3_lime=[]

    #For each fragment
    for r in range(3):
        #For the top 1 disc plus and bottom 1 disc minus scores (if they exist for both Attention and LIME)
        if(not np.isnan(video_scores_attention[r][0]) and not np.isnan(video_scores_attention[r][6]) and not np.isnan(video_scores_lime[r][0]) and not np.isnan(video_scores_lime[r][6])):
            #Copy the top/bottom 1 Attention and LIME scores
            temp_attention = np.copy(video_scores_attention[r]).reshape(1, -1)
            temp_attention=np.column_stack((temp_attention, np.zeros((temp_attention.shape[0],4))))
            temp_lime = np.copy(video_scores_lime[r]).reshape(1, -1)
            temp_lime=np.column_stack((temp_lime, np.zeros((temp_lime.shape[0],4))))

            #Discard the scores for the top 2 and top 3 visual objects for Attention and LIME fragments
            temp_attention[0][1] = np.nan; temp_attention[0][2] = np.nan
            temp_attention[0][4] = np.nan; temp_attention[0][5] = np.nan
            temp_attention[0][7] = np.nan; temp_attention[0][8] = np.nan
            temp_attention[0][10:] = np.nan

            temp_lime[0][1] = np.nan; temp_lime[0][2] = np.nan
            temp_lime[0][4] = np.nan; temp_lime[0][5] = np.nan
            temp_lime[0][7] = np.nan; temp_lime[0][8] = np.nan
            temp_lime[0][10:] = np.nan

            #Compute and set the sanity violation scores for the top/bottom 1 visual objects for Attention
            temp_attention[0][12] = compute_sanity_violation(temp_attention[0,0:1], temp_attention[0,6:7])
            temp_attention[0][13] = compute_sanity_violation(temp_attention[0,3:4], temp_attention[0,9:10])

            #Compute and set the sanity violation scores for the top/bottom 1 visual objects for LIME
            temp_lime[0][12] = compute_sanity_violation(temp_lime[0, 0:1], temp_lime[0, 6:7])
            temp_lime[0][13] = compute_sanity_violation(temp_lime[0, 3:4], temp_lime[0, 9:10])

            #Compute the arctan values of the disc plus and disc minus scores (in order to average them as they can have values between -1 and 1)
            temp_attention[:, :12] = np.arctanh(temp_attention[:, :12])
            temp_lime[:, :12] = np.arctanh(temp_lime[:, :12])

            #Append the scores to the corresponding lists
            t1_attention.append(temp_attention)
            t1_lime.append(temp_lime)

        else:
            #Otherwise append empty arrays
            t1_attention.append(np.full((1, 18), np.nan))
            t1_lime.append(np.full((1, 18), np.nan))

        #Do the same for the top 2 disc plus and bottom 2 disc minus scores
        if (not np.isnan(video_scores_attention[r][1]) and not np.isnan(video_scores_attention[r][7]) and not np.isnan(video_scores_lime[r][1]) and not np.isnan(video_scores_lime[r][7])):
            temp_attention = np.copy(video_scores_attention[r]).reshape(1, -1)
            temp_attention=np.column_stack((temp_attention, np.zeros((temp_attention.shape[0],4))))
            temp_lime = np.copy(video_scores_lime[r]).reshape(1, -1)
            temp_lime=np.column_stack((temp_lime, np.zeros((temp_lime.shape[0],4))))

            temp_attention[0][2] = np.nan; temp_attention[0][5] = np.nan
            temp_attention[0][8] = np.nan; temp_attention[0][11:] = np.nan

            temp_lime[0][2] = np.nan; temp_lime[0][5] = np.nan
            temp_lime[0][8] = np.nan; temp_lime[0][11:] = np.nan

            temp_attention[0][12] = compute_sanity_violation(temp_attention[0,0:1], temp_attention[0,6:7])
            temp_attention[0][13] = compute_sanity_violation(temp_attention[0,3:4], temp_attention[0,9:10])

            temp_lime[0][12] = compute_sanity_violation(temp_lime[0, 0:1], temp_lime[0, 6:7])
            temp_lime[0][13] = compute_sanity_violation(temp_lime[0, 3:4], temp_lime[0, 9:10])

            #Compute and set the sanity violation scores for the top/bottom 2 visual objects for Attention
            temp_attention[0][14] = compute_sanity_violation(temp_attention[0, 1:2], temp_attention[0, 7:8])
            temp_attention[0][15] = compute_sanity_violation(temp_attention[0, 4:5], temp_attention[0, 10:11])

            #Compute and set the sanity violation scores for the top/bottom 2 visual objects for Attention
            temp_lime[0][14] = compute_sanity_violation(temp_lime[0, 1:2], temp_lime[0, 7:8])
            temp_lime[0][15] = compute_sanity_violation(temp_lime[0, 4:5], temp_lime[0, 10:11])

            temp_attention[:, :12] = np.arctanh(temp_attention[:, :12])
            temp_lime[:, :12] = np.arctanh(temp_lime[:, :12])

            t2_attention.append(temp_attention)
            t2_lime.append(temp_lime)

        else:
            t2_attention.append(np.full((1, 18), np.nan))
            t2_lime.append(np.full((1, 18), np.nan))

        #Do the same for the top 3 disc plus and bottom 3 disc minus scores
        if (not np.isnan(video_scores_attention[r][2]) and not np.isnan(video_scores_attention[r][8]) and not np.isnan(video_scores_lime[r][2]) and not np.isnan(video_scores_lime[r][8])):
            temp_attention = np.copy(video_scores_attention[r]).reshape(1, -1)
            temp_attention=np.column_stack((temp_attention, np.zeros((temp_attention.shape[0],4))))
            temp_lime = np.copy(video_scores_lime[r]).reshape(1, -1)
            temp_lime=np.column_stack((temp_lime, np.zeros((temp_lime.shape[0],4))))

            temp_attention[0][12:] = np.nan
            temp_lime[0][12:] = np.nan

            temp_attention[0][12] = compute_sanity_violation(temp_attention[0,0:1], temp_attention[0,6:7])
            temp_attention[0][13] = compute_sanity_violation(temp_attention[0,3:4], temp_attention[0,9:10])

            temp_lime[0][12] = compute_sanity_violation(temp_lime[0, 0:1], temp_lime[0, 6:7])
            temp_lime[0][13] = compute_sanity_violation(temp_lime[0, 3:4], temp_lime[0, 9:10])

            temp_attention[0][14] = compute_sanity_violation(temp_attention[0, 1:2], temp_attention[0, 7:8])
            temp_attention[0][15] = compute_sanity_violation(temp_attention[0, 4:5], temp_attention[0, 10:11])

            temp_lime[0][14] = compute_sanity_violation(temp_lime[0, 1:2], temp_lime[0, 7:8])
            temp_lime[0][15] = compute_sanity_violation(temp_lime[0, 4:5], temp_lime[0, 10:11])

            #Compute and set the sanity violation scores for the top/bottom 3 visual objects for Attention
            temp_attention[0][16] = compute_sanity_violation(temp_attention[0, 2:3], temp_attention[0, 8:9])
            temp_attention[0][17] = compute_sanity_violation(temp_attention[0, 5:6], temp_attention[0, 11:12])

            #Compute and set the sanity violation scores for the top/bottom 3 visual objects for LIME
            temp_lime[0][16] = compute_sanity_violation(temp_lime[0, 2:3], temp_lime[0, 8:9])
            temp_lime[0][17] = compute_sanity_violation(temp_lime[0, 5:6], temp_lime[0, 11:12])

            temp_attention[:, :12] = np.arctanh(temp_attention[:, :12])
            temp_lime[:, :12] = np.arctanh(temp_lime[:, :12])

            t3_attention.append(temp_attention)
            t3_lime.append(temp_lime)

        else:
            t3_attention.append(np.full((1, 18), np.nan))
            t3_lime.append(np.full((1, 18), np.nan))

    #Vertically stack and append the scores of each fragment to the corresponding lists
    attention_scores_1.append(np.vstack(t1_attention))
    attention_scores_2.append(np.vstack(t2_attention))
    attention_scores_3.append(np.vstack(t3_attention))

    lime_scores_1.append(np.vstack(t1_lime))
    lime_scores_2.append(np.vstack(t2_lime))
    lime_scores_3.append(np.vstack(t3_lime))

#Compute the object-level evaluation scores separately for the top 1,2,3 visual objects on the fragments returned by the summarizer to be included in the summary
def top_fragments_explanation_scores(video_path):
    #Read the csv file of the video containing the evaluation fragment scores and place them into a dataframe
    df = pd.read_csv(video_path + "Top Fragments/fragments_spatial_explanation_evaluation_metrics.csv")

    #Get the video scores from the dataframe for the desired number of fragments
    video_scores = df.to_numpy()[:3,:]
    #Add or subtract a insignificant epsilon value from elements that are 1 or -1 respectively, in order to be able to compute the arctan of the scores
    video_scores[video_scores == 1.0] = 1.0 - EPSILON
    video_scores[video_scores == -1.0] = -1.0 + EPSILON

    t1=[]; t2=[]; t3=[]

    #For each fragment
    for r in range(3):
        #For the top 1 disc plus and bottom 1 disc minus scores (if they exist)
        if(not np.isnan(video_scores[r][0]) and not np.isnan(video_scores[r][6])):
            #Copy the top/bottom 1 scores
            temp=np.copy(video_scores[r]).reshape(1,-1)
            temp = np.column_stack((temp, np.zeros((temp.shape[0], 4))))

            #Discard the scores for the top 2 and top 3 visual objects for the summary fragments
            temp[0][1] = np.nan; temp[0][2] = np.nan
            temp[0][4] = np.nan; temp[0][5] = np.nan
            temp[0][7] = np.nan; temp[0][8] = np.nan
            temp[0][10:] = np.nan

            #Compute and set the sanity violation scores for the top/bottom 1 visual objects
            temp[0][12] = compute_sanity_violation(temp[0,0:1], temp[0,6:7])
            temp[0][13] = compute_sanity_violation(temp[0,3:4], temp[0,9:10])

            # Compute the arctan values of the disc plus and disc minus scores (in order to average them as they can have values between -1 and 1)
            temp[:, :12] = np.arctanh(temp[:, :12])

            #Append the scores to the corresponding list
            t1.append(temp)
        else:
            #Otherwise append empty arrays
            t1.append(np.full((1, 18), np.nan))

        #Do the same for the top 2 disc plus and bottom 2 disc minus scores
        if (not np.isnan(video_scores[r][1]) and not np.isnan(video_scores[r][7])):
            temp = np.copy(video_scores[r]).reshape(1, -1)
            temp = np.column_stack((temp, np.zeros((temp.shape[0], 4))))

            temp[0][2] = np.nan; temp[0][5] = np.nan
            temp[0][8] = np.nan; temp[0][11:] = np.nan

            temp[0][12] = compute_sanity_violation(temp[0,0:1], temp[0,6:7])
            temp[0][13] = compute_sanity_violation(temp[0,3:4], temp[0,9:10])

            # Compute and set the sanity violation scores for the top/bottom 2 visual objects
            temp[0][14] = compute_sanity_violation(temp[0, 1:2], temp[0, 7:8])
            temp[0][15] = compute_sanity_violation(temp[0, 4:5], temp[0, 10:11])

            temp[:, :12] = np.arctanh(temp[:, :12])
            t2.append(temp)
        else:
            t2.append(np.full((1, 18), np.nan))

        #Do the same for the top 3 disc plus and bottom 3 disc minus scores
        if (not np.isnan(video_scores[r][2]) and not np.isnan(video_scores[r][8])):
            temp = np.copy(video_scores[r]).reshape(1, -1)
            temp = np.column_stack((temp, np.zeros((temp.shape[0], 4))))

            temp[0][12:] = np.nan

            temp[0][12] = compute_sanity_violation(temp[0,0:1], temp[0,6:7])
            temp[0][13] = compute_sanity_violation(temp[0,3:4], temp[0,9:10])

            temp[0][14] = compute_sanity_violation(temp[0, 1:2], temp[0, 7:8])
            temp[0][15] = compute_sanity_violation(temp[0, 4:5], temp[0, 10:11])

            # Compute and set the sanity violation scores for the top/bottom 3 visual objects
            temp[0][16] = compute_sanity_violation(temp[0, 2:3], temp[0, 8:9])
            temp[0][17] = compute_sanity_violation(temp[0, 5:6], temp[0, 11:12])

            temp[:, :12] = np.arctanh(temp[:, :12])
            t3.append(temp)
        else:
            t3.append(np.full((1, 18), np.nan))

    #Vertically stack and append the scores of each fragment to the corresponding lists
    top_fragments_scores_1.append(np.vstack(t1))
    top_fragments_scores_2.append(np.vstack(t2))
    top_fragments_scores_3.append(np.vstack(t3))

#For each dataset
for d in range(len(dataset)):

    fragment_scores_1 = []; fragment_scores_2 = []; fragment_scores_3 = []
    attention_scores_1 = []; attention_scores_2 = []; attention_scores_3 = []
    lime_scores_1 = []; lime_scores_2 = []; lime_scores_3 = []
    top_fragments_scores_1 = []; top_fragments_scores_2 = []; top_fragments_scores_3 = []

    #For each video of the dataset
    for v in range(len(videos[d])):
        #Set the explanation path containing the video evaluation scores
        video_path = "../../CA-SUM/data/" + dataset[d] + "/video_" + str(videos[d][v]) + "/explanation/"

        #Compute the evaluation scores for the top 1,2,3 fragments and visual objects seperately
        #The videos containing more than top 1 fragments/visual objects are a subset
        fragments_explanation_scores(video_path)
        fragments_spatial_explanation_scores(video_path)
        top_fragments_explanation_scores(video_path)

    #Average the evaluation scores of the videos
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)

        #Compute the mean for the fragment-level scores on the top and bottom 1 fragment for all the videos
        fragment_scores_1=np.nanmean(fragment_scores_1,axis=0)
        #Revert the disc plus and disc minus back with computing their tanh values
        fragment_scores_1[:, 0:4] = np.tanh(fragment_scores_1[:, 0:4])
        fragment_scores_1[:, 6:10] = np.tanh(fragment_scores_1[:, 6:10])

        fragment_scores_2=np.nanmean(fragment_scores_2,axis=0)
        fragment_scores_2[:, 0:4] = np.tanh(fragment_scores_2[:, 0:4])
        fragment_scores_2[:, 6:10] = np.tanh(fragment_scores_2[:, 6:10])

        fragment_scores_3=np.nanmean(fragment_scores_3,axis=0)
        fragment_scores_3[:, 0:4] = np.tanh(fragment_scores_3[:, 0:4])
        fragment_scores_3[:, 6:10] = np.tanh(fragment_scores_3[:, 6:10])

        #Compute the mean for the object-level scores on the top and bottom 1 visual objects with the fragments returned by the fragment-level explanation with Attention for all the videos
        #Average between fragments first
        attention_scores_1=np.nanmean(attention_scores_1,axis=0)
        attention_scores_1=np.nanmean(attention_scores_1,axis=0).reshape(1,-1)
        #Revert the disc plus and disc minus back with computing their tanh values
        attention_scores_1[:,:12] = np.tanh(attention_scores_1[:,:12])

        attention_scores_2=np.nanmean(attention_scores_2,axis=0)
        attention_scores_2=np.nanmean(attention_scores_2,axis=0).reshape(1,-1)
        attention_scores_2[:,:12] = np.tanh(attention_scores_2[:,:12])

        attention_scores_3=np.nanmean(attention_scores_3,axis=0)
        attention_scores_3=np.nanmean(attention_scores_3,axis=0).reshape(1,-1)
        attention_scores_3[:,:12] = np.tanh(attention_scores_3[:,:12])

        #Same for LIME
        lime_scores_1=np.nanmean(lime_scores_1,axis=0)
        lime_scores_1=np.nanmean(lime_scores_1,axis=0).reshape(1,-1)
        lime_scores_1[:, :12] = np.tanh(lime_scores_1[:, :12])

        lime_scores_2=np.nanmean(lime_scores_2,axis=0)
        lime_scores_2=np.nanmean(lime_scores_2,axis=0).reshape(1,-1)
        lime_scores_2[:, :12] = np.tanh(lime_scores_2[:, :12])

        lime_scores_3=np.nanmean(lime_scores_3,axis=0)
        lime_scores_3=np.nanmean(lime_scores_3,axis=0).reshape(1,-1)
        lime_scores_3[:, :12] = np.tanh(lime_scores_3[:, :12])

        #Compute the mean for the object-level scores on the top and bottom 1 visual objects with the fragments returned by the summarizer to be included in the summary for all the videos
        #Average between fragments first
        top_fragments_scores_1=np.nanmean(top_fragments_scores_1,axis=0)
        top_fragments_scores_1=np.nanmean(top_fragments_scores_1,axis=0).reshape(1,-1)
        #Revert the disc plus and disc minus back with computing their tanh values
        top_fragments_scores_1[:, :12] = np.tanh(top_fragments_scores_1[:, :12])

        top_fragments_scores_2=np.nanmean(top_fragments_scores_2,axis=0)
        top_fragments_scores_2=np.nanmean(top_fragments_scores_2,axis=0).reshape(1,-1)
        top_fragments_scores_2[:, :12] = np.tanh(top_fragments_scores_2[:, :12])

        top_fragments_scores_3=np.nanmean(top_fragments_scores_3,axis=0)
        top_fragments_scores_3=np.nanmean(top_fragments_scores_3,axis=0).reshape(1,-1)
        top_fragments_scores_3[:, :12] = np.tanh(top_fragments_scores_3[:, :12])

    #Set the path to save the final scores
    scores_path="./final_scores/"+ dataset[d] + "/"
    #If it does not already exist then create it
    if (not (os.path.exists(scores_path))):
        os.makedirs(scores_path)

    #Define the column names of the dataframe
    columns_names=["Attention Disc Plus One By One", "Attention Disc Plus Sequentially", "Attention Disc Minus One By One", "Attention Disc Minus Sequentially",
                   "Attention Sanity Violation One By One", "Attention Sanity Violation Sequentially", "Lime Disc Plus One By One", "Lime Disc Plus Sequentially",
                   "Lime Disc Minus One By One", "Lime Disc Minus Sequentially", "Lime Sanity Violation One By One", "Lime Sanity Violation Sequentially"]

    #Create and save as csv file the fragment-level scores for the top 1,2,3 fragments
    df1 = pd.DataFrame(fragment_scores_1, columns=columns_names)
    df1.index = ['Top 1']
    df2 = pd.DataFrame(fragment_scores_2)
    df2.index = ['Top 1', 'Top 2']
    df3 = pd.DataFrame(fragment_scores_3)
    df3.index = ['Top 1', 'Top 2', 'Top 3']

    with open(scores_path + "fragment_explanation_scores.csv", 'w') as f:
        df1.to_csv(f)
        f.write('\n')
        df2.to_csv(f, header=False)
        f.write('\n')
        df3.to_csv(f, header=False)

    #Define the column names of the dataframe
    columns_names = ["Disc Plus One By One", "Disc Plus Sequentially",
                     "Disc Minus One By One", "Disc Minus Sequentially",
                     "Sanity Violation One By One", "Sanity Violation Sequentially"]

    attention_scores_1=attention_scores_1[0,[0,3,6,9,12,13]].reshape(1,-1)
    attention_scores_2=np.array((attention_scores_2[0,[0,3,6,9,12,13]],attention_scores_2[0,[1,4,7,10,14,15]]))
    attention_scores_3=np.array((attention_scores_3[0,[0,3,6,9,12,13]],attention_scores_3[0,[1,4,7,10,14,15]],attention_scores_3[0,[2,5,8,11,16,17]]))

    #Create and save as csv files the object-level scores for the top 1,2,3 visual objects on the fragments returned by the summarizer and the fragment-level explanation methods
    df1 = pd.DataFrame(attention_scores_1, columns=columns_names)
    df1.index = ['Top 1']
    df2 = pd.DataFrame(attention_scores_2, columns=columns_names)
    df2.index = ['Top 1', 'Top 2']
    df3 = pd.DataFrame(attention_scores_3, columns=columns_names)
    df3.index = ['Top 1', 'Top 2', 'Top 3']

    with open(scores_path + "spatial_explanation_scores_attention_fragments.csv", 'w') as f:
        df1.to_csv(f)
        f.write('\n')
        df2.to_csv(f, header=False)
        f.write('\n')
        df3.to_csv(f, header=False)


    lime_scores_1=lime_scores_1[0,[0,3,6,9,12,13]].reshape(1,-1)
    lime_scores_2=np.array((lime_scores_2[0,[0,3,6,9,12,13]],lime_scores_2[0,[1,4,7,10,14,15]]))
    lime_scores_3=np.array((lime_scores_3[0,[0,3,6,9,12,13]],lime_scores_3[0,[1,4,7,10,14,15]],lime_scores_3[0,[2,5,8,11,16,17]]))


    df1 = pd.DataFrame(lime_scores_1, columns=columns_names)
    df1.index = ['Top 1']
    df2 = pd.DataFrame(lime_scores_2, columns=columns_names)
    df2.index = ['Top 1', 'Top 2']
    df3 = pd.DataFrame(lime_scores_3, columns=columns_names)
    df3.index = ['Top 1', 'Top 2', 'Top 3']

    with open(scores_path + "spatial_explanation_scores_lime_fragments.csv", 'w') as f:
        df1.to_csv(f)
        f.write('\n')
        df2.to_csv(f, header=False)
        f.write('\n')
        df3.to_csv(f, header=False)

    top_fragments_scores_1=top_fragments_scores_1[0,[0,3,6,9,12,13]].reshape(1,-1)
    top_fragments_scores_2=np.array((top_fragments_scores_2[0,[0,3,6,9,12,13]],top_fragments_scores_2[0,[1,4,7,10,14,15]]))
    top_fragments_scores_3=np.array((top_fragments_scores_3[0,[0,3,6,9,12,13]],top_fragments_scores_3[0,[1,4,7,10,14,15]],top_fragments_scores_3[0,[2,5,8,11,16,17]]))

    df1 = pd.DataFrame(top_fragments_scores_1, columns=columns_names)
    df1.index = ['Top 1']
    df2 = pd.DataFrame(top_fragments_scores_2, columns=columns_names)
    df2.index = ['Top 1', 'Top 2']
    df3 = pd.DataFrame(top_fragments_scores_3, columns=columns_names)
    df3.index = ['Top 1', 'Top 2', 'Top 3']

    with open(scores_path + "spatial_explanation_scores_summary_fragments.csv", 'w') as f:
        df1.to_csv(f)
        f.write('\n')
        df2.to_csv(f, header=False)
        f.write('\n')
        df3.to_csv(f, header=False)