import cv2
from features.feature_extraction import extract_fragment_features
from explanation.utils import predict
import numpy as np
import torch
from scipy.stats import kendalltau
from abc import ABC, abstractmethod

class MetricsCalculator(ABC):
    @abstractmethod
    def compute_discoverability(self):
        pass

    def compute_sanity_violation(self, disc_plus, disc_minus):
        #Initialize the sanity violation scores for one-by-one and sequential (batch) discoverability scores
        sv1 = 0; sv2 = 0
        #Ensure that the number of scores for comparison is the minimum number of the top or bottom amount of fragments/segments
        num = min(len(disc_plus[0]), len(disc_minus[0]))
        for i in range(num):
            if (disc_minus[0][i] < disc_plus[0][i]):
                #Sanity violation one-by-one
                sv1 += 1
            if (disc_minus[1][i] < disc_plus[1][i]):
                #Sanity violation sequentially (batch)
                sv2 += 1

        #Compute the mean score(if there were any discoverability scores)
        if (num > 0):
            sv1 = sv1 / num; sv2 = sv2 / num
        else:
            sv1 = np.nan; sv2 = np.nan

        #Return the sanity violation metrics
        return [sv1, sv2]

#Visual-level explanation evaluation metrics
class MetricsSpatialCalculator(MetricsCalculator):
    def __init__(self, model, original_features, frames, segments, fragments_frame_index):
        self.model = model
        self.features = original_features
        self.segments = np.array(segments)-1
        self.frames = frames
        self.fragments_frame_index=fragments_frame_index

    #Compute the discoverability scores
    def compute_discoverability(self,ranked_segments,number_of_segments,only_fragment_scores):
        #If we only want to consider the scores of the current fragment (fragment selected by the summarizer to be included in the summary)
        if(only_fragment_scores):
            #Keep the frame scores of the fragment frames only
            result = predict(self.features, self.model)[self.fragments_frame_index[0]:self.fragments_frame_index[-1]+1]
        #Otherwise
        else:
            #Keep all of the frame scores
            result = predict(self.features, self.model)

        results = []
        #Compute the discoverability scores in a one-by-one manner (mask out one segment each time)
        #For the number of top segments (if they exist)
        for s in ranked_segments[:number_of_segments]:
            #Mask out the current segment from all the fragment frames
            masked = np.array(self.frames)
            masked[self.segments==s] = (0,0,0)
            masked = list(masked)
            #Resize the masked frames
            for k in range(len(masked)):
                masked[k] = cv2.resize(masked[k], (224, 224))
            #Extract the deep features
            frame_features=extract_fragment_features(np.array(masked))
            frame_features = torch.Tensor(np.array(frame_features)).view(-1, 1024)
            frame_features = frame_features.to(self.model.linear_1.weight.device)
            new_features = self.features.detach().clone()
            #Replace the new deep features of the masked fragment
            new_features[self.fragments_frame_index]=frame_features
            # If we only want to consider the scores of the current fragment
            if(only_fragment_scores):
                #Keep the new frame scores of the masked fragment frames only
                masked_result=predict(new_features,self.model)[self.fragments_frame_index[0]:self.fragments_frame_index[-1]+1]
            #Otherwise
            else:
                #Keep all of the new frame scores
                masked_result = predict(new_features,self.model)
            #Compute the kendall coefficient between the original and the masked fragment frames scores
            results.append(kendalltau(result, masked_result)[0])
        discoverability1 = results

        results = []
        masked = np.array(self.frames)
        #Similarly, compute the discoverability scores in a sequential (batch) manner (mask out the segments sequentially)
        for s in ranked_segments[:number_of_segments]:
            masked=np.array(masked)
            masked[self.segments == s] = (0,0,0)
            masked_i = list(masked)
            for k in range(len(masked_i)):
                masked_i[k] = cv2.resize(masked_i[k], (224, 224))
            frame_features = extract_fragment_features(np.array(masked_i))
            frame_features = torch.Tensor(np.array(frame_features)).view(-1, 1024)
            frame_features = frame_features.to(self.model.linear_1.weight.device)
            new_features = self.features.detach().clone()
            new_features[self.fragments_frame_index] = frame_features
            if(only_fragment_scores):
                masked_result = predict(new_features, self.model)[self.fragments_frame_index[0]:self.fragments_frame_index[-1] + 1]
            else:
                masked_result = predict(new_features, self.model)
            results.append(kendalltau(result, masked_result)[0])
        discoverability2 = results

        #Return the discoverability scores
        return [discoverability1,discoverability2]

#Fragment-level explanation evaluation metrics
class MetricsFragmentCalculator(MetricsCalculator):
    def __init__(self, model, original_features, fragments_frame_index):
        self.model = model
        self.features = original_features
        self.fragments_frame_index=fragments_frame_index

    #Compute the discoverability scores
    def compute_discoverability(self, fragments, number_of_fragments):
        results=[]
        result = predict(self.features, self.model)
        #Compute the discoverability scores in a one-by-one manner (mask out one fragment each time)
        #For the number of top fragments (if they exist)
        for s in fragments[:number_of_fragments]:
            #Mask out the current fragment frames with black frames
            masked = [np.zeros((224, 224, 3), dtype=np.float32) for _ in range(len(self.fragments_frame_index[s]))]
            #Extract the deep features
            frame_features=extract_fragment_features(np.array(masked))
            frame_features = torch.Tensor(np.array(frame_features)).view(-1, 1024)
            frame_features = frame_features.to(self.model.linear_1.weight.device)
            new_features = self.features.detach().clone()
            #Replace the new deep features of the masked fragment
            new_features[self.fragments_frame_index[s]]=frame_features
            masked_result=predict(new_features,self.model)
            #Compute the kendall coefficient between the original and the masked fragment scores
            results.append(kendalltau(result, masked_result)[0])
        discoverability1 = results

        #Similarly, compute the discoverability scores in a sequential (batch) manner (mask out the fragments sequentially)
        results = []
        new_features = self.features.detach().clone()
        for s in fragments[:number_of_fragments]:
            masked = [np.zeros((224, 224, 3), dtype=np.float32) for _ in range(len(self.fragments_frame_index[s]))]
            frame_features = extract_fragment_features(np.array(masked))
            frame_features = torch.Tensor(np.array(frame_features)).view(-1, 1024)
            frame_features = frame_features.to(self.model.linear_1.weight.device)
            new_features[self.fragments_frame_index[s]] = frame_features
            masked_result = predict(new_features, self.model)
            results.append(kendalltau(result, masked_result)[0])
        discoverability2 = results

        return [discoverability1,discoverability2]
