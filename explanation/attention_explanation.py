import numpy as np

#Attention-based explanation to compute the fragment-level explanation using a model-specific approach
def explain_with_attention(features,model,fragment_frames_index):
    #Get the attention weights returned by the model
    _, attn_weights = model(features)
    attn_weights = attn_weights.detach().cpu().numpy()
    #Get the frame attention weights from the diagonal
    attn_frame_weights=attn_weights.diagonal()

    #Compute the score of each fragment by averaging the frame attention weights of its frames
    scores=[]
    for fragment in fragment_frames_index:
        scores.append(np.mean(attn_frame_weights[fragment]))

    #Return the ranked fragment indexes (reverse the array to get the fragments with descending score order)
    return np.argsort(scores)[::-1]