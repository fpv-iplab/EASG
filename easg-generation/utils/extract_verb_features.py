import pandas as pd
import json
import os
import json
import torch

def clip_to_windows(root_folder):
    result_dict = {}
    for subdir, _, files in os.walk(root_folder):
        for file in files:
            if file.endswith(".json"):
                annot_uid = os.path.basename(subdir)  # assuming annot_uid is the subfolder name
                file_path = os.path.join(subdir, file)
                with open(file_path, 'r') as json_file:
                    data = json.load(json_file)
                    
                    clip_uid = data.get('clip_uid')
                    pre_frame_number = data.get('pre_frame')['clip_frame_number']
                    pnr_frame_number = data.get('pnr_frame')['clip_frame_number']
                    post_frame_number = data.get('post_frame')['clip_frame_number']
                    
                    video_uid = data.get('video_uid')
                    pre_frame_number_video = data.get('pre_frame_number')
                    pnr_frame_number_video = data.get('pnr_frame_number')
                    post_frame_number_video = data.get('post_frame_number')
                    
                    if video_uid:
                        if video_uid not in result_dict:
                            result_dict[video_uid] = {}
                    result_dict[video_uid][annot_uid] = (pre_frame_number_video, post_frame_number_video)
    return result_dict


def calculate_mean_feature(file_path, start_frame, end_frame, window_size=32, stride=16):
    
    # Calculate the number of features
    num_features = features.shape[0]
    
    # Initialize a list to collect the relevant feature vectors
    relevant_features = []
    
    # Iterate over each feature vector to check if it's in the specified range
    for i in range(num_features):
        # Calculate the frame range for the current feature
        frame_start = i * stride
        frame_end = frame_start + window_size
        
        # Check if the feature vector is in the specified range
        if frame_end > start_frame and frame_start < end_frame:
            # Check that the last frame does not exceed the end frame by more than 16
            if frame_end > end_frame + 16:
                continue
            # Add the feature vector to the relevant features list
            relevant_features.append(features[i])

    # Convert the list of tensors to a tensor
    relevant_features_tensor = torch.stack(relevant_features)
    
    # Calculate the mean of the relevant feature vectors
    mean_feature = torch.mean(relevant_features_tensor, dim=0)
    
    return mean_feature

def main():
    FPS = 30
    W = 32
    S = 16
    clip_2_frames = clip_to_windows('annotations_all')
    res_dict = {}
    for clip_uid, data in clip_2_frames.items():
        features = torch.load("slowfast8x8_r101_k400/{}.pt".format(clip_uid))
        for uid, window in data.items():
            mean_fea = calculate_mean_feature(features, window[0], window[1])
            res_dict[uid] = mean_fea
            
    torch.save(res_dict, 'verb_features.pt')

if __name__ == "__main__":
    main()