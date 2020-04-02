#!/bin/bash
cd ..

### for convolutional model
# generating video frames
#python3 -m gait_recognition.preprocessing.gen_video_frames_for_train

# detecting and cropping
#python3 -m gait_recognition.preprocessing.process_video_frames_for_train

# train gallery the network
#python3 -m gait_recognition.spatio_temporal_network.src.train


# test gallery the network
#python3 -m gait_recognition.spatio_temporal_network.src.test

### for rnn model
#python3 -m gait_recognition.temporal_network.src.run_openpose


# train gallery the network
#python3 -m gait_recognition.temporal_network.src.train


# test gallery the network
#python3 -m gait_recognition.temporal_network.src.test

# run two stage network for final test
python3 -m gait_recognition.two_stage_network.final_test
