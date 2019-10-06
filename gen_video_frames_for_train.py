"""making image (video frame) sequence from given video files for training"""

# python packages
import cv2
import os



# project modules
from .. import root_dir



# declaring path variables
input_dir = root_dir.casia_gait_dataset_B()
output_dir = os.path.join(root_dir.data_path(), "video_frames")



# for training CASIA-B Dataset
def get_all_video_files_for_train (start_id, end_id):
    ls_video_files = []

    for subject_id in range(start_id, (end_id + 1)):
        if(subject_id < 10): id_num = "00" + str(subject_id)
        elif(subject_id < 100): id_num = "0" + str(subject_id)
        else: id_num = str(subject_id)

        # set training angle
        view_angles = [0, 18, 36, 54, 72, 90, 108, 126, 144, 162, 180]

        for v_a in view_angles:
            if (v_a == 0): vid_name = "00" + str(v_a) + ".avi"
            elif (v_a < 100): vid_name = "0" + str(v_a) + ".avi"
            else: vid_name = str(v_a) + ".avi"

            
        # input video files for all given angles
            sub_videos = []

            sub_videos.append(id_num + "-" + "nm" + "-01" + "-" + vid_name)
            sub_videos.append(id_num + "-" + "nm" + "-02" + "-" + vid_name)
            sub_videos.append(id_num + "-" + "nm" + "-03" + "-" + vid_name)
            sub_videos.append(id_num + "-" + "nm" + "-04" + "-" + vid_name)
            sub_videos.append(id_num + "-" + "nm" + "-05" + "-" + vid_name)
            sub_videos.append(id_num + "-" + "nm" + "-06" + "-" + vid_name)

            sub_videos.append(id_num + "-" + "bg" + "-01" + "-" + vid_name)
            sub_videos.append(id_num + "-" + "bg" + "-02" + "-" + vid_name)

            sub_videos.append(id_num + "-" + "cl" + "-01" + "-" + vid_name)
            sub_videos.append(id_num + "-" + "cl" + "-02" + "-" + vid_name)
            ls_video_files.append(sub_videos)

    return ls_video_files





# making video frames
def gen_video_frames(ls_video_files):
    for video_files in ls_video_files:
        
        save_subject_dir = (os.path.join(output_dir, "p" + video_files[0].split("-")[0]))
        os.makedirs(save_subject_dir, exist_ok = True)

        # getting angle info
        v_a = video_files[0].split(".")[0][-3:]
        save_angle_dir = (os.path.join(save_subject_dir, "angle_" + v_a))
        os.makedirs(save_angle_dir, exist_ok = True)



        # making all frame sequence per subject
        for i, input_file in enumerate(video_files):
            
            # getting sequence info
            seq_name = input_file.split("-")[1] + input_file.split("-")[2] 
            
            save_seq_dir = os.path.join(save_angle_dir, seq_name)
            os.makedirs(save_seq_dir, exist_ok = True)

            # capturing video
            vidcap = cv2.VideoCapture(os.path.join(input_dir, input_file))
            success, image = vidcap.read()

            if(vidcap.isOpened() == True):
                count = 0
                success = True

            while success:
                success, image = vidcap.read()
                count += 1
                print('reading a new frame: ', success)

                if(success == True):
                    #save frame as JPEG file
                    cv2.imwrite(os.path.join(save_seq_dir, ("%03d.jpg") % count), image)  
        




# set training subject id (including end id)
start_id = 67
end_id = 67

# getting associated video files
ls_video_files = get_all_video_files_for_train (start_id, end_id)


# making video frames for given videos
gen_video_frames(ls_video_files)






