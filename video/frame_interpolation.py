import os
import subprocess
import cv2
from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
from modelscope.outputs import OutputKeys

def video_frame_interpolation(input_video):
    """ Apply video frame interpolation to a video. """
    try:
        video_frame_interpolation_pipeline = pipeline(Tasks.video_frame_interpolation, 'damo/cv_raft_video-frame-interpolation')
        print(video_frame_interpolation_pipeline.cfg)
        result = video_frame_interpolation_pipeline(input_video)[OutputKeys.OUTPUT_VIDEO]
        print('pipeline: the output video path is {}'.format(result))
        return result
    except Exception as e:
        print(f"Error during video frame interpolation: {e}")
        return None

def extract_frames(video_path, output_folder):
    """ Extract frames from a video using ffmpeg. """
    try:
        if os.path.exists(output_folder):
            # 清空文件夹中的内容
            for file in os.listdir(output_folder):
                file_path = os.path.join(output_folder, file)
                if os.path.isfile(file_path):
                    os.unlink(file_path)
        else:
            os.makedirs(output_folder)

        command = ['ffmpeg', '-i', video_path, '-vf', 'fps=24', os.path.join(output_folder, 'frame_%04d.png')]
        subprocess.run(command, check=True)
    except Exception as e:
        print(f"Error during frame extraction: {e}")

def apply_super_resolution(input_folder):
    """ Apply super resolution to each image in a folder. """
    try:
        sr = pipeline(Tasks.image_super_resolution, model='damo/cv_rrdb_image-super-resolution')
        for image_name in os.listdir(input_folder):
            img_path = os.path.join(input_folder, image_name)
            result = sr(img_path)
            cv2.imwrite(img_path, result[OutputKeys.OUTPUT_IMG])
    except Exception as e:
        print(f"Error during super resolution: {e}")

def reassemble_video(input_folder, output_video):
    """ Reassemble a video from image frames. """
    try:
        os.system(f'ffmpeg -framerate 24 -i {os.path.join(input_folder, "frame_%04d.png")} -c:v libx264 -pix_fmt yuv420p -vf "scale=trunc(iw/2)*2:trunc(ih/2)*2" {output_video}')
    except Exception as e:
        print(f"Error during video reassembly: {e}")

# Main process
video_path = '5ca8cb9263927621cd63535210202cf87f361b5dbd2b2475f3165aef.mp4'
interpolated_video = video_frame_interpolation(video_path)
if interpolated_video:
    extract_frames(interpolated_video, 'video_frames')
    apply_super_resolution('video_frames')
    reassemble_video('video_frames', 'output.mp4')