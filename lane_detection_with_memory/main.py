import cv2      # For video/image processing
from moviepy.editor import VideoFileClip    # For image recording and saving

from lane_detection_algorithm import process_frame

def video_process(path_in, path_out, mode, resolution):
    if mode == "read":
        video = cv2.VideoCapture(path_in) # Read video with opecv
        while True:            
            ret, frame = video.read()   # Read each frame of the video
            
            # Change the resolution
            if ret:     # Process while video is open
                if resolution == (1280, 720):
                    resized = cv2.resize(frame, resolution)
                elif resolution == (1024, 768):
                    resized = cv2.resize(frame, resolution)
                elif resolution == (800, 600):
                    resized = cv2.resize(frame, resolution)
                elif resolution == (640, 480):
                    resized = cv2.resize(frame, resolution)
                elif resolution == (400, 300):
                    resized = cv2.resize(frame, resolution)
                else:
                    raise Exception('No RESOLUTION IS SET!')

                final_frame = process_frame(resized)    # Process each frames
                cv2.imshow("Result", final_frame) # Output processed frames

                if cv2.waitKey(1) == 27:    # Break if ESC is pressed
                    break
            else:
                break   # Break when video is over.

        video.release() # Release frame
        cv2.destroyAllWindows() # Destroy all windows
    
    elif mode == "write":
        video_clip = VideoFileClip(path_in)   # Read image with moviepy
        project_video = video_clip.fl_image(process_frame)    # Process each frame of the video
        project_video.write_videofile(path_out, audio=False)    # Save whole video
    
    else:
        raise Exception("CHOOSE THE VIDEO MODE!")

if __name__ == "__main__":
    video_process("../input/white_road.mp4", "../output/output_lane_detection.mp4", "read", (1280, 720))