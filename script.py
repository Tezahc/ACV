from matplotlib import pyplot as plt
import numpy as np
from collections import deque
from PoseClassification.utils import show_image
from PoseClassification.pose_embedding import FullBodyPoseEmbedding
from PoseClassification.pose_classifier import PoseClassifier
from PoseClassification.utils import EMADictSmoothing
from PoseClassification.utils import RepetitionCounter
from PoseClassification.visualize import PoseClassificationVisualizer
from PoseClassification.bootstrap import BootstrapHelper
from mediapipe.python.solutions import pose as mp_pose
from mediapipe.python.solutions import drawing_utils as mp_drawing
import cv2
import tqdm
import os

# Pose class to count repetitions for
class_name = 'pushups_down'
out_video_path = 'pushups-sample-out.mp4'

# Open webcam or video file
video_cap = cv2.VideoCapture(0)

# Video parameters
video_n_frames = video_cap.get(cv2.CAP_PROP_FRAME_COUNT)
video_fps = video_cap.get(cv2.CAP_PROP_FPS) or 25
video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

print(f"video_n_frames: {video_n_frames}")
print(f"video_fps: {video_fps}")
print(f"video_width: {video_width}")
print(f"video_height: {video_height}")

# Pose samples folder
pose_samples_folder = 'fitness_poses_csvs_out'

# Initialize components
pose_tracker = mp_pose.Pose()
pose_embedder = FullBodyPoseEmbedding()
pose_classifier = PoseClassifier(
    pose_samples_folder=pose_samples_folder,
    pose_embedder=pose_embedder,
    top_n_by_max_distance=30,
    top_n_by_mean_distance=10)

pose_classification_filter = EMADictSmoothing(window_size=10, alpha=0.2)
repetition_counter = RepetitionCounter(class_name=class_name, enter_threshold=6, exit_threshold=4)
pose_classification_visualizer = PoseClassificationVisualizer(
    class_name=class_name,
    plot_x_max=video_n_frames,
    plot_y_max=10)

# Output video writer
out_video = cv2.VideoWriter(out_video_path, cv2.VideoWriter_fourcc(*'mp4v'), video_fps, (video_width, video_height))

# --- Real-time plotting setup ---
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6))
line1, = ax1.plot([], [], label='Right Wrist Y')
line2, = ax2.plot([], [], label='Velocity (dy/dt)', color='orange')
ax1.set_ylabel('Y Position')
ax2.set_ylabel('Velocity')
ax2.set_xlabel('Time (s)')
ax1.legend()
ax2.legend()

# Buffers
plot_window = 100
right_wrist_y = deque(maxlen=plot_window)
right_wrist_dy = deque(maxlen=plot_window)
frame_times = deque(maxlen=plot_window)

frame_idx = 0
output_frame = None

with tqdm.tqdm(total=video_n_frames, position=0, leave=True) as pbar:
    while True:
        success, input_frame = video_cap.read()
        if not success:
            print("Unable to read input video frame, breaking!")
            break

        # Pose tracking
        input_frame_rgb = cv2.cvtColor(input_frame, cv2.COLOR_BGR2RGB)
        result = pose_tracker.process(image=input_frame_rgb)
        pose_landmarks = result.pose_landmarks

        # Visualization frame
        output_frame = input_frame_rgb.copy()

        if pose_landmarks is not None:
            mp_drawing.draw_landmarks(
                image=output_frame,
                landmark_list=pose_landmarks,
                connections=mp_pose.POSE_CONNECTIONS)

            # Convert pose landmarks
            frame_height, frame_width = output_frame.shape[0], output_frame.shape[1]
            pose_landmarks = np.array([
                [lmk.x * frame_width, lmk.y * frame_height, lmk.z * frame_width]
                for lmk in pose_landmarks.landmark
            ], dtype=np.float32)

            assert pose_landmarks.shape == (33, 3), 'Unexpected landmarks shape: {}'.format(pose_landmarks.shape)

            # Pose classification
            pose_classification = pose_classifier(pose_landmarks)
            pose_classification_filtered = pose_classification_filter(pose_classification)
            repetitions_count = repetition_counter(pose_classification_filtered)

            # --- Track landmark for plotting ---
            right_wrist_y_val = pose_landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value][1]
            time_sec = frame_idx / video_fps

            right_wrist_y.append(right_wrist_y_val)
            frame_times.append(time_sec)

            step = 15
            if len(right_wrist_y) > step +5:
                dy = (right_wrist_y[-1] - right_wrist_y[-2-step]) / (frame_times[-1] - frame_times[-2-step])
            else:
                dy = 0
            right_wrist_dy.append(dy)

        else:
            # Handle case with no pose
            pose_classification = None
            pose_classification_filtered = pose_classification_filter(dict())
            pose_classification_filtered = None
            repetitions_count = repetition_counter.n_repeats

        # Draw classification and counter
        output_frame = pose_classification_visualizer(
            frame=output_frame,
            pose_classification=pose_classification,
            pose_classification_filtered=pose_classification_filtered,
            repetitions_count=repetitions_count)

        # Save and display output frame
        out_video.write(cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))
        cv2.imshow('Pose Classification', cv2.cvtColor(np.array(output_frame), cv2.COLOR_RGB2BGR))

        # --- Update plot ---
        line1.set_xdata(frame_times)
        line1.set_ydata(right_wrist_y)
        ax1.relim()
        ax1.autoscale_view()

        line2.set_xdata(frame_times)
        line2.set_ydata(right_wrist_dy)
        ax2.relim()
        ax2.autoscale_view()

        plt.pause(0.001)
        plt.draw()

        frame_idx += 1

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

        pbar.update()

# Cleanup
out_video.release()
video_cap.release()
pose_tracker.close()
cv2.destroyAllWindows()
plt.ioff()
plt.show()
