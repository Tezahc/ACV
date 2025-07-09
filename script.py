import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import platform
import threading
import matplotlib.gridspec as gridspec
import argparse
from src import args_validation
import record

args = args_validation.args_validation(argparse.ArgumentParser())
input_video_file = args.input_video_file
out_video_path = args.output_video_file

# Sound alert setup
try:
    if platform.system() == "Windows":
        import winsound
        def play_sound():
            winsound.Beep(1000, 700)
    else:
        from playsound import playsound
        def play_sound():
            threading.Thread(target=playsound, args=('alert.mp3',), daemon=True).start()
except ImportError:
    def play_sound():
        print("Sound module not available")

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

video_cap = cv2.VideoCapture(input_video_file)
if not video_cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

video_fps = video_cap.get(cv2.CAP_PROP_FPS)
video_width = int(video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
video_height = int(video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

ret, frame = video_cap.read()
if not ret:
    print("Error: Could not read frame from webcam.")
    exit()

buf_size = 150
times = deque(maxlen=buf_size)

# Raw Y positions
nose_raw = deque(maxlen=buf_size)
left_hip_raw = deque(maxlen=buf_size)

# Filtered Y positions
nose_filt = deque(maxlen=buf_size)
left_hip_filt = deque(maxlen=buf_size)

# Filtered velocities
nose_vel_filt = deque(maxlen=buf_size)
left_hip_vel_filt = deque(maxlen=buf_size)

# Conditions (0 or 1)
pos_cond_nose = deque(maxlen=buf_size)
pos_cond_lhip = deque(maxlen=buf_size)
vel_cond_nose = deque(maxlen=buf_size)
vel_cond_lhip = deque(maxlen=buf_size)
fall_cond = deque(maxlen=buf_size)

# EMA smoothing factor
alpha = 0.3

# Thresholds
fall_threshold_lhip = video_height * 0.75
fall_threshold_nose = video_height * 0.6
velocity_threshold = 7.0
fall_frame_limit = 3
fall_counter = 0
fall_alert_triggered = False

min_valid_y = video_height * 0.2
max_valid_y = video_height * 0.95

plt.ion()

# Use gridspec for better layout
fig = plt.figure(figsize=(14, 8))

# Main gridspec: 1 row, 2 columns (video + plots)
gs_main = gridspec.GridSpec(1, 2, width_ratios=[4, 3], wspace=0.3)

# Video feed axis (left)
ax_video = fig.add_subplot(gs_main[0])
img_plot = ax_video.imshow(np.zeros((video_height, video_width, 3), dtype=np.uint8))
ax_video.axis('off')
ax_video.set_title("Pose Landmarks")

# Nested gridspec for 7 stacked plots (right)
gs_plots = gridspec.GridSpecFromSubplotSpec(7, 1, subplot_spec=gs_main[1], hspace=0.5)

axs_plots = [fig.add_subplot(gs_plots[i]) for i in range(7)]

# Nose position
l_nose_pos, = axs_plots[0].plot([], [], label='Nose Y (filtered)', color='m')
axs_plots[0].axhline(fall_threshold_nose, color='r', linestyle='--', label='Nose Fall Threshold')
axs_plots[0].invert_yaxis()
axs_plots[0].set_ylabel('Pixels')
axs_plots[0].legend()
axs_plots[0].set_title('Nose Y Position')

# Left hip position
l_lhip_pos, = axs_plots[1].plot([], [], label='Left Hip Y (filtered)', color='b')
axs_plots[1].axhline(fall_threshold_lhip, color='r', linestyle='--', label='Left Hip Fall Threshold')
axs_plots[1].invert_yaxis()
axs_plots[1].set_ylabel('Pixels')
axs_plots[1].legend()
axs_plots[1].set_title('Left Hip Y Position')

# Nose velocity
l_nose_vel, = axs_plots[2].plot([], [], label='Nose Velocity', color='c')
axs_plots[2].axhline(velocity_threshold, color='r', linestyle='--', label='Velocity Threshold')
axs_plots[2].set_ylabel('Pixels/frame')
axs_plots[2].legend()
axs_plots[2].set_title('Nose Velocity')

# Left hip velocity
l_lhip_vel, = axs_plots[3].plot([], [], label='Left Hip Velocity', color='g')
axs_plots[3].axhline(velocity_threshold, color='r', linestyle='--', label='Velocity Threshold')
axs_plots[3].set_ylabel('Pixels/frame')
axs_plots[3].legend()
axs_plots[3].set_title('Left Hip Velocity')

# Position and velocity conditions
l_pos_cond_nose, = axs_plots[4].step([], [], label='Nose Pos > Threshold', color='m')
l_pos_cond_lhip, = axs_plots[4].step([], [], label='Left Hip Pos > Threshold', color='b')
l_vel_cond_nose, = axs_plots[4].step([], [], label='Nose Vel > Threshold', color='c')
l_vel_cond_lhip, = axs_plots[4].step([], [], label='Left Hip Vel > Threshold', color='g')
axs_plots[4].set_ylabel('Bool (0/1)')
axs_plots[4].legend(loc='upper right')
axs_plots[4].set_title('Position and Velocity Conditions')

# Fall detection condition
l_fall_cond, = axs_plots[5].step([], [], label='Fall Condition', color='r')
axs_plots[5].set_ylabel('Bool (0/1)')
axs_plots[5].legend()
axs_plots[5].set_title('Fall Detection Condition')
axs_plots[5].set_xlabel('Frame')

# Empty last plot or extra info
axs_plots[6].axis('off')

# Fix axis limits on plots
for ax in axs_plots[:6]:
    ax.set_autoscale_on(False)
    ax.set_xlim(0, buf_size)
axs_plots[0].set_ylim(video_height, 0)
axs_plots[1].set_ylim(video_height, 0)
axs_plots[2].set_ylim(-1, max(20, velocity_threshold + 5))
axs_plots[3].set_ylim(-1, max(20, velocity_threshold + 5))
axs_plots[4].set_ylim(-0.1, 1.1)
axs_plots[5].set_ylim(-0.1, 1.1)

frame_idx = 0

def ema_filter(prev, new, alpha):
    if prev is None:
        return new
    return alpha * new + (1 - alpha) * prev

def compute_velocity(data_deque):
    if len(data_deque) < 2:
        return 0.0
    return data_deque[-1] - data_deque[-2]

# check si un output a été renseigné
record_enabled = bool(out_video_path)
video_recorder = record.VideoRecorder(
    output_path = out_video_path,
    fps = video_fps,
    frame_size = (int(video_width), int(video_height)),
    enabled = record_enabled
)

try:
    while True:
        ret, frame = video_cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        output = frame.copy()

        nose_y = None
        left_hip_y = None

        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            for landmark in lm:
                cx, cy = int(landmark.x * video_width), int(landmark.y * video_height)
                cv2.circle(output, (cx, cy), 5, (0,255,0), -1)

            nose_y = lm[mp_pose.PoseLandmark.NOSE].y * video_height
            left_hip_y = lm[mp_pose.PoseLandmark.LEFT_HIP].y * video_height

        nose_raw.append(nose_y if nose_y is not None else (nose_raw[-1] if nose_raw else video_height))
        left_hip_raw.append(left_hip_y if left_hip_y is not None else (left_hip_raw[-1] if left_hip_raw else video_height))

        # Filter positions
        nose_filt_val = ema_filter(nose_filt[-1] if nose_filt else None, nose_raw[-1], alpha)
        left_hip_filt_val = ema_filter(left_hip_filt[-1] if left_hip_filt else None, left_hip_raw[-1], alpha)

        nose_filt.append(nose_filt_val)
        left_hip_filt.append(left_hip_filt_val)

        # Compute velocities
        nose_vel = compute_velocity(nose_filt)
        left_hip_vel = compute_velocity(left_hip_filt)

        nose_vel_filt.append(nose_vel)
        left_hip_vel_filt.append(left_hip_vel)

        times.append(frame_idx)
        frame_idx += 1

        # Valid detection check
        valid_detection = all(min_valid_y < y < max_valid_y for y in [nose_filt_val, left_hip_filt_val])

        # Conditions
        pos_nose_cond = int(nose_filt_val > fall_threshold_nose)
        pos_lhip_cond = int(left_hip_filt_val > fall_threshold_lhip)
        vel_nose_cond = int(nose_vel > velocity_threshold)
        vel_lhip_cond = int(left_hip_vel > velocity_threshold)

        pos_cond_nose.append(pos_nose_cond)
        pos_cond_lhip.append(pos_lhip_cond)
        vel_cond_nose.append(vel_nose_cond)
        vel_cond_lhip.append(vel_lhip_cond)

        # Fall detection logic: fall if position and velocity condition for either nose or left hip
        fall_detected_cond = valid_detection and (
            (pos_nose_cond and vel_nose_cond) or
            (pos_lhip_cond and vel_lhip_cond)
        )

        if fall_detected_cond:
            fall_counter += 1
        else:
            fall_counter = max(0, fall_counter - 1)

        fall_cond_flag = int(fall_counter >= fall_frame_limit)
        fall_cond.append(fall_cond_flag)

        # Visual + sound alert
        if fall_cond_flag:
            alpha_overlay = 0.6 + 0.4 * np.sin(frame_idx * 0.3)
            overlay = output.copy()
            cv2.rectangle(overlay, (0, 0), (video_width, video_height), (0, 0, 255), -1)
            cv2.addWeighted(overlay, alpha_overlay, output, 1 - alpha_overlay, 0, output)
            cv2.putText(output, "FALL DETECTED!", (50, int(video_height / 2)), cv2.FONT_HERSHEY_SIMPLEX,
                        3, (255, 255, 255), 8, cv2.LINE_AA)
            if not fall_alert_triggered:
                play_sound()
                fall_alert_triggered = True
        else:
            fall_alert_triggered = False

        rgb_frame = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        img_plot.set_data(rgb_frame)
        video_recorder.write(output) # output car en BGR pour cv2

        # Update plots with fixed x-axis range using range(len(...))
        l_nose_pos.set_data(range(len(nose_filt)), nose_filt)
        l_lhip_pos.set_data(range(len(left_hip_filt)), left_hip_filt)
        l_nose_vel.set_data(range(len(nose_vel_filt)), nose_vel_filt)
        l_lhip_vel.set_data(range(len(left_hip_vel_filt)), left_hip_vel_filt)
        l_pos_cond_nose.set_data(range(len(pos_cond_nose)), pos_cond_nose)
        l_pos_cond_lhip.set_data(range(len(pos_cond_lhip)), pos_cond_lhip)
        l_vel_cond_nose.set_data(range(len(vel_cond_nose)), vel_cond_nose)
        l_vel_cond_lhip.set_data(range(len(vel_cond_lhip)), vel_cond_lhip)
        l_fall_cond.set_data(range(len(fall_cond)), fall_cond)

        plt.pause(0.001)

except KeyboardInterrupt:
    print("Interrupted by user")

finally:
    video_cap.release()
    plt.ioff()
    plt.show()
    video_recorder.release()
    cv2.destroyAllWindows()
