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

# Argument parsing
parser = argparse.ArgumentParser()
args = args_validation.args_validation(parser)

input_video_file = args.input_video_file
out_video_path = args.output_video_file
show_plots = args.show_plots

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

# Pose detection setup
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Open video file or webcam
cap = cv2.VideoCapture(input_video_file)
if not cap.isOpened():
    print("Error: Could not open video source.")
    exit()

ret, frame = cap.read()
if not ret:
    print("Error: Could not read initial frame.")
    exit()

height, width, _ = frame.shape

# Buffers
buf_size = 50
times = deque(maxlen=buf_size)
nose_raw = deque(maxlen=buf_size)
left_hip_raw = deque(maxlen=buf_size)
nose_filt = deque(maxlen=buf_size)
left_hip_filt = deque(maxlen=buf_size)
nose_vel_filt = deque(maxlen=buf_size)
left_hip_vel_filt = deque(maxlen=buf_size)
pos_cond_nose = deque(maxlen=buf_size)
pos_cond_lhip = deque(maxlen=buf_size)
vel_cond_nose = deque(maxlen=buf_size)
vel_cond_lhip = deque(maxlen=buf_size)
fall_cond = deque(maxlen=buf_size)

# Constants
alpha = 0.3
fall_threshold_lhip = height * 0.7
fall_threshold_nose = height * 0.6
velocity_threshold = 7.0
fall_frame_limit = 3
fall_counter = 0
fall_alert_triggered = False
min_valid_y = height * 0.3
max_valid_y = height * 0.95

# Plot setup (only if enabled)
if show_plots:
    plt.ion()
    fig = plt.figure(figsize=(14, 8))
    gs_main = gridspec.GridSpec(1, 2, width_ratios=[4, 3], wspace=0.3)
    ax_video = fig.add_subplot(gs_main[0])
    img_plot = ax_video.imshow(np.zeros((height, width, 3), dtype=np.uint8))
    ax_video.axis('off')
    ax_video.set_title("Pose Landmarks")

    gs_plots = gridspec.GridSpecFromSubplotSpec(7, 1, subplot_spec=gs_main[1], hspace=0.5)
    axs_plots = [fig.add_subplot(gs_plots[i]) for i in range(7)]

    l_nose_pos, = axs_plots[0].plot([], [], label='Nose Y (filtered)', color='m')
    axs_plots[0].axhline(fall_threshold_nose, color='r', linestyle='--')
    axs_plots[0].invert_yaxis()
    axs_plots[0].set_title("Nose Y Position")

    l_lhip_pos, = axs_plots[1].plot([], [], label='Left Hip Y (filtered)', color='b')
    axs_plots[1].axhline(fall_threshold_lhip, color='r', linestyle='--')
    axs_plots[1].invert_yaxis()
    axs_plots[1].set_title("Left Hip Y Position")

    l_nose_vel, = axs_plots[2].plot([], [], label='Nose Velocity', color='c')
    axs_plots[2].axhline(velocity_threshold, color='r', linestyle='--')
    axs_plots[2].set_title("Nose Velocity")

    l_lhip_vel, = axs_plots[3].plot([], [], label='Left Hip Velocity', color='g')
    axs_plots[3].axhline(velocity_threshold, color='r', linestyle='--')
    axs_plots[3].set_title("Left Hip Velocity")

    l_pos_cond_nose, = axs_plots[4].step([], [], label='Nose Pos > Threshold', color='m')
    l_pos_cond_lhip, = axs_plots[4].step([], [], label='LHip Pos > Threshold', color='b')
    l_vel_cond_nose, = axs_plots[4].step([], [], label='Nose Vel > Threshold', color='c')
    l_vel_cond_lhip, = axs_plots[4].step([], [], label='LHip Vel > Threshold', color='g')
    axs_plots[4].set_title("Position/Velocity Conditions")

    l_fall_cond, = axs_plots[5].step([], [], label='Fall Condition', color='r')
    axs_plots[5].set_title("Fall Detection")
    axs_plots[6].axis('off')

    for ax in axs_plots[:6]:
        ax.set_xlim(0, buf_size)
    axs_plots[0].set_ylim(height, 0)
    axs_plots[1].set_ylim(height, 0)
    axs_plots[2].set_ylim(-1, 20)
    axs_plots[3].set_ylim(-1, 20)
    axs_plots[4].set_ylim(-0.1, 1.1)
    axs_plots[5].set_ylim(-0.1, 1.1)

frame_idx = 0

def ema_filter(prev, new, alpha):
    return new if prev is None else alpha * new + (1 - alpha) * prev

def compute_velocity(deq):
    return deq[-1] - deq[-2] if len(deq) >= 2 else 0

# Main loop
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        output = frame.copy()

        nose_y, left_hip_y = None, None
        if results.pose_landmarks:
            lm = results.pose_landmarks.landmark
            for landmark in lm:
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                cv2.circle(output, (cx, cy), 5, (0, 255, 0), -1)

            nose_y = lm[mp_pose.PoseLandmark.NOSE].y * height
            left_hip_y = lm[mp_pose.PoseLandmark.LEFT_HIP].y * height

        # Fill raw and filtered data
        nose_raw.append(nose_y if nose_y else nose_raw[-1] if nose_raw else height)
        left_hip_raw.append(left_hip_y if left_hip_y else left_hip_raw[-1] if left_hip_raw else height)

        nose_filt_val = ema_filter(nose_filt[-1] if nose_filt else None, nose_raw[-1], alpha)
        left_hip_filt_val = ema_filter(left_hip_filt[-1] if left_hip_filt else None, left_hip_raw[-1], alpha)

        nose_filt.append(nose_filt_val)
        left_hip_filt.append(left_hip_filt_val)

        nose_vel = compute_velocity(nose_filt)
        left_hip_vel = compute_velocity(left_hip_filt)
        nose_vel_filt.append(nose_vel)
        left_hip_vel_filt.append(left_hip_vel)

        # Detection conditions
        valid = min_valid_y < nose_filt_val < max_valid_y and min_valid_y < left_hip_filt_val < max_valid_y
        pos_n = int(nose_filt_val > fall_threshold_nose)
        pos_l = int(left_hip_filt_val > fall_threshold_lhip)
        vel_n = int(nose_vel > velocity_threshold)
        vel_l = int(left_hip_vel > velocity_threshold)

        pos_cond_nose.append(pos_n)
        pos_cond_lhip.append(pos_l)
        vel_cond_nose.append(vel_n)
        vel_cond_lhip.append(vel_l)

        fall_cond_now = valid and ((pos_n and vel_n) or (pos_l and vel_l))
        fall_counter = fall_counter + 1 if fall_cond_now else max(0, fall_counter - 1)
        fall_flag = int(fall_counter >= fall_frame_limit)
        fall_cond.append(fall_flag)

        if fall_flag:
            overlay = output.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
            cv2.addWeighted(overlay, 0.4, output, 0.6, 0, output)
            cv2.putText(output, "FALL DETECTED!", (50, height // 2), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 4)
            if not fall_alert_triggered:
                play_sound()
                fall_alert_triggered = True
        else:
            fall_alert_triggered = False

        # === Display ===
        if show_plots:
            img_plot.set_data(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
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
        else:
            cv2.imshow("Fall Detection", output)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        frame_idx += 1

except KeyboardInterrupt:
    print("Interrupted")

finally:
    cap.release()
    if show_plots:
        plt.ioff()
        plt.show()
    else:
        cv2.destroyAllWindows()
