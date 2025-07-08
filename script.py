import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from collections import deque
import platform

# Sound alert setup
try:
    if platform.system() == "Windows":
        import winsound
        def play_sound():
            winsound.Beep(1000, 500)  # frequency 1000 Hz, duration 500 ms
    else:
        from playsound import playsound
        import threading

        def play_sound():
            # Play sound asynchronously to avoid blocking main loop
            threading.Thread(target=playsound, args=('alert.mp3',), daemon=True).start()

except ImportError:
    def play_sound():
        print("Sound module not available")

# Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5)

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

ret, frame = cap.read()
if not ret:
    print("Error: Could not read frame from webcam.")
    exit()

height, width, _ = frame.shape

# Buffers
buf_size = 100
hipYs = deque(maxlen=buf_size)
velocities = deque(maxlen=buf_size)
times = deque(maxlen=buf_size)

# Fall detection parameters (more sensitive)
fall_velocity_threshold = 5       # Lower threshold for velocity (pixels/frame)
fall_acceleration_threshold = 10  # Acceleration threshold
fall_frame_limit = 2               # Frames to confirm fall
fall_counter = 0
fall_alert_triggered = False  # To control sound playback once per fall

# Set up Matplotlib
plt.ion()
fig, (ax_img, ax1, ax2) = plt.subplots(3, 1, figsize=(8, 10))

# Video display
img_plot = ax_img.imshow(np.zeros((height, width, 3), dtype=np.uint8))
ax_img.axis('off')
ax_img.set_title("Pose Landmarks")

# Hip Y and velocity plots
l1, = ax1.plot([], [], label='Hip Y')
l2, = ax2.plot([], [], label='Velocity', color='r')
ax1.invert_yaxis()
ax1.set_ylabel('Hip Y (px)')
ax2.set_ylabel('Velocity')
ax2.set_xlabel('Frame')
ax1.legend()
ax2.legend()

frame_idx = 0

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(image)
        output = frame.copy()

        hip_y = None

        if results.pose_landmarks:
            for landmark in results.pose_landmarks.landmark:
                cx, cy = int(landmark.x * width), int(landmark.y * height)
                cv2.circle(output, (cx, cy), 5, (0, 255, 0), -1)

            hip_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
            #hip_landmark = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP]
            hip_y = hip_landmark.y * height

        # Update data buffers and detect fall
        if hip_y is not None:
            hipYs.append(hip_y)
            times.append(frame_idx)

            if len(hipYs) > 1:
                velocity = hipYs[-1] - hipYs[-2]
                velocities.append(velocity)

                acceleration = 0
                if len(velocities) > 1:
                    acceleration = velocities[-1] - velocities[-2]

                # Fall detection logic (more sensitive)
                if velocity > fall_velocity_threshold and acceleration > fall_acceleration_threshold:
                    fall_counter += 1
                else:
                    fall_counter = max(0, fall_counter - 1)
            else:
                velocities.append(0)
                fall_counter = 0
        else:
            # No detection, keep previous values or zero
            hipYs.append(hipYs[-1] if hipYs else 0)
            velocities.append(0)
            times.append(frame_idx)
            fall_counter = max(0, fall_counter - 1)

        frame_idx += 1

        # Visual and audio alert if fall detected
        if fall_counter >= fall_frame_limit:
            # Dramatic red flashing overlay
            alpha = 0.6 + 0.4 * np.sin(frame_idx * 0.3)  # oscillate transparency
            overlay = output.copy()
            cv2.rectangle(overlay, (0, 0), (width, height), (0, 0, 255), -1)
            cv2.addWeighted(overlay, alpha, output, 1 - alpha, 0, output)

            cv2.putText(output, "FALL DETECTED!", (50, int(height/2)), cv2.FONT_HERSHEY_SIMPLEX,
                        3, (255, 255, 255), 8, cv2.LINE_AA)

            # Play sound once per fall event
            if not fall_alert_triggered:
                play_sound()
                fall_alert_triggered = True
        else:
            fall_alert_triggered = False

        # Update plots
        rgb_frame = cv2.cvtColor(output, cv2.COLOR_BGR2RGB)
        img_plot.set_data(rgb_frame)

        l1.set_data(times, hipYs)
        l2.set_data(times, velocities)

        ax1.set_xlim(max(0, frame_idx - buf_size), frame_idx + 1)
        ax2.set_xlim(ax1.get_xlim())

        if hipYs:
            ax1.set_ylim(min(hipYs), max(hipYs))
        if velocities:
            ax2.set_ylim(min(velocities), max(velocities))

        plt.pause(0.001)

except KeyboardInterrupt:
    print("Interrupted.")

finally:
    cap.release()
    plt.ioff()
    plt.show()
    cv2.destroyAllWindows()
