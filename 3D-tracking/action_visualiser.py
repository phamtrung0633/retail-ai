import cv2 as cv
import json
import os
# Constants
FRAMERATE = 15
RESOLUTION = (1920, 1080)
TIME_LENGTH_SHOW = 150

fourcc = cv.VideoWriter_fourcc(*'MJPG')
video_writers = [cv.VideoWriter('videos/demo_video_l0.avi', fourcc, FRAMERATE, RESOLUTION),
                cv.VideoWriter('videos/demo_video_r0.avi', fourcc, FRAMERATE, RESOLUTION)]
output_dirs = ["frames_data_cam_0", "frames_data_cam_1"]
with open('interaction_data.json') as file:
    action_data = json.load(file)

for i, output_dir in enumerate(output_dirs):
    action_history = []
    action_count = 0
    for index, file in enumerate(sorted(os.listdir(output_dir), key = lambda s: float(s[:-4]))):
        file_path = os.path.join(output_dir, file)
        image = cv.imread(file_path)
        timestamp = file[:-4]
        if timestamp in action_data[str(i)]:
            action_data_this_timestamp = action_data[str(i)][timestamp]
            action_data_with_count = []
            for action in action_data_this_timestamp:
                action_count += 1
                action_data_with_count.append([action_count, action])
            action_history.append([index, action_data_with_count])

        action_amount = 0
        for action in action_history:
            action_amount += len(action[1])
        if len(action_history) != 0:
            cv.rectangle(image, (1450, 0), (1920, action_amount * 40), (245, 117, 16), -1)
        start_y = 20
        action_history_dup = action_history.copy()
        action_amount = 0
        for j, (index_created, actions) in enumerate(action_history_dup):
            for count, action in actions:
                cv.putText(image, f"Event {count}: {action}", (1465, start_y), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv.LINE_AA)
                start_y += 40
                action_amount += 1
            if index - index_created > TIME_LENGTH_SHOW:
                del action_history[j]

        video_writers[i].write(image)

for recorder in video_writers:
    recorder.release()

