import cv2
import mediapipe as mp
import time

cap = cv2.VideoCapture(0)
p_time = 0

mp_face_detection = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.75)


while True:
    success, frame = cap.read()
    # flip the frame, as video is being displayed upside down
    frame = cv2.flip(frame, 1)

    # convert the frame from BGR to RGB
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_detection.process(rgb_frame)
    # print(results)

    if results.detections:
        for id, detection in enumerate(results.detections):
            relative_bounding_box = detection.location_data.relative_bounding_box
            fw, fh, fc = frame.shape
            rect_start_point = mp_draw._normalized_to_pixel_coordinates(
                relative_bounding_box.xmin,
                relative_bounding_box.ymin,
                fh,
                fw,
            )
            rect_end_point = mp_draw._normalized_to_pixel_coordinates(
                relative_bounding_box.xmin + relative_bounding_box.width,
                relative_bounding_box.ymin + relative_bounding_box.height,
                fh,
                fw,
            )
            cv2.rectangle(
                frame,
                rect_start_point,
                rect_end_point,
                (0, 0, 255),
                2,
            )
            cv2.putText(
                frame,
                f"{int(detection.score[0] * 100)}%",
                (rect_start_point[0], rect_start_point[1] - 20),
                cv2.FONT_HERSHEY_PLAIN,
                3,
                (0, 255, 0),
                2,
            )
            # mp_draw.draw_detection(frame, detection)
            # print(id, detection)

    # calculate fps
    c_time = time.time()
    if c_time - p_time > 0:
        fps = 1 / (c_time - p_time)
    else:
        fps = 0
    p_time = c_time

    # display fps on frame
    cv2.putText(
        frame,
        f"FPS: {int(fps)}",
        (0, 30),
        cv2.FONT_HERSHEY_PLAIN,
        3,
        (0, 255, 0),
        2,
    )

    cv2.imshow("Frame", frame)
    cv2.waitKey(1)
