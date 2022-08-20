import cv2
import mediapipe as mp
import time


class FaceDetector:
    def __init__(self, min_detection_confidence=0.5, model_selection=0):
        self.min_detection_confidence = min_detection_confidence
        self.model_selection = model_selection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_draw = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            self.min_detection_confidence, self.model_selection
        )

    def fancy_draw(sef, frame, rect_start_point, rect_end_point, l=20, t=2, rt=1):
        x, y, w, h = (
            rect_start_point[0],
            rect_start_point[1],
            rect_end_point[0] - rect_start_point[0],
            rect_end_point[1] - rect_start_point[1],
        )
        x1, y1 = x + w, y + h
        cv2.rectangle(
            frame,
            rect_start_point,
            rect_end_point,
            (0, 0, 255),
            rt,
        )
        # Top left
        cv2.line(frame, (x, y), (x + l, y), (0, 0, 255), t)
        cv2.line(frame, (x, y), (x, y + l), (0, 0, 255), t)
        # Top right
        cv2.line(frame, (x1, y), (x1 - l, y), (0, 0, 255), t)
        cv2.line(frame, (x1, y), (x1, y + l), (0, 0, 255), t)
        # Bottom left
        cv2.line(frame, (x, y1), (x + l, y1), (0, 0, 255), t)
        cv2.line(frame, (x, y1), (x, y1 - l), (0, 0, 255), t)
        # Bottom right
        cv2.line(frame, (x1, y1), (x1 - l, y1), (0, 0, 255), t)
        cv2.line(frame, (x1, y1), (x1, y1 - l), (0, 0, 255), t)

        return frame

    def find_faces(self, frame, draw=True):
        img_RGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.results = self.face_detection.process(img_RGB)
        bboxs = []

        if self.results.detections:
            for id, detection in enumerate(self.results.detections):
                relative_bounding_box = detection.location_data.relative_bounding_box
                fw, fh, fc = frame.shape
                rect_start_point = self.mp_draw._normalized_to_pixel_coordinates(
                    relative_bounding_box.xmin,
                    relative_bounding_box.ymin,
                    fh,
                    fw,
                )
                rect_end_point = self.mp_draw._normalized_to_pixel_coordinates(
                    relative_bounding_box.xmin + relative_bounding_box.width,
                    relative_bounding_box.ymin + relative_bounding_box.height,
                    fh,
                    fw,
                )
                bbox = (
                    rect_start_point[0],
                    rect_start_point[1],
                    rect_end_point[0],
                    rect_end_point[1],
                )
                bboxs.append([id, bbox, detection.score])
                if draw:
                    frame = self.fancy_draw(frame, rect_start_point, rect_end_point)
                    cv2.putText(
                        frame,
                        f"{int(detection.score[0] * 100)}%",
                        (rect_start_point[0], rect_start_point[1] - 20),
                        cv2.FONT_HERSHEY_PLAIN,
                        3,
                        (0, 255, 0),
                        2,
                    )
        return frame, bboxs


def main():
    cap = cv2.VideoCapture(0)
    p_time = 0
    detector = FaceDetector()

    while True:
        success, frame = cap.read()
        # flip the frame, as video needs to be displayed like mirror image
        frame = cv2.flip(frame, 1)
        c_time = time.time()
        frame, bboxs = detector.find_faces(frame)

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


if __name__ == "__main__":
    main()
