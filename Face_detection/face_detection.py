import cv2
import numpy as np
import mediapipe as mp
import matplotlib.pyplot as plt

mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def process_image(image_path):
    sample_img = cv2.imread(image_path)
    frame_rgb = cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB)
    
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)
    face_detection_results = face_detection.process(frame_rgb)

    if face_detection_results.detections:
        for face in face_detection_results.detections:
            mp_drawing.draw_detection(image=sample_img, detection=face,
                                       keypoint_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0),
                                                                                    thickness=2,
                                                                                    circle_radius=2))
    
    plt.figure(figsize=[10, 10])
    plt.title("Detected Faces in Image")
    plt.axis('off')
    plt.imshow(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
    plt.show()

def real_time_detection():
    cap = cv2.VideoCapture(0)
    face_detection = mp_face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_detection_results = face_detection.process(frame_rgb)

        if face_detection_results.detections:
            for face in face_detection_results.detections:
                mp_drawing.draw_detection(image=frame, detection=face,
                                           keypoint_drawing_spec=mp_drawing.DrawingSpec(color=(255, 0, 0),
                                                                                        thickness=2,
                                                                                        circle_radius=2))

        cv2.imshow('Real-time Face Detection', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def main():
    choice = input("Choose detection mode (1 for Image, 2 for Real-time): ")
    if choice == '1':
        image_path = input("Enter the path of the image file: ")
        process_image(image_path)
    elif choice == '2':
        real_time_detection()
    else:
        print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main()
