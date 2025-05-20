import cv2
import os

def capture_images(name):
    cap = cv2.VideoCapture(0)
    os.makedirs(f"data/{name}", exist_ok=True)
    count = 0

    while count < 50:  # Capture 50 images
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imshow("Capturing", frame)
        cv2.imwrite(f"data/{name}/{count}.jpg", frame)
        count += 1
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

capture_images("film_star_name")
