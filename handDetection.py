import cv2
import numpy as np

cap = cv2.VideoCapture(0)  # Open the default camera

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Convert the frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the lower and upper range of skin color
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)

    # Apply skin color segmentation
    mask = cv2.inRange(hsv, lower_skin, upper_skin)

    # Refine the mask
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mask = cv2.erode(mask, kernel, iterations=2)
    mask = cv2.dilate(mask, kernel, iterations=2)

    # Find hand contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    hand_contours = []

    for contour in contours:
        # Filter contours based on area and shape if needed
        # ...

        hand_contours.append(contour)

    # Draw hand contours on the original frame
    cv2.drawContours(frame, hand_contours, -1, (0, 255, 0), 2)

    # Display the processed frame
    cv2.imshow('Hand Detection', frame)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release resources
cap.release()
cv2.destroyAllWindows()
