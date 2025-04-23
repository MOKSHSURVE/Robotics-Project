import cv2
import numpy as np

# Function for lane detection
def region_of_interest(img, vertices):
    mask = np.zeros_like(img)
    cv2.fillPoly(mask, vertices, 255)
    masked_img = cv2.bitwise_and(img, mask)
    return masked_img

def draw_lines(img, lines, color=(0, 255, 0), thickness=3):
    for line in lines:
        for x1, y1, x2, y2 in line:
            cv2.line(img, (x1, y1), (x2, y2), color, thickness)

def process_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    
    height, width = edges.shape
    roi_vertices = [(0, height), (width / 2, height / 2), (width, height)]
    roi_edges = region_of_interest(edges, np.array([roi_vertices], np.int32))
    
    lines = cv2.HoughLinesP(roi_edges, 1, np.pi / 180, 50, minLineLength=100, maxLineGap=150)
    if lines is not None:
        draw_lines(frame, lines)
    
    return frame

# Function for detecting obstacles (simple contour detection)
def detect_obstacles(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    _, thresholded = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresholded, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    obstacles = []
    for contour in contours:
        if cv2.contourArea(contour) > 500:  # Filter out small objects
            obstacles.append(contour)
    
    return obstacles

def draw_obstacles(frame, obstacles):
    for obstacle in obstacles:
        x, y, w, h = cv2.boundingRect(obstacle)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 3)
    return frame

# Start video capture (0 for webcam, or provide a video file path)
cap = cv2.VideoCapture(0)  # Use 0 for webcam, replace with a video path if needed

if not cap.isOpened():
    print("Error: Could not open video or webcam.")
    exit()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Process the frame for lane detection
    lane_frame = process_frame(frame)
    
    # Detect obstacles
    obstacles = detect_obstacles(frame)
    
    # Draw obstacles on the lane detection frame
    obstacle_frame = draw_obstacles(lane_frame, obstacles)
    
    # Show the final output with both lane and obstacle information
    cv2.imshow('Self-Driving Car Simulation', obstacle_frame)
    
    # Exit the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()
