# Import libraries
import math     # For math functions
import time     # For timer to calculate fps
import numpy as np  # For array operations
import cv2      # For video/image processing

import matplotlib.pyplot as plt

def denoise_frame(frame):
    """ Function for denoising 
    image with Gaussian Blur """   
    
    kernel = np.ones((3, 3), np.float32) / 9   # We used 3x3 kernel
    denoised_frame = cv2.filter2D(frame, -1, kernel)   # Applying filter on frame
    
    return denoised_frame   # Return denoised frame

def detect_edges(frame):
    """ Function for detecting edges 
    on frame with Canny Edge Detection """ 
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Convert frame to grayscale
    canny_edges = cv2.Canny(gray, 50, 150)  # Apply Canny edge detection function with thresh ratio 1:3
    
    return canny_edges  # Return edged frame

def region_of_interest(frame):
    """ Function for drawing region of 
    interest on original frame """
    
    height, width = frame.shape
    mask = np.zeros_like(frame)
    # only focus lower half of the screen
    polygon = np.array([[
        (int(width*0.30), height),              # Bottom-left point
        (int(width*0.46),  int(height*0.72)),   # Top-left point
        (int(width*0.58), int(height*0.72)),    # Top-right point
        (int(width*0.82), height),              # Bottom-right point
    ]], np.int32)
    
    cv2.fillPoly(mask, polygon, 255)
    roi = cv2.bitwise_and(frame, mask)
    
    return roi

def histogram(frame):
    """ Function for histogram 
    projection to find leftx and rightx bases """
    
    histogram = np.sum(frame, axis=0)   # Build histogram
    midpoint = np.int(histogram.shape[0]/2)     # Find mid point on histogram
    left_x_base = np.argmax(histogram[:midpoint])    # Compute the left max pixels
    right_x_base = np.argmax(histogram[midpoint:]) + midpoint    # Compute the right max pixels

    return left_x_base, right_x_base

def warp_perspective(frame):
    """ Function for warping the frame 
    to process it on skyview angle """
    
    height, width = frame.shape    # Get image size
    offset = 50     # Offset for frame ration saving
    
    # Perspective points to be warped
    source_points = np.float32([[int(width*0.46), int(height*0.72)], # Top-left point
                      [int(width*0.58), int(height*0.72)],           # Top-right point
                      [int(width*0.30), height],                     # Bottom-left point
                      [int(width*0.82), height]])                    # Bottom-right point
    
    # Window to be shown
    destination_points = np.float32([[offset, 0],                   # Top-left point
                      [width-2*offset, 0],                  # Top-right point
                      [offset, height],                      # Bottom-left point
                      [width-2*offset, height]])     # Bottom-right point
    
    matrix = cv2.getPerspectiveTransform(source_points, destination_points)    # Matrix to warp the image for skyview window
    skyview = cv2.warpPerspective(frame, matrix, (width, height))    # Final warping perspective 

    return skyview  # Return skyview frame

def detect_lines(frame):
    """ Function for line detection 
    via Hough Lines Polar """
    
    line_segments = cv2.HoughLinesP(frame, 1, np.pi/180 , 20, 
                                    np.array([]), minLineLength=40, maxLineGap=150)
    
    return line_segments    # Return line segment on road

def optimize_lines(frame, lines):
    """ Function for line optimization and 
    outputing one solid line on the road """
    
    height, width, _ = frame.shape  # Take frame size
    
    if lines is not None:   # If there no lines we take line in memory
        # Initializing variables for line distinguishing
        lane_lines = [] # For both lines
        left_fit = []   # For left line
        right_fit = []  # For right line
        
        for line in lines:  # Access each line in lines scope
            x1, y1, x2, y2 = line.reshape(4)    # Unpack actual line by coordinates

            parameters = np.polyfit((x1, x2), (y1, y2), 1)  # Take parameters from points gained
            slope = parameters[0]       # First parameter in the list parameters is slope
            intercept = parameters[1]   # Second is intercept
            
            if slope < 0:   # Here we check the slope of the lines 
                left_fit.append((slope, intercept))
            else:   
                right_fit.append((slope, intercept))

        if len(left_fit) > 0:       # Here we ckeck whether fit for the left line is valid
            left_fit_average = np.average(left_fit, axis=0)     # Averaging fits for the left line
            lane_lines.append(map_coordinates(frame, left_fit_average)) # Add result of mapped points to the list lane_lines
            
        if len(right_fit) > 0:       # Here we ckeck whether fit for the right line is valid
            right_fit_average = np.average(right_fit, axis=0)   # Averaging fits for the right line
            lane_lines.append(map_coordinates(frame, right_fit_average))    # Add result of mapped points to the list lane_lines
        
    return lane_lines       # Return actual detected and optimized line 


def map_coordinates(frame, parameters):
    """ Function for mapping given 
    parameters for line construction """
    
    height, width, _ = frame.shape  # Take frame size
    slope, intercept = parameters   # Unpack slope and intercept from the given parameters
    
    if slope == 0:      # Check whether the slope is 0
        slope = 0.1     # handle it for reducing Divisiob by Zero error
    
    y1 = height             # Point bottom of the frame
    y2 = int(height*0.72)  # Make point from middle of the frame down  
    x1 = int((y1 - intercept) / slope)  # Calculate x1 by the formula (y-intercept)/slope
    x2 = int((y2 - intercept) / slope)  # Calculate x2 by the formula (y-intercept)/slope
    
    return [[x1, y1, x2, y2]]   # Return point as array

def display_lines(frame, lines):
    """ Function for displaying lines 
    on the original frame """
    
    mask = np.zeros_like(frame)   # Create array with zeros using the same dimension as frame
    if lines is not None:       # Check if there is a existing line
        for line in lines:      # Iterate through lines list
            for x1, y1, x2, y2 in line: # Unpack line by coordinates
                cv2.line(mask, (x1, y1), (x2, y2), (0, 255, 0), 5)    # Draw the line on the created mask
    
    mask = cv2.addWeighted(frame, 0.8, mask, 1, 1)    # Merge mask with original frame
    
    return mask

def display_heading_line(frame, up_center, low_center):
    heading_image = np.zeros_like(frame)
    height, width, _ = frame.shape
    
    x1 = int(low_center)
    y1 = height
    x2 = int(up_center)
    y2 = int(height*0.72)
    
    cv2.line(heading_image, (x1, y1), (x2, y2), (0, 0, 255), 5)
    heading_image = cv2.addWeighted(frame, 0.8, heading_image, 1, 1)
    
    return heading_image

def get_floating_center(frame, lane_lines):
    """ Function for calculating steering angle 
    on the road looking at the lanes on it """
    
    height, width, _ = frame.shape # Take frame size
    
    if len(lane_lines) == 2:    # Here we check if there is 2 lines detected
        left_x1, _, left_x2, _ = lane_lines[0][0]   # Unpacking left line
        right_x1, _, right_x2, _ = lane_lines[1][0] # Unpacking right line
        
        low_mid = (right_x1 + left_x1) / 2  # Calculate the relative position of the lower middle point
        up_mid = (right_x2 + left_x2) / 2

    else:       # Handling undetected lines
        up_mid = int(width*1.9)
        low_mid = int(width*1.9)
    
    return up_mid, low_mid       # Return shifting points

def add_text(frame, image_center, left_x_base, right_x_base):
    """ Function for text outputing
    Output the direction of turn"""

    lane_center = left_x_base + (right_x_base - left_x_base) / 2 # Find lane center between two lines
    
    deviation = image_center - lane_center    # Find the deviation

    if deviation > 160:         # Prediction turn according to the deviation
        text = "Smooth Left"
        memory_text = text
    elif deviation < 40 or deviation > 150 and deviation <= 160:
        text = "Smooth Right"
        memory_text = text
    elif deviation >= 40 and deviation <= 150:
        text = "Straight"
        memory_text = text
    else:
        text = memory_text
    
    cv2.putText(frame, "DIRECTION: " + text, (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) # Draw direction
    
    return frame    # Retrun frame with the direction on it

def process_frame(frame):
    """ Main orchestrator that defines
    whole process of program execution"""
    
    # Declaring variables for fps
    avg_fps = 0
    fps_list = []
    
    start_time = time.time()    # Start the timer

    edges = detect_edges(frame)

    denoised_frame = denoise_frame(frame)   # Denoise frame from artifacts

    canny_edges = detect_edges(denoised_frame)  # Find edges on the frame

    roi_frame = region_of_interest(canny_edges)   # Draw region of interest

    warped_frame = warp_perspective(canny_edges)    # Warp the original frame, make it skyview
    left_x_base, right_x_base = histogram(warped_frame)         # Take x bases for two lines
    lines = detect_lines(roi_frame)                 # Detect lane lines on the frame
    lane_lines = optimize_lines(frame, lines)       # Optimize detected line
    mul_lines = display_lines(frame, lines)
    lane_lines_image = display_lines(frame, lane_lines) # Display solid and optimized lines
    
    up_center, low_center = get_floating_center(frame, lane_lines) # Calculate the center between two lines

    heading_line = display_heading_line(lane_lines_image, up_center, low_center)

    final_frame = add_text(heading_line, low_center, left_x_base, right_x_base) # Predict and draw turn

    fps = round(1.0 / (time.time() - start_time), 1)    # Here we calculate the fps
    fps_list.append(fps)           # Append fps to fps list
    cv2.putText(final_frame, "FPS: " + str(fps), (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) # Draw FPS

    if len(fps_list) == 60: # Calculate avg fps every timestamp
        avg_fps = round(sum(fps_list) / len(fps_list), 1)   # Averaging existing fps in the list
        fps_list = []
    cv2.putText(final_frame, "AVG FPS: " + str(avg_fps), (50, 150), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 0, 0), 2, cv2.LINE_AA) # Draw AVG FPS
    
    return final_frame  # Return final frame