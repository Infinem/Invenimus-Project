# Import libraries
import math     # For math functions
import time     # For timer to calculate fps
import numpy as np  # For array operations
import cv2      # For video/image processing

from global_variables import *
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
    
    # Global variable declaration scope for function use
    global memory_line
    global memory_lines
    global memory_left_fit
    global memory_right_fit
    global memory_x1_list
    global memory_x2_list

    height, width, _ = frame.shape  # Take frame size
    
    if lines is not None:   # If there no lines we take line in memory
        # Initializing variables for line distinguishing
        lane_lines = [] # For both lines
        left_fit = []   # For left line
        right_fit = []  # For right line
        
        for line in lines:  # Access each line in lines scope
            if len(memory_x1_list) < 10 and len(memory_x2_list) < 10:     # Including memory feature
                x1, y1, x2, y2 = line.reshape(4)    # Unpack actual line by coordinates
                memory_line = line  # Remember detected line
                memory_x1_list.append(x1)   # Append x1 to the queque
                memory_x2_list.append(x2)    # Append x2 to the queque

            else:
                x1, y1, x2, y2 = line.reshape(4)    # Unpack actual line by coordinates
                avg_x1 = sum(memory_x1_list) / len(memory_x1_list)  # Averaging points saved in queue for x1
                avg_x2 = sum(memory_x2_list) / len(memory_x2_list)  # Averaging points saved in queue for x2

                if abs(x1 - avg_x1) > width*0.21 and abs(x2 - avg_x2) > width*0.21: # Here we check the difference between new lines
                    x1, y1, x2, y2 = memory_line.reshape(4)                         # coming and average of point saved in the queue

                memory_x1_list.append(x1)   # Add new point for x1
                memory_x2_list.append(x2)   # Add new point for x2
                memory_x1_list.popleft()    # Remove first element from the queue for x1
                memory_x2_list.popleft()    # Remove first element from the queue for x2
            
            parameters = np.polyfit((x1, x2), (y1, y2), 1)  # Take parameters from points gained
            slope = parameters[0]       # First parameter in the list parameters is slope
            intercept = parameters[1]   # Second is intercept
            
            if (slope < -0.3) and (slope > -1.8376):   # Here we check the slope of the lines (-0.3, -1.8)
                left_fit.append((slope, intercept))       # and remove the outliers
                memory_left_fit = left_fit          # remember fit for left line
            elif (slope < 1.5) and (slope > 0.3):       #(1.8, 0.3)
                right_fit.append((slope, intercept))
                memory_right_fit = right_fit        # remember fit for right line
            else:
                left_fit = memory_left_fit          # Recall fits if there is no lines
                right_fit = memory_right_fit        # which satisfies the conditions
        
        if len(left_fit) > 0:       # Here we ckeck whether fit for the left line is valid
            left_fit_average = np.average(left_fit, axis=0)     # Averaging fits for the left line
            lane_lines.append(map_coordinates(frame, left_fit_average)) # Add result of mapped points to the list lane_lines
        else:
            left_fit_average = np.average(memory_left_fit, axis=0)      # Take average of previous fit if there is no fits
            lane_lines.append(map_coordinates(frame, left_fit_average)) # which satisfies the condition
            
        if len(right_fit) > 0:       # Here we ckeck whether fit for the right line is valid
            right_fit_average = np.average(right_fit, axis=0)   # Averaging fits for the right line
            lane_lines.append(map_coordinates(frame, right_fit_average))    # Add result of mapped points to the list lane_lines
        else:
            right_fit_average = np.average(memory_right_fit, axis=0)     # Take average of previous fit if there is no fits
            lane_lines.append(map_coordinates(frame, right_fit_average))  # which satisfies the condition
            
        memory_lines = lane_lines   # Rewrite memory line
        
        return lane_lines       # Return actual detected and optimized line 
    
    else:
        return memory_lines     # Return saved line if there no other lines

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
    
    # Global variable declaration scope for function use
    global memory_up_mid
    global memory_low_mid
    
    height, width, _ = frame.shape # Take frame size
    
    if len(lane_lines) == 2:    # Here we check if there is 2 lines detected
        left_x1, _, left_x2, _ = lane_lines[0][0]   # Unpacking left line
        right_x1, _, right_x2, _ = lane_lines[1][0] # Unpacking right line
        if len(memory_low_mid) < 5 and len(memory_up_mid) < 5:   # Adding memory feature
            low_mid = (right_x1 + left_x1) / 2  # Here we calculate the relative position of the lower middle point of roi
            up_mid = (left_x2 + right_x2) / 2  # Here we calculate the relative position of the upper middle point of roi
            
            memory_up_mid.append(up_mid)  # Saving upper mid point
            memory_low_mid.append(low_mid)  # Saving lower mid point
            
        else:                               # Process here if queue is filled
            low_mid = (right_x1 + left_x1) / 2  # Calculate the relative position of the lower middle point

            avg_low_mid = sum(memory_low_mid) / len(memory_low_mid) # Averaging existing mid points in queue for lower

            dx1 = low_mid - avg_low_mid # Calculate delta between avg mid points and new mid point for lower
            
            memory_low_mid.append(low_mid)  # Append to the queue new lower mid point
            memory_low_mid.popleft()        # Remove first element of the queue
            
            low_mid -= dx1  # Substact delta from the actual lower mid point
            dx2 = dx1 * (right_x2 - left_x2) / (right_x1 - left_x1) # Find delta for the upper mid point using the relative ratio
            up_mid = (left_x2 + right_x2) / 2 - dx2*2     # Find new upper mid point

            memory_up_mid.append(up_mid)  # Append to the queue new upper mid point
            memory_up_mid.popleft()        # Remove first element of the queue
    else:       # Handling undetected lines
        up_mid = 0
        low_mid = int(width*1.9)

    avg_up_mid = sum(memory_up_mid) / len(memory_up_mid)
    
    return avg_up_mid, low_mid       # Return shifting points

def add_text(frame, image_center, left_x_base, right_x_base):
    """ Function for text outputing
    Output the direction of turn"""
    
    global memory_text
    global memory_deviation

    lane_center = left_x_base + (right_x_base - left_x_base) / 2 # Find lane center between two lines
    
    if len(memory_deviation) < 5:
        deviation = image_center - lane_center    # Find the deviation
        memory_deviation.append(deviation)
    else:
        deviation = image_center - lane_center    # Find the deviation
        memory_deviation.append(deviation)
        memory_deviation.popleft()
        deviation = sum(memory_deviation) / len(memory_deviation)

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
    
    # Declaring global variable for fps for function use
    global avg_fps
    global fps_list
    
    start_time = time.time()    # Start the timer

    edges = detect_edges(frame)

    denoised_frame = denoise_frame(frame)   # Denoise frame from artifacts

    canny_edges = detect_edges(denoised_frame)  # Find edges on the frame

    roi_frame = region_of_interest(canny_edges)   # Draw region of interest

    warped_frame = warp_perspective(canny_edges)    # Warp the original frame, make it skyview
    left_x_base, right_x_base = histogram(warped_frame)         # Take x bases for two lines
    lines = detect_lines(roi_frame)                 # Detect lane lines on the frame
    lane_lines = optimize_lines(frame, lines)       # Optimize detected line
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