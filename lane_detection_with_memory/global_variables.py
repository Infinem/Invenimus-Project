import numpy as np  # For array operations
from collections import deque   # For using queues

# Here we initializing global variables for adding memory feature to our funtion
# This feature helps us to optimize lines on road
memory_line = np.zeros((1, 4), np.int32)
memory_lines = []
memory_left_fit = []
memory_right_fit = []
memory_x1_list = deque([])
memory_x2_list = deque([])

# Here we initializing global variables for adding memory feature to our funtion
# This feature helps us to optimize steering angle calculated from lane lines
memory_up_mid = deque([])
memory_low_mid = deque([])

memory_deviation = deque([])
memory_text = "Straight"

# Variable for fps calculation
avg_fps = 0
fps_list = []