import rclpy
from rclpy.node import Node
import numpy as np
from queue import PriorityQueue
from rclpy.qos import QoSProfile
import math
from nav_msgs.msg import OccupancyGrid, Odometry 
from geometry_msgs.msg import PoseStamped, Twist
from sensor_msgs.msg import Image
import time
import cv2
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA
import os
from cv_bridge import CvBridge

class Collecter(Node):
    def __init__(self):
        super().__init__('data_collecter')

        self.bridge = CvBridge()
        self.img_sub = self.create_subscription(
            Image, '/camera/image_raw',
            self.camera_callback,
            10
        )
        self.data = None
        self.path_to_images = 'collected_images'
        self.img_count = 0
        self.save_timer = self.create_timer(2.0, self.collecter_timer)
    
    
    def camera_callback(self, msg):
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        self.data = cv_image

    def collecter_timer(self):
        if self.data is not None:
            filename = f'image_{self.img_count}.jpg'
            filepath = os.path.join(self.path_to_images, filename)
            cv2.imwrite(filepath, self.data)
            print('img saved')
            self.img_count += 1

def main(args = None):
    rclpy.init(args=args)
    collector = Collecter()
    try:
        rclpy.spin(collector)
    except KeyboardInterrupt:
        print("you stopped it")
    finally:
        collector.destroy_node()
        rclpy.shutdown()

if __name__== '__main__':
    main()