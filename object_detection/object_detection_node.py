import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection3DArray, Detection2DArray
from builtin_interfaces.msg import Time as HeaderTime
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import threading
from yolo_detector import YOLODetector

'''
Subscribes to stereo camera and rgb images.

[ obj1[ObjectHypothesisWithPose, BoundingBox3D], obj2, ... ] 

Use YOLO

Publishes the detected objects in the image using either Detection3DArray

'''

class ObjectDetectionNode(Node):

    def __init__(self):
        '''
        params:


        '''
        super().__init__('object_detection', parameter_overrides=[])

        self.declare_parameter('depth_camera_topic', '/stereo_camera/depth/image_raw') 
        self.declare_parameter('rgb_topic', '/rgb_camera/color/image_raw')
        self.declare_parameter('objects_3D_topic', '/objects_3D')
        self.declare_parameter('objects_2D_topic', '/objects_2D')
        self.declare_parameter('stereo_image_dimensions', [648, 480])
        self.declare_parameter('rgb_image_dimensions', [1280, 720])
        
        if self.has_parameter('depth_camera_topic'):
            self.depth_topic_ = self.get_parameter('depth_camera_topic').get_parameter_value().string_value

        if self.has_parameter('rgb_topic'):
            self.rgb_topic_ = self.get_parameter('rgb_topic').get_parameter_value().string_value 

        if self.has_parameter('objects_3D_topic'):
            self.objects_3D_topic_ = self.get_parameter('objects_3D_topic').get_parameter_value().string_value

        if self.has_parameter('objects_2D_topic'):
            self.objects_2D_topic_ = self.get_parameter('objects_2D_topic').get_parameter_value().string_value

        if self.has_parameter('stereo_image_dimensions'):
            self.stereo_image_dimensions = self.get_parameter('stereo_image_dimensions').get_parameter_value().integer_array_value

        if self.has_parameter('rgb_image_dimensions'):
            self.rgb_image_dimensions = self.get_parameter('rgb_image_dimensions').get_parameter_value().integer_array_value
        # Fields

        self.depth_image_= np.zeros(self.stereo_image_dimensions)
        self.rgb_image_ = np.zeros(self.rgb_image_dimensions)

        self.objects_2D_ = Detection2DArray()
        self.objects_3D_ = Detection3DArray()

        self.depth_image_ts_ = HeaderTime()
        self.rgb_image_ts_ = HeaderTime()

        # Subscribers
        self.depth_sub_ = self.create_subscription(Image, self.depth_topic_, self.depth_callback, 10)
        self.rgb_sub_ = self.create_subscription(Image, self.rgb_topic_, self.rgb_callback, 10)

        # Publishers
        self.objects_pub_3D_ = self.create_publisher(Detection3DArray, self.objects_3D_topic_, 10)
        self.objects_pub_2D_ = self.create_publisher(Detection2DArray, self.objects_2D_topic_, 10)

        self.sync_timer_ = self.create_timer(0.033, self.sync_images)

        self.detector_ = YOLODetector('yolov8s.pt')

        self.lock_images = threading.Lock()

    def get_ts(self, time_stamp: HeaderTime) -> float:
        '''
        params:
        seconds: int
        nanoseconds: int

        returns:
        Time
        '''
        return float(time_stamp.sec + (time_stamp.nanosec / 1000000000.0))


    def depth_callback(self, depth_image: Image):
        '''
        params:
        msg: Image

        returns:
        None
        '''
        try:
            self.depth_image_ts_ = self.get_ts(depth_image.header.stamp)
            self.depth_image_ = CvBridge().imgmsg_to_cv2(depth_image, desired_encoding="passthrough")
        except CvBridgeError as e:
            self._logger.warning(f'Issue Converting left stereo image: {e}')

    def rgb_callback(self, rgb_image: Image):
        '''
        params:
        msg: Image

        returns:
        None
        '''
        try:
            self.rgb_image_ts_ = self.get_ts(rgb_image.header.stamp)
            self.rgb_image_ = CvBridge().imgmsg_to_cv2(rgb_image)
        except CvBridgeError as e:
            self._logger.warning(f'Issue Converting rgb image: {e}')

    def sync_images(self):
        '''
        params:
        None

        returns:
        None
        '''
        self.lock_images.acquire()

        cv2.imshow('Stereo Image', self.rgb_image_)
        cv2.waitKey(1)
            
        self.detect_objects(self.depth_image_, self.rgb_image_)

        self.lock_images.release()
    
    def detect_objects(self, depth_image: np.ndarray, rgb_image: np.ndarray):
        '''
        params:
        left_stereo_image: np.ndarray
        right_stereo_image: np.ndarray
        rgb_image: np.ndarray

        returns:
        None
        '''
        pass

    def dewarp_and_level_rgb(self, rgb_image: np.ndarray):
        '''
        params:
        rgb_image: np.ndarray

        returns:
        None
        '''
        pass

    def match_rgb_depth(self, rgb_image: np.ndarray, depth_image: np.ndarray):
        '''
        params:
        rgb_image: np.ndarray
        depth_image: np.ndarray

        returns:
        rgb_matching
        '''
        pass

    def get_object_depth(self, depth_image: np.ndarray, box: np.ndarray) -> float:
        '''
        params:
        depth_image: np.ndarray
        box: np.ndarray -> (x, y, w, h)

        returns:
        depth: float
        '''
        
        obj_depth = depth_image[box[1]:(box[1] + box[3]), box[0]:(box[0] + box[2])] * 0.001

        return obj_depth


    def publish_objects(self):
        '''
        params:
        None

        returns:
        None
        '''
        pass