import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection3DArray, Detection2DArray
from builtin_interfaces.msg import Time as HeaderTime
from cv_bridge import CvBridge, CvBridgeError
import numpy as np
import threading

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

        if self.has_parameter('left_stereo_topic'):
            self.left_stereo_topic_ = self.get_parameter('left_stereo_topic').get_parameter_value().string_value

        if self.has_parameter('right_stereo_topic'):
            self.right_stereo_topic_ = self.get_parameter('right_stereo_topic').get_parameter_value().string_value

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

        self.left_stereo_image_ = np.zeros(self.stereo_image_dimensions)
        self.right_stereo_image_ = np.zeros(self.stereo_image_dimensions)
        self.rgb_image_ = np.zeros(self.rgb_image_dimensions)

        self.objects_2D_ = Detection2DArray()
        self.objects_3D_ = Detection3DArray()

        self.left_stereo_image_ts_ = HeaderTime()
        self.right_stereo_image_ts_ = HeaderTime()
        self.rgb_image_ts_ = HeaderTime()

        # Subscribers
        self.left_stereo_sub_ = self.create_subscription(Image, self.left_stereo_topic_, self.left_stereo_callback, 10)
        self.right_stereo_sub_ = self.create_subscription(Image, self.right_stereo_topic_, self.right_stereo_callback, 10)
        self.rgb_sub_ = self.create_subscription(Image, self.rgb_topic_, self.rgb_callback, 10)

        # Publishers
        self.objects_pub_3D_ = self.create_publisher(Detection3DArray, self.objects_3D_topic_, 10)
        self.objects_pub_2D_ = self.create_publisher(Detection2DArray, self.objects_2D_topic_, 10)

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


    def left_stereo_callback(self, left_image: Image):
        '''
        params:
        msg: Image

        returns:
        None
        '''
        try:
            self.left_stereo_image_ts_ = self.get_ts(left_image.header.stamp)
            self.left_stereo_image = CvBridge().imgmsg_to_cv2(left_image)
        except CvBridgeError as e:
            self._logger.warning(f'Issue Converting left stereo image: {e}')

    def right_stereo_callback(self, right_image: Image):
        '''
        params:
        msg: Image

        returns:
        None
        '''
        try:
            self.right_stereo_image_ts_ = self.get_ts(right_image.header.stamp)
            self.right_stereo_image = CvBridge().imgmsg_to_cv2(right_image)
        except CvBridgeError as e:
            self._logger.warning(f'Issue Converting right stereo image: {e}')

    def rgb_callback(self, rgb_image: Image):
        '''
        params:
        msg: Image

        returns:
        None
        '''
        try:
            self.rgb_image_ts_ = self.get_ts(rgb_image.header.stamp)
            self.rgb_image = CvBridge().imgmsg_to_cv2(rgb_image)
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

        if self.left_stereo_image_ts_ != self.right_stereo_image_ts_:
            self._logger.warning('Left and Right Stereo Images Synchronized')
        # elif abs(self.left_stereo_image_ts_ - self.rgb_image_ts_) > 0.0333:
        #     self._logger.warning('Left Stereo and RGB Images Synchronized')   
            
        self.detect_objects(self.left_stereo_image_, self.right_stereo_image_, self.rgb_image_)
            
        self.lock_images.release()

    
    def detect_objects(self, left_stereo_image: np.ndarray, right_stereo_image: np.ndarray, rgb_image: np.ndarray):
        '''
        params:
        left_stereo_image: np.ndarray
        right_stereo_image: np.ndarray
        rgb_image: np.ndarray

        returns:
        None
        '''
        pass

    def publish_objects(self):
        '''
        params:
        None

        returns:
        None
        '''
        pass

    def run(self):
        '''
        params:
        None

        returns:
        None
        '''
        pass    
