import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from vision_msgs.msg import Detection2DArray, Detection2D, ObjectHypothesisWithPose
from builtin_interfaces.msg import Time as HeaderTime
from cv_bridge import CvBridge, CvBridgeError
import cv2
import numpy as np
import threading
from yolo_detector import YOLODetector
import yaml

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
        self.declare_parameter('rgb_image_dimensions', [1920, 1080])
        self.declare_parameter('model_path', 'yolov8s.pt')
        self.declare_parameter('intrinsic_path', '/home/formatspecifier/Projects/object_detection/object_detection/intrinsics/chain_b0179.yaml')
        
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

        if self.has_parameter('model_path'):
            self.model_path = self.get_parameter('model_path').get_parameter_value().string_value

        if self.has_parameter('intrinsic_path'):
            self.intrinsic_path = self.get_parameter('intrinsic_path').get_parameter_value().string_value
            fs = cv2.FileStorage(self.intrinsic_path, cv2.FILE_STORAGE_READ)
            self.mtx = fs.getNode('camera_matrix').mat()
            self.dist = fs.getNode('dist_coeff').mat()

        # Fields

        self.depth_image_= np.zeros(self.stereo_image_dimensions)
        self.rgb_image_ = np.zeros(self.rgb_image_dimensions)

        self.objects_2D_ = Detection2DArray()

        self.depth_image_ts_ = HeaderTime()
        self.rgb_image_ts_ = HeaderTime()

        # Subscribers
        self.depth_sub_ = self.create_subscription(Image, self.depth_topic_, self.depth_callback, 10)
        self.rgb_sub_ = self.create_subscription(Image, self.rgb_topic_, self.rgb_callback, 10)

        # Publishers
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
        
        rect_rgb = self.dewarp_and_level_rgb(rgb_image)
        rgb_matching = self.match_rgb_depth(rect_rgb, depth_image)
        boxes, conf = self.detector_.detect(rgb_matching)
        boxes_depth = []
        for box in boxes:
            obj_depth = self.get_object_depth(depth_image, box)
            boxes_depth.append(obj_depth)

        self.publish_objects(boxes, conf, boxes_depth)
        

    def dewarp_and_level_rgb(self, rgb_image: np.ndarray) -> np.ndarray:
        '''
        params:
        rgb_image: np.ndarray

        returns:
        None
        '''
        cv2.undistort(rgb_image, self.mtx, self.dist, rgb_image, self.mtx)
        cv2.imshow('Undistorted Image', rgb_image)
        cv2.waitKey(1)
        return rgb_image

    def match_rgb_depth(self, rgb_image: np.ndarray, depth_image: np.ndarray) -> np.ndarray:
        '''
        params:
        rgb_image: np.ndarray
        depth_image: np.ndarray

        returns:
        rgb_matching
        '''
        # rgb_matching = np.zeros(depth_image.shape)
        rgb_matching = rgb_image

        return rgb_matching

    def get_object_depth(self, depth_image: np.ndarray, box: np.ndarray) -> float:
        '''
        params:
        depth_image: np.ndarray
        box: np.ndarray -> (x, y, w, h)

        returns:
        depth: float
        '''
        
        obj_depth = depth_image[box[1]:(box[1] + box[3]), box[0]:(box[0] + box[2])] * 0.001

        return float(obj_depth)


    def publish_objects(self, boxes: list, conf: list, boxes_depth: list):
        '''
        params:
        boxes: list
        conf: list
        boxes_depth: list

        returns:
        None
        '''

        # self.objects_2D_ = Detection2DArray()

        for i in range(len(boxes)):
            detection = Detection2D()
            detection.results = []
            detection.header.stamp.sec = self.rgb_image_ts_
            detection.bbox.center.x = boxes[i][0]
            detection.bbox.center.y = boxes[i][1]
            detection.bbox.size_x = boxes[i][2]
            detection.bbox.size_y = boxes[i][3]
            det = ObjectHypothesisWithPose()
            det.id = 0
            det.score = conf[i]
            det.pose.pose.position.z = boxes_depth[i]

            detection.results.append(det)
            # self.objects_2D_.detections.append(detection)


