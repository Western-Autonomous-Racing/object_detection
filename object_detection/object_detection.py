from object_detection_node import ObjectDetectionNode
import rclpy

def main(args=None):
    try:
        rclpy.init(args=args)
        object_detection_node = ObjectDetectionNode()
        rclpy.spin(object_detection_node)
    except KeyboardInterrupt:
        object_detection_node.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()