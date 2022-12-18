#!/usr/bin/env python
# Lucas Walter
# December 2022

# Subscribe to a ros image, send into successive torch multiprocessing threads for alteration on the gpu,
# publish out as ros message for visualization

import copy
from threading import Lock

import rospy
import torch
import torchvision
from ddynamic_reconfigure_python.ddynamic_reconfigure import DDynamicReconfigure
from cv_bridge import CvBridge
from sensor_msgs.msg import Image


class Demo:
    def __init__(self):
        self.cv_bridge = CvBridge()
        self.pub = rospy.Publisher("image_out", Image, queue_size=2)
        self.sub = rospy.Subscriber("image_in", Image, self.image_callback, queue_size=2)

        self.config_lock = Lock()
        self.new_config = False
        self.config = None
        ddr = DDynamicReconfigure("")
        ddr.add_variable("angle_degrees", "angle degrees", 25.0, -270.0, 270.0)
        # ddr.add_variable("angle_degrees", "angle degrees", 0.0, -1.0, 1.0)
        self.ddr = ddr
        self.ddr.start(self.config_callback)

    def config_callback(self, config, level):
        with self.config_lock:
            self.config = copy.deepcopy(config)
        return config

    def get_config(self):
        with self.config_lock:
            config = copy.deepcopy(self.config)
            return config

    def image_callback(self, ros_image):
        t0 = rospy.Time.now()
        config = self.get_config()

        np_image = self.cv_bridge.imgmsg_to_cv2(ros_image, ros_image.encoding)
        torch_image = torch.from_numpy(np_image).to("cuda").permute(2, 0, 1).unsqueeze(0)

        torch_image2 = torchvision.transforms.functional.rotate(torch_image, config.angle_degrees)
        # print(torch_image2[0, ...].shape)
        np_image2 = torch_image2[0].permute(1, 2, 0).cpu().numpy()
        # print(np_image2.shape)
        ros_image2 = self.cv_bridge.cv2_to_imgmsg(np_image2, ros_image.encoding)
        ros_image2.header = ros_image.header
        ros_image2.encoding = ros_image.encoding
        self.pub.publish(ros_image2)

        te = rospy.Time.now() - t0
        rospy.loginfo_throttle(1.0, f"{ros_image.width}x{ros_image.height} {te.to_sec():0.3f}s")


if __name__ == "__main__":
    rospy.init_node("demo")
    node = Demo()
    rospy.spin()
