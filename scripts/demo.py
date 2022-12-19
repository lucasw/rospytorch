#!/usr/bin/env python
# Lucas Walter
# December 2022

# Subscribe to a ros image, send into successive torch multiprocessing threads for alteration on the gpu,
# publish out as ros message for visualization

import copy
from threading import Lock

import numpy as np
import rospy
import torch
import torchvision
from ddynamic_reconfigure_python.ddynamic_reconfigure import DDynamicReconfigure
from cv_bridge import CvBridge
from sensor_msgs.msg import CompressedImage
from sensor_msgs.msg import Image


class Demo:
    def __init__(self):
        self.cv_bridge = CvBridge()
        self.pub = rospy.Publisher("image_out", Image, queue_size=2)
        use_compressed = rospy.get_param("~compressed", True)
        if use_compressed:
            self.sub1 = rospy.Subscriber("image_in/compressed", CompressedImage,
                                         self.compressed_image_callback, queue_size=2)
        else:
            self.sub0 = rospy.Subscriber("image_in", Image, self.image_callback, queue_size=2)

        self.config_lock = Lock()
        self.new_config = False
        self.config = None
        ddr = DDynamicReconfigure("")
        ddr.add_variable("angle_degrees", "angle degrees", 25.0, -270.0, 270.0)
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
        # TODO(lucasw) send the raw bytes into torch and do any conversion needed there
        np_image = self.cv_bridge.imgmsg_to_cv2(ros_image, ros_image.encoding)
        torch_image = torch.from_numpy(np_image).to("cuda").permute(2, 0, 1).unsqueeze(0)
        self.process_torch_image(torch_image, ros_image.header, ros_image.encoding)
        te = rospy.Time.now() - t0
        rospy.loginfo_throttle(1.0, f"{ros_image.width}x{ros_image.height} {te.to_sec():0.3f}s")

    def compressed_image_callback(self, ros_compressed_image):
        image_format = ros_compressed_image.format
        if image_format not in ["jpg", "png"]:
            rospy.logwarn_once(4.0, f"{image_format}")
            return
        t0 = rospy.Time.now()
        # `The given buffer is not writeable`
        # torch_compressed_image = torch.frombuffer(ros_compressed_image.data, dtype=torch.uint8).to("cuda")
        np_compressed_image = np.copy(np.frombuffer(ros_compressed_image.data, np.uint8))
        # can't be on gpu yet
        torch_compressed_image = torch.from_numpy(np_compressed_image)  # .to("cuda")
        # only will work on jpg and png
        # rospy.loginfo(f"{image_format} {torch_compressed_image.shape}")
        # torch_image = torchvision.io.decode_image(torch_compressed_image).permute(2, 0, 1).unsqueeze(0)
        torch_image = torchvision.io.decode_jpeg(torch_compressed_image,
                                                 mode=torchvision.io.ImageReadMode.RGB,
                                                 device="cuda").unsqueeze(0)
        # rospy.loginfo(f"{torch_image.shape} {torch_image.device}")
        # TODO(lucasw) switch to bgr8
        encoding = "rgb8"
        self.process_torch_image(torch_image, ros_compressed_image.header, encoding)
        te = rospy.Time.now() - t0
        th = rospy.Time.now() - ros_compressed_image.header.stamp
        rospy.loginfo_throttle(1.0, f"{torch_image.shape} {te.to_sec():0.3f}s {th.to_sec():0.3f}s")
        # TODO(lucasw) avoid nvjpeg 5 errors
        text = f"{torch.cuda.memory_allocated()} {torch.cuda.memory_reserved()}"
        del torch_image
        torch.cuda.empty_cache()
        rospy.loginfo(f"{text} -> {torch.cuda.memory_allocated()} {torch.cuda.memory_reserved()}")

    def process_torch_image(self, torch_image, header, encoding):
        '''
        torch_image should be Tensor (N, 3, H, W)
        '''

        config = self.get_config()

        torch_image2 = torchvision.transforms.functional.rotate(torch_image, config.angle_degrees)
        rospy.loginfo_once(f"{torch_image.shape} -> {torch_image2.shape}")
        # rospy.loginfo(torch_image2.shape)
        np_image2 = torch_image2[0].permute(1, 2, 0).cpu().numpy()

        del torch_image2
        torch.cuda.empty_cache()

        # rospy.loginfo(np_image2.shape)
        # print(np_image2.shape)
        ros_image2 = self.cv_bridge.cv2_to_imgmsg(np_image2, encoding)
        ros_image2.header = header
        ros_image2.encoding = encoding
        self.pub.publish(ros_image2)


if __name__ == "__main__":
    rospy.init_node("demo")
    node = Demo()
    rospy.spin()
