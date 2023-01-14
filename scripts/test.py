#!/usr/bin/env python
# see how much memory using torch takes

import time

# import rospy
import torch

# rospy.init_node("test")

# device = "cuda:0"
device = "cpu"

# while not rospy.is_shutdown():
while True:
    wd = 640
    ht = 360
    with torch.no_grad():
        a = torch.ones((ht, wd, 3), dtype=torch.float32, device=device)
        b = torch.reshape(a, (wd * ht, 3))
        c = torch.tensor([0, 1, 2, 3, 4, 5], device=device)
        text = f"{a.shape} {b.shape} {c}"
        print(text)
        # rospy.loginfo(text)
    # rospy.sleep(1.0)
    time.sleep(0.1)
