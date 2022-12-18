# rospytorch
rospy image sub/pub with pytorch multiprocessing experimentation

Subscribe to an Image, convert to a torch Tensor, send to the gpu, do a series of manipulations to create an altered image, then hand to a second torch thread, do some more manipulations while the first thread is freed up to handle another image, then return the image to cpu memory and publish out with rospy.

Also make use of kornia later.
