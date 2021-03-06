import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import logging
from scipy.misc import *

#Test sequence constants
frame_num = 1700
rel_path  = './input/in00%04d.jpg'
dest_path = './output/out00%04d.jpg' 

nmixtures = 10

# Kernel function
kernel_src = 'in_2_out.cl'

# Get the first available graphic device. Prioritize GPU over CPU
def device_choose():
    preffered_device = None
    for platform in cl.get_platforms():
        for device in platform.get_devices():
            device_type = cl.device_type.to_string(device.type)
            if device_type == "GPU":
                logging.debug(device)
                return device
            elif device_type == "CPU" and preffered_device == None:
                preffered_device = device
    logging.debug(preffered_device)
    return preffered_device

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Choose graphic device and create context for it
    ctx = cl.Context([device_choose()])
    mf  = cl.mem_flags

    # Create queue for each kernel execution
    queue = cl.CommandQueue(ctx)

    #Kernel function instantiation
    kernel = str()
    with open(kernel_src, 'r') as content_file:
        kernel = content_file.read()
    prg = cl.Program(ctx, kernel).build()


    for i in range(1, frame_num+1):
    	#Read in image
    	img = imread(rel_path % i, flatten=False, mode = 'RGBA')
        img_shape = (img.shape[1], img.shape[0])

        f = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
        img_g = cl.image_from_array(ctx, img, 4)
        result_g = cl.Image(ctx, mf.WRITE_ONLY, f, shape=img_shape)

    	prg.in_2_out(queue, img_shape, None, img_g, result_g)
    	
        result = np.empty_like(img)
        cl.enqueue_copy(queue, result, result_g, origin=(0, 0), region=img_shape)

    	imsave(dest_path % i, result)
        logging.info('Iteration %d done.' % i)
