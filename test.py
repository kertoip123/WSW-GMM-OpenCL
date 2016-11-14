import pyopencl as cl
import numpy as np
import logging
from scipy.misc import *

# Start of configuration

# Test sequence constants
frame_num = 1700
rel_path  = './input/in00%04d.jpg'
dest_path = './output/out00%04d.jpg' 

# Kernel function path
kernel_src = 'median.cl'

# End of configuration

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
    logging.basicConfig(level=logging.DEBUG)
    
    # Choose graphic device and create context for it
    ctx = cl.Context([device_choose()])
    mf  = cl.mem_flags

    # Create queue for each kernel execution
    queue = cl.CommandQueue(ctx)

    # Kernel function instantiation
    kernel = str()
    with open(kernel_src, 'r') as content_file:
        kernel = content_file.read()
    prg = cl.Program(ctx, kernel).build()
    
    for i in range(1, frame_num+1):
        # Read in image
        img = imread(rel_path % i, flatten=True).astype(np.float32)
    
        #Allocate memory for variables on the device
        img_g    = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)
        result_g = cl.Buffer(ctx, mf.WRITE_ONLY                  , img.nbytes)
        width_g  = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(img.shape[1]))
        height_g = cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=np.int32(img.shape[0]))
    
        # Call Kernel. Automatically takes care of block/grid distribution
        prg.medianFilter(queue, img.shape, None , img_g, result_g, width_g, height_g)
        result = np.empty_like(img)
        cl.enqueue_copy(queue, result, result_g)
    
        # Show the blurred image
        imsave(dest_path % i, result)
        logging.info('Iteration %d done.' % i)
