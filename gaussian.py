import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import logging
from scipy.misc import *

# Get platforms, both CPU and GPU
plat = cl.get_platforms()
CPU = plat[0].get_devices()
try:
    GPU = plat[1].get_devices()
except IndexError:
    GPU = "none"

#Create context for GPU/CPU
if GPU!= "none":
    ctx = cl.Context(GPU)
else:
    ctx = cl.Context(CPU)

# Create queue for each kernel execution
queue = cl.CommandQueue(ctx)

mf = cl.mem_flags

#Test sequence constants
frame_num = 1700
rel_path  = './input/in00%04d.jpg'
dest_path = './output/out00%04d.jpg' 

nmixtures = 10

# Kernel function
kernel_src = 'mixture-of-gaussian.cl'

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

    mixture_data_buff = np.zeros(3000000, dtype=np.float32)
    mog_params = np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)
    alpha = 0.1

    for i in range(1, frame_num+1):
    	#Read in image
    	img = imread(rel_path % i, flatten=True).astype(np.float32)

        f = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
        #img_g = cl.Image(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, f, shape=img.shape, hostbuf = img)
        img_g = cl.image_from_array(ctx, img)
        result_g = cl.Image(ctx, mf.WRITE_ONLY, f, shape=img.shape)

    	#Allocate memory for variables on the device
        #img_g =  cl.Buffer(ctx, mf.READ_ONLY | mf.COPY_HOST_PTR, hostbuf=img)
    	#result_g = cl.Buffer(ctx, mf.WRITE_ONLY, img.nbytes)
    	mixture_data_g = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=mixture_data_buff)
    	mog_params_g = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=mog_params)

    	# Call Kernel. Automatically takes care of block/grid distribution
    	prg.mog_image(queue, img.shape, None, img_g, result_g, mixture_data_g, mog_params_g, np.float32(alpha))
    	#result = np.empty_like(img)
    	result = np.zeros((img.shape[0], img.shape[1], 4), dtype=np.uint8)
        cl.enqueue_copy(queue, result, result_g, origin=(0, 0), region=img.shape)


    	# Show the blurred image
    	imsave(dest_path % i, img)
        logging.info('Iteration %d done.' % i)
