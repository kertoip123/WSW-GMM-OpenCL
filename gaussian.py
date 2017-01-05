import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import logging
import time
import cv2
from performance import *
from scipy.misc import *
from device_choose import *

#Test sequence constants
sequence_name = 'highway'
frame_num = 1200

rel_path  = './input/' + sequence_name + '/in00%04d.jpg'
gt_rel_path = './gt/' + sequence_name + '/gt00%04d.png'
dest_path = './output/' + sequence_name + '/out00%04d.jpg' 

nmixtures = 5

# Kernel function
kernel_src = 'mixture-of-gaussian.cl'

#choose INTEL_PLATFORM or NVIDIA_PLATFORM
pref_platform = NVIDIA_PLATFORM
#choose GPU_DEVICE or CPU_DEVICE
pref_device = GPU_DEVICE 

compute_performance = False

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Choose graphic device and create context for it
    ctx = cl.Context([device_choose(pref_platform, pref_device)])
    mf  = cl.mem_flags

    # Create queue for each kernel execution
    queue = cl.CommandQueue(ctx)

    #Kernel function instantiation
    kernel = str()
    with open(kernel_src, 'r') as content_file:
        kernel = content_file.read()
    prg = cl.Program(ctx, kernel).build()

    mixture_data_buff = np.zeros(13000000, dtype=np.float32)
    params_list = [2.5, 0.5, 0.02, 75.0, 25.0]
    mog_params = np.array(params_list, dtype=np.float32)
    alpha = 0.1

    f = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8)
	
	#Allocate memory for variables on the device
    mixture_data_g = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=mixture_data_buff)
    mog_params_g = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=mog_params)
    
    time_begin = time.time()
    
    performance = Performance(sequence_name)
    
    for i in range(1, frame_num+1):
    	#Read in image
        img = imread(rel_path % i, flatten=False, mode = 'RGBA')
        img_g = cl.image_from_array(ctx, img, 4, mode = 'r', norm_int = True)
        img_shape = (img.shape[1], img.shape[0])

        result_g = cl.Image(ctx, mf.WRITE_ONLY, f, shape=img_shape)

        # Call Kernel. Automatically takes care of block/grid distribution
        prg.mog_image(queue, img_shape, None, img_g, result_g, mixture_data_g, mog_params_g, np.float32(alpha))
    	
        result = np.empty_like(img)
        cl.enqueue_copy(queue, result, result_g, origin=(0, 0), region=img_shape)
		
        if compute_performance:
            gt_img = cv2.imread(gt_rel_path % i, cv2.IMREAD_GRAYSCALE)
            performance.update(cv2.cvtColor(result, cv2.COLOR_BGRA2GRAY), gt_img)
        
        # Show out image
        imsave(dest_path % i, cv2.cvtColor(result, cv2.COLOR_BGRA2GRAY))
        #logging.debug('Iteration %d done.' % i)      
        cv2.imshow('GMM', result)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    time_end = time.time()
    avg_fps = frame_num/(time_end - time_begin)
    logging.info('Average fps: %.1f' % avg_fps)
    if compute_performance:
        logging.info(performance)