import pyopencl as cl
import pyopencl.array as cl_array
import numpy as np
import logging
import time
import cv2
from scipy.misc import *
from device_choose import *
from performance import compute_performance


sequence_num = 3

#choose INTEL_PLATFORM or NVIDIA_PLATFORM
pref_platform = NVIDIA_PLATFORM
#choose GPU_DEVICE or CPU_DEVICE
pref_device = GPU_DEVICE 

show_output = True
performance = True

nmixtures = 5
alpha = 0.01
k = 2.5
T = 0.5
#init_weight = 0.02
init_var = 25.0
min_var = 0.0

#Test sequence constants
sequences = ({'name':'highway', 'frame_num':1700, 'resolution': 320*240},
             {'name':'office', 'frame_num':2050, 'resolution': 360*240},
             {'name':'pedestrians', 'frame_num':1099, 'resolution': 360*240},
             {'name':'PETS2006', 'frame_num':1200, 'resolution': 720*576},
             {'name':'720p', 'frame_num':40, 'resolution': 1280*720},
             {'name':'1080p', 'frame_num':30, 'resolution': 1920*1080}
)

seq = sequences[sequence_num]
sequence_name = seq['name']
frame_num = seq['frame_num']
frame_resolution = seq['resolution']

in_path  = './tests/' + sequence_name + '/input/in00%04d.jpg'
gt_path = './tests/' + sequence_name + '/groundtruth/gt00%04d.png'
out_path = './tests/' + sequence_name + '/output/out00%04d.jpg' 


# Kernel function
kernel_src = 'mixture-of-gaussian.cl'

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

    mixture_data_buff = np.zeros(3*nmixtures*frame_resolution, dtype=np.float32)
    mixture_data_buff[0:frame_resolution*nmixtures] = 1.0/nmixtures
    mixture_data_buff[frame_resolution*nmixtures+1:2*frame_resolution*nmixtures] = init_var
    
    params_list = [k, T, init_var, min_var]
    mog_params = np.array(params_list, dtype=np.float32)

    f = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNORM_INT8)
	
	#Allocate memory for variables on the device
    mixture_data_g = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=mixture_data_buff)
    mog_params_g = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=mog_params)
    
    time_begin = time.time()
    
    cnt = 0
    for i in range(1, frame_num+1):
    	#Read in image
        img = imread(in_path % i, flatten=False, mode = 'RGBA')
        img_g = cl.image_from_array(ctx, img, 4, mode = 'r', norm_int = True)
        img_shape = (img.shape[1], img.shape[0])

        result_g = cl.Image(ctx, mf.WRITE_ONLY, f, shape=img_shape)

        # Call Kernel. Automatically takes care of block/grid distribution
        prg.mog_image(queue, img_shape, None, img_g, result_g, mixture_data_g, 
                      mog_params_g, np.float32(alpha))
    	
        result = np.empty_like(img)
        cl.enqueue_copy(queue, result, result_g, origin=(0, 0), region=img_shape)
		
        cnt += 1
        # Show out image
        imsave(out_path % i, cv2.cvtColor(result, cv2.COLOR_BGRA2GRAY))
        #logging.debug('Iteration %d done.' % i)      
        if show_output:
            cv2.imshow('GMM', result)
                    
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    
    time_end = time.time()
    avg_fps = cnt/(time_end - time_begin)
    logging.info('Average fps: %.1f' % avg_fps)
    
    if performance:
        logging.info('Computing performance...')
        compute_performance(out_path, gt_path, cnt, frame_resolution, ctx, queue)
