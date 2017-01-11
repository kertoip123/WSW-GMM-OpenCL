import cv2
import numpy as np
from performance import *
from scipy.misc import *
from device_choose import *

 
def compute_performance(out_path, gt_path, frame_num, resolution, ctx, queue):

    mf  = cl.mem_flags

    #Kernel function instantiation
    kernel_src = 'performance.cl'
    kernel = str()
    with open(kernel_src, 'r') as content_file:
        kernel = content_file.read()
    prg = cl.Program(ctx, kernel).build()
    
    performance_data = np.zeros(4*resolution, dtype=np.uint32)

    f = cl.ImageFormat(cl.channel_order.RGBA, cl.channel_type.UNSIGNED_INT8)
    
    performance_data_g = cl.Buffer(ctx, mf.COPY_HOST_PTR, hostbuf=performance_data)
        
    for i in range(1, frame_num+1):
    	#Read in image
        out = imread(out_path % i, flatten=False, mode = 'RGBA')
        gt = imread(gt_path % i, flatten=False, mode = 'RGBA')
        out_g = cl.image_from_array(ctx, out, 4, mode = 'r', norm_int = True)
        gt_g = cl.image_from_array(ctx, gt, 4, mode = 'r', norm_int = True)
        img_shape = (out.shape[1], out.shape[0])

        # Call Kernel. Automatically takes care of block/grid distribution
        prg.performance(queue, img_shape, None, out_g, gt_g, performance_data_g)
        
        cl.enqueue_copy(queue, performance_data, performance_data_g)
    	  
    TP = sum(performance_data[0:resolution])
    TN = sum(performance_data[resolution:2*resolution])
    FN = sum(performance_data[2*resolution:3*resolution])
    FP = sum(performance_data[3*resolution:4*resolution])
    
    Re = float(TP)/(TP+FN)
    Spec = float(TN)/(TN+FP)
    FPR = float(FP)/(FP+TN)
    FNR = float(FN)/(FN+TP)
    PWC = 100*(FN+FP)/(TP+FN+FP+TN)
    Pr = float(TP)/(TP+FP)
    F1 = 2*Pr*Re/(Pr+Re)
    
    print 'TP = %d  TN = %d  FP = %d  FN = %d' %(TP, TN, FP, FN)
    print 'Re = %.4f' % Re
    print 'Spec = %.4f' % Spec
    print 'FPR = %.4f' % FPR
    print 'FNR = %.4f' % FNR
    print 'PWC = %.4f' % PWC
    print 'Pr = %.4f' % Pr
    print 'F1 = %.4f' % F1
    
    log_file = open("output.txt", "w")
    log_file.write('%d\n' % TP)
    log_file.write('%d\n' % TN)
    log_file.write('%d\n' % FP)
    log_file.write('%d\n' % FN)
    log_file.write('%.4f\n' % Re)
    log_file.write('%.4f\n' % Spec)
    log_file.write('%.4f\n' % FPR)
    log_file.write('%.4f\n' % FNR)
    log_file.write('%.4f\n' % PWC)
    log_file.write('%.4f\n' % Pr)
    log_file.write('%.4f\n' % F1)
    log_file.close()
    