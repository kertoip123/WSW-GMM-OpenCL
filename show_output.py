import cv2
from time import sleep

def show_output(name, rel_path, frame_num):
    
    cv2.namedWindow(name)
    
    for i in range(1, frame_num+1):
        img_out = cv2.imread(rel_path % i, cv2.IMREAD_GRAYSCALE)
        cv2.imshow(name, img_out)
        
        #about 60fps
        sleep(17/1000)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
   
if __name__ == "__main__":
    show_output('highway', './output/highway/out00%04d.jpg', 1200)
    