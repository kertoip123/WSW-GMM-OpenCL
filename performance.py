import cv2
import numpy

class Performance:

    def __init__(self, name):
        self.name = name
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
    
    def compute(self, out_path, gt_path, frame_num):
        for i in range(1, frame_num+1):
            print i
            out = cv2.imread(out_path % i, cv2.IMREAD_GRAYSCALE)
            gt = cv2.imread(gt_path % i, cv2.IMREAD_GRAYSCALE)
            self.update(out, gt)
        
    
    def update(self, img, gt):
    
        for x in range(len(img)):
            for y in range(len(img[x])):
                if gt[x][y] < 85:
                    if img[x][y] == 0:
                        self.tn += 1
                    else:
                        self.fn += 1
                elif gt[x][y] > 85:
                    if img[x][y] == 255:
                        self.tp += 1
                    else:
                        self.fp += 1
                      
    
    def __str__(self):
        return 'Method: %s \nTP: %d\nTN: %d\nFP: %d\nFN: %d\n' % \
            (self.tp, self.tn, self.fp, self.fn)
            
if __name__ == "__main__":
    
    sequence_name = 'highway'
    gt_rel_path = './gt/' + sequence_name + '/gt00%04d.png'
    out_path = './output/' + sequence_name + '/out00%04d.jpg'
    
    performance = Performance(sequence_name)
    performance.compute(out_path, gt_rel_path, 1200)
    print performance
    
    