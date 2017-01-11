#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/video/background_segm.hpp>
#include <opencv2/opencv.hpp>

#include <iostream>
#include <string>
#include <cstdarg>
#include <ctime>

#include "file_name_generator.hpp"
#include "window_manager.hpp"
#include "mixture_of_gaussians.hpp"
#include "performance.hpp"

using namespace cv;
using namespace std;

#define MODE 			0
#define TEST_GROUP      0

bool compute_coef = true;
bool grayscale_mode = true;

#if MODE == 0
const int windows_num = 4;
#elif MODE == 1
const int windows_num = 1;
#elif MODE == 2
const int windows_num = 1;
#endif

//#define DEBUG


#if TEST_GROUP == 0

const int frame_num = 1700;
const string input_frame_prefix = "../../tests/highway/input/in";
const string gt_frame_prefix = "../../tests/highway/groundtruth/gt";

#elif TEST_GROUP ==  1

const int frame_num = 2050;
const string input_frame_prefix = "../../tests/office/input/in";
const string gt_frame_prefix = "../../tests/office/groundtruth/gt";

#elif TEST_GROUP ==  2

const int frame_num = 1099;
const string input_frame_prefix = "../../tests/pedestrians/input/in";
const string gt_frame_prefix = "../../tests/pedestrians/groundtruth/gt";

#elif TEST_GROUP ==  3

const int frame_num = 1200;
const string input_frame_prefix = "../../tests/PETS2006/input/in";
const string gt_frame_prefix = "../../tests/PETS2006/groundtruth/gt";

#elif TEST_GROUP ==  4

const int frame_num = 40;
const string input_frame_prefix = "../../tests/720p/input/in";
const string gt_frame_prefix = "../../tests/720p/groundtruth/gt";

#elif TEST_GROUP ==  5

const int frame_num = 30;
const string input_frame_prefix = "../../tests/1080p/input/in";
const string gt_frame_prefix = "../../tests/1080p/groundtruth/gt";

#endif

void print_image(const Mat & image, int my_row = -1, int my_col = -1)
{
    int rows = image.rows;
    int cols = image.cols;
    int bgr[3];
    if(my_row == -1 && my_col == -1)
        cout<<"Image size: "<<rows<<" x "<<cols<<endl;
    for(int row = 0; row < rows; ++row)
    {
        const uchar* p = image.ptr(row);
        for(int col = 0; col < cols; ++col)
        {
            //points to each pixel B,G,R value in turn assuming a CV_8UC3 color image
            for(int i=0; i < 3; ++i)
            {
                bgr[i] = (int) *p++;
            }
            if((my_row == -1 && my_col == -1) || (my_row == row && my_col == col))
            {
                cout<<"["<<row<<"]["<<col<<"]: ";
                cout<<"("<<bgr[2]<<", "<<bgr[1]<<", "<<bgr[0]<<")\t";
            }
        }
        if(my_row == -1 && my_col == -1)
            cout<<endl;
    }
}

int main(int argc, char** argv)
{
    MixtureOfGaussians MoG(5, 0.01, 0.5, 5 , 0, grayscale_mode);
    Performance mog_performance;
    Performance mog_cv_performance;

    FileNameGenerator input_file_name_generator(input_frame_prefix, JPG);
    FileNameGenerator ground_truth_file_name_generator(gt_frame_prefix, PNG);

    initialize_windows();

    Mat input_frame, gt_frame, cv_mixture_of_gaussians_frame, output_frame;
    string frame_name, gt_name;

    Ptr< BackgroundSubtractor> cv_mixture_of_gaussians;
    cv_mixture_of_gaussians = createBackgroundSubtractorMOG2();

#ifdef DEBUG
    const int observed_x = 120;
    const int observed_y = 180;
    const uchar RED_COLOR[3] = {255,0,0};
#endif

    clock_t begin = clock();

    for(int frame_id = 1; frame_id < frame_num; frame_id++)
    {
        frame_name = input_file_name_generator.get_frame_name(frame_id);
        input_frame = imread(frame_name, 1);
#if MODE == 0
        gt_name = ground_truth_file_name_generator.get_frame_name(frame_id);
        gt_frame = imread(gt_name, 0);

        cv_mixture_of_gaussians->apply(input_frame, cv_mixture_of_gaussians_frame);
        MoG.update(input_frame, output_frame);

    	imshow(INPUT, input_frame);
    	imshow(OPENCV_MOG_2, cv_mixture_of_gaussians_frame);
    	imshow(GROUND_TRUTH, gt_frame);
    	imshow(MY_MOG, output_frame);

    	if(compute_coef)
    	{
    		mog_performance.count_coefficients(output_frame, gt_frame);
    		mog_cv_performance.count_coefficients(cv_mixture_of_gaussians_frame, gt_frame);
    	}

#elif MODE == 1
        MoG.update(input_frame, output_frame);
    	imshow(MY_MOG, output_frame);

    	if(compute_coef)
    	{
            gt_name = ground_truth_file_name_generator.get_frame_name(frame_id);
    		gt_frame = imread(gt_name, 0);
    		imshow(GROUND_TRUTH, gt_frame);
    		mog_performance.count_coefficients(output_frame, gt_frame);
    	}
#elif MODE == 2
        cv_mixture_of_gaussians->apply(input_frame, cv_mixture_of_gaussians_frame);
    	imshow(OPENCV_MOG_2, cv_mixture_of_gaussians_frame);

    	if(compute_coef)
    	{
            gt_name = ground_truth_file_name_generator.get_frame_name(frame_id);
    		gt_frame = imread(gt_name, 0);
    		imshow(GROUND_TRUTH, gt_frame);
    		mog_cv_performance.count_coefficients(cv_mixture_of_gaussians_frame, gt_frame);
    	}
#endif



#ifdef DEBUG
        print_image(input_frame, observed_x , observed_y );
        MoG.print_parameters(observed_x , observed_y );
        uchar* p = output_frame.ptr(observed_x);
        for(int i=0; i < 3; ++i)
        {
            p[3*observed_y + i] = RED_COLOR[2-i];
        }
#endif



        if(waitKey(10) != -1)//experimental value ~~~63fps
            break;
    }
    clock_t end = clock();

    if (compute_coef)
    {
    	cout << "GMM" << endl;
    	mog_performance.print_metrics();

    	cout << "OpenCV - GMM" << endl;
    	mog_cv_performance.print_metrics();

    }

    double t_elapsed = double(end - begin)/CLOCKS_PER_SEC;
    cout << "Avg FPS: " << frame_num/t_elapsed << endl;

    return 0;
}
