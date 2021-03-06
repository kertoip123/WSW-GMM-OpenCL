#ifndef SRC_WINDOW_MANAGER_HPP_
#define SRC_WINDOW_MANAGER_HPP_

#include <string>
#include <cstdarg>

using namespace std;
using namespace cv;

const string INPUT = "Input";
const string GROUND_TRUTH = "Ground truth";
const string OPENCV_MOG_2 = "OpenCV MoG v2";
const string MY_MOG = "My MoG";

const int WINDOW_PADDING = 30;
const int WINDOW_X_POS_OFFSET = 350;
const int WINDOW_Y_POS_OFFSET = 300;

void initialize_windows();

void update_windows(int windows_num, ...);

#endif /* SRC_WINDOW_MANAGER_HPP_ */
