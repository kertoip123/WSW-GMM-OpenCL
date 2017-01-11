#include "mixture_of_gaussians.hpp"
#include "tools.hpp"
#include "gaussian.hpp"
#include <iostream>
#include <cstdlib>
#include <ctime>

MixtureOfGaussians::MixtureOfGaussians(int k, double alpha, double bg_classifier,
								double init_std_dev, double min_var, bool grayscale_mode)
{
    this->width = 0;
    this->height = 0;
    this->k = k;
    this->min_var = min_var;
    this->init_std_dev = init_std_dev;
    this->grayscale_mode = grayscale_mode;

    Pixel::set_T(bg_classifier);
    Pixel::set_k(k);
    Gaussian::set_alpha(alpha);

    is_initialized = false;
    pixels = NULL;
}

MixtureOfGaussians::~MixtureOfGaussians()
{
	for(int i=0; i < height; i++)
		delete [] pixels[i];
	delete [] pixels;

}

void MixtureOfGaussians::update(const Mat & input_frame, Mat & result_frame)
{

    if(!is_initialized)
    {
    	Mat temp(input_frame.size(), CV_8U);
        initialise(input_frame);
        result_frame = temp.clone();
        is_initialized = true;
    }

    double rgb[RGB_COMPONENTS_NUM];
    const uchar * input_pixel_ptr;
    uchar * result_pixel_ptr;
    uchar mask;
    for(int row = 0; row < height; ++row)
    {
        input_pixel_ptr = input_frame.ptr(row);
        result_pixel_ptr = result_frame.ptr(row);
        for(int col = 0; col < width; ++col)
        {
            //RGB reverted order
            rgb[2] = (double) *input_pixel_ptr++;
            rgb[1] = (double) *input_pixel_ptr++;
            rgb[0] = (double) *input_pixel_ptr++;

            if (this->grayscale_mode)
            {
            	rgb[0] = (rgb[0]+rgb[1]+rgb[2])/RGB_COMPONENTS_NUM;
            	rgb[1] = rgb[0];
            	rgb[2] = rgb[0];
            }

            mask = (pixels[row][col].is_foreground(rgb)) ? WHITE : BLACK;
            //for(int i = 0; i < RGB_COMPONENTS_NUM; ++i)
            *result_pixel_ptr++ = mask;
        }
    }
}

void MixtureOfGaussians::print_parameters(int row, int col, int gaussian_num)
{
    if(row == -1 || col == -1)
    {
        for(int row = 0; row < height; ++row)
        {
            for(int col = 0; col < width; ++col)
            {
                cout<<"["<<row<<"]"<<"["<<col<<"]: ";
                pixels[row][col].print(gaussian_num);
                cout<<endl;
            }
        }
    }
    else
    {
        if(row < 0 && row >= height)
        {
            cout<<"Row out of bounds: "<<row<<endl;
            return;
        }
        if(col < 0 && col >= width)
        {
            cout<<"Col out of bounds: "<<col<<endl;
            return;
        }
        cout<<"["<<row<<"]"<<"["<<col<<"]: ";
        pixels[row][col].print(gaussian_num);
        cout<<endl;
    }
}

void MixtureOfGaussians::initialise(const Mat & input_frame)
{
    height = input_frame.rows;
    width = input_frame.cols;

	double** new_gaussian_means = new double* [k];
	double * new_weight = new double [k];
	double * new_deviation = new double[k];

	//Pixel::init_std_dev = this->init_std_dev;

	for(int i=0; i<k; i++)
		new_gaussian_means[i] = new double [RGB_COMPONENTS_NUM];

	for(int i=0; i<k; i++)
	{
		for(int j=0; j<RGB_COMPONENTS_NUM; j++)
			new_gaussian_means[i][j] = 0;
			//new_gaussian_means[i][j] = mean_value + i*step;

		new_weight[i] = 1.0/this->k;
		new_deviation[i] = init_std_dev;
	}

    pixels = new Pixel*[height];
    for(int i=0; i < height; ++i)
    {
        pixels[i] = new Pixel[width];
    }

    for(int i=0; i < height; ++i)
    {
        for(int j=0; j < width; ++j)
        {
            pixels[i][j].initialise(new_weight, new_gaussian_means, new_deviation);
        }
    }

    for(int i=0; i<k; i++)
        delete [] new_gaussian_means[i];
    delete [] new_weight;
    delete [] new_deviation;
    delete [] new_gaussian_means;
}

