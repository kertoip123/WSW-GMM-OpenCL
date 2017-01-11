#include "pixel.hpp"
#include "gaussian.hpp"
#include "tools.hpp"
#include <iostream>
#include <algorithm>

double Pixel::T = 0;
int Pixel::k = 0;
//double Pixel::init_std_dev = 0.0;

void Pixel::initialise(double *weight, double **gaussian_means, double *standard_devation){
    gaussian_ptr = new Gaussian[k];
    for(int i=0; i < k; ++i)
    {
        gaussian_ptr[i].initialise(weight[i], gaussian_means[i], standard_devation[i]);
    }
}

void Pixel::print(int gaussian_num)
{
    if(gaussian_num == -1)
    {
        for(int gauss_num = 0; gauss_num < k; ++gauss_num)
        {
            cout<<gauss_num<<":";
            gaussian_ptr[gauss_num].print();
            cout<<" ";
        }
    }
    else
    {
        if(gaussian_num >= 0 && gaussian_num < k)
        {
            cout<<gaussian_num<<":";
            gaussian_ptr[gaussian_num].print();
        }
        else
        {
            print_error(gaussian_num);
        }
    }
}

void Pixel::get_rgb_mean(int gaussian_num, double * gaussian_means)
{
    if(gaussian_num < k)
    {
        gaussian_ptr[gaussian_num].get_rgb_mean(gaussian_means);
    }
    else
    {
        print_error(gaussian_num);
    }
}

void Pixel::print_error(int gaussian_num)
{
    cout<<"No Gaussian with id: "<<gaussian_num<<endl;
}

void Pixel::sort()
{
    std::sort(gaussian_ptr, (gaussian_ptr + k));

    double sum = 0;
    bool isForeground = false;
    for(int i=0; i<k; i++)
    {
    	sum += gaussian_ptr[i].get_weight();
    	gaussian_ptr[i].set_isForeground(isForeground);
    	if(sum > T)
    		isForeground = true;
    }
}

bool Pixel::is_foreground(double * rgb)
{
    // Check if any gaussian matches current pixel
    bool match_found = false;
    int match_index;
    for(match_index=0; match_index < k; ++match_index)
    {
        match_found = gaussian_ptr[match_index].check_pixel_match(rgb);
        if(match_found == true)
            break;
    }

    if(match_found)// Case 1: Match is found
    {

        for(int i=0; i < k; i++)
        {
            if(i == match_index)
            {
                gaussian_ptr[i].update_matched(rgb);
                continue;
            }
            gaussian_ptr[i].update_unmatched();
        }

        bool retval = gaussian_ptr[match_index].isForeground();
        sort();

        return retval;
    }
    else //Case 2: No match is found
    {
    	double * new_means = rgb;
    	double new_deviation = get_max_deviation();
    	//double new_deviation = init_std_dev;
    	double new_weight = gaussian_ptr[k-1].get_weight();

    	gaussian_ptr[k-1].initialise(new_weight, new_means, new_deviation);

        return true;
    }
}

double Pixel::get_max_deviation()
{
	double max_dev = gaussian_ptr[0].get_deviation();
	for(int i=0; i<k; ++i)
		max_dev = (gaussian_ptr[i].get_deviation() > max_dev) ? gaussian_ptr[i].get_deviation() : max_dev;

	return max_dev;
}
