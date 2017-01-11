#ifndef SRC_MIXTURE_OF_GAUSSIANS_HPP_
#define SRC_MIXTURE_OF_GAUSSIANS_HPP_

#include <opencv2/core/core.hpp>
using namespace cv;

#include "pixel.hpp"
#include "tools.hpp"

const uchar BLACK = (uchar) 0;
const uchar WHITE = (uchar) 255;

/**
 * Klasa realizujaca algorytm GMM
 */
class MixtureOfGaussians
{
    private:
        Pixel **pixels;
        int height, width;
        int k;
        double init_std_dev;
        double min_var;
        bool is_initialized;
        bool grayscale_mode;

        void initialise(const Mat & input_frame);
    public:
        /**
         * Konstruktor
         * \param k - liczba rozkladow Gaussa dla kazdego piksela
         * \param alpha - wspolczynnik uczenia
         * \param bg_classifier - parametr T
         * \param min_var - minimalna dopuszczalna wariancja
         * \param init_std_dev - poczatkowe odchylenie standardowe
         * \param grayscale_mode - wlaczenie wersji algorytmu operujacej na obrazie w skali szarosci
         *
         */
        MixtureOfGaussians(int k, double alpha, double bg_classifier,
        			double init_std_dev, double min_var, bool grayscale_mode);

        /**
         * Desk=truktor
         */
        ~MixtureOfGaussians();

        /**
         * Uaktualnia rozklady Gaussa na podstawie kolejnej ramki wejsciowej, zwraca aktualna maske
         * wyjsciowa
         * \param input_frame - ramka wejsciowa
         * \param result_frame - maska wyjsciowa
         */
        void update(const Mat & input_frame, Mat & result_frame);

        void print_parameters(int row = -1, int col = -1, int gaussian_num = -1);
};

#endif /* SRC_MIXTURE_OF_GAUSSIANS_HPP_ */
