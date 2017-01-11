#ifndef SRC_PIXEL_HPP_
#define SRC_PIXEL_HPP_

#include "gaussian.hpp"
#include "tools.hpp"

using namespace std;

/**
 * Klasa reprezentujaca pojedynczy piksel na obrazie
 */
class Pixel
{
    private:
        static double T;
        static int k;

        Gaussian *gaussian_ptr;

        void print_error(int gaussian_num);
        void sort(); //Parameter T from article, background classifier
        double get_max_deviation();

    public:
        /**
         * Ustawia parametr T
         * \param T - wartosc parametru
         * \return void
         */
        static void set_T(double T) { Pixel::T = T; }
        /**
         * Ustawia parametr k
         * \param k - wartosc parametru
         * \return void
         */
        static void set_k(double k) { Pixel::k = k; }

        /**
         * Inicjalizuje rozklady Gaussa dla danego piksela
         * \param weigth - tablica wag poszczegolnych rozkladow Gaussa
         * \param gaussian_means - tablica wartosci srednich dla poszczegolnych rozkladow
         * \param stadnard_deviation - odchylenie standardowe dla poszczegolnych rozkladow
         * \return void
         */
        void initialise(double *weight, double **gaussian_means, double *standard_devation);

        /**
         * Pobiera wartosci srednie dla danego rozkladu Gaussa
         * \param gaussian_num - numer rozkladu Gaussa
         * \param gaussian_means - tablica do ktorej zapisane zostana wartosci srednie
         */
        void get_rgb_mean(int gaussian_num, double * gaussian_means);

        /**
         * Sprawdza czy nowy piksel nalezy do pierwszego planu
         * \param rgb - wektor RGB nowego piksela
         * \return true jezeli piksel jest obiektem pierwszoplanowym, w przeciwnym przypadku false
         */
        bool is_foreground(double * rgb);

        void print(int gaussian_num = -1);

        //static double init_std_dev;
};

#endif /* SRC_PIXEL_HPP_ */
