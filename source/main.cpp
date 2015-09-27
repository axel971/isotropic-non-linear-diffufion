
#include <iostream>
#include <string>  
#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;

double computeDiffusion(double v, double lambda);
double flux(const Mat& img, int x, int y, double lambda);
void peronaMalik(Mat& out, const Mat& in, double dt, double tMax, double lambda);

int main(int argc , char** argv)
{
  //Initialization of the data
  Mat img = imread("../office_noisy.png", CV_LOAD_IMAGE_GRAYSCALE);
  Mat out;

  double dt = 0.1, lambda = 10., tMax = 100.; 

  img.convertTo(img, CV_64F); //Convert the image source in CV_64F (i.e float with a range of [0;1])

  peronaMalik(out, img, dt, tMax, lambda);

  double min, max;
  minMaxLoc(out, &min, &max);
  out.convertTo(out, CV_8U, 255./(max-min), -255.0*min/(max-min));

  //save
  imwrite("../result/AnisotropicOfficeLambda="+ to_string(lambda) + "t=" + to_string(tMax) + ".png", out);

  imshow("Non linear anisotropic diffusion", out);
  waitKey(0);
    
  return 0;
}

double computeDiffusion(double v, double lambda) 
{
  return  1. /( 1 + (v*v/(lambda*lambda)) );
}

double flux(const Mat& img, int i, int j, double lambda)
{
  double current = img.at<double>(i, j); 
  int width = img.cols;
  int height = img.rows;

  int prevI = i - 1;
  int nextI = i + 1;
  int prevJ = j - 1;
  int nextJ = j + 1;

  //Condition at the border of the image
  if(prevI < 0)
    prevI = 0;

  if(nextI >= height)
    nextI = height - 1;

  if(prevJ < 0)
    prevJ = 0;

  if(nextJ >= width)
    nextJ = width - 1;

  //Get the pixel in the neighborhood of (i,j)
  double pixelN = img.at<double>(prevI, j);
  double pixelS = img.at<double>(nextI, j);
  double pixelE = img.at<double>(i, nextJ);
  double pixelW = img.at<double>(i, prevJ);

  double pixelNW = img.at<double>(prevI, prevJ);
  double pixelSW = img.at<double>(nextI, prevJ);
  double pixelNE = img.at<double>(prevI, nextJ);
  double pixelSE = img.at<double>(nextI, nextJ); 

  //Compute the conduction coefficients
  double diffN = computeDiffusion(abs(current - pixelN), lambda);
  double diffS = computeDiffusion(abs(current - pixelS), lambda);
  double diffE = computeDiffusion(abs(current - pixelE), lambda);
  double diffW = computeDiffusion(abs(current - pixelW), lambda);
  /*
  double diffNW = computeDiffusion(abs(pixelNW - current), lambda);
  double diffSW = computeDiffusion(abs(pixelSW - current), lambda);
  double diffNE = computeDiffusion(abs(pixelNE - current), lambda);
  double diffSE = computeDiffusion(abs(pixelSE - current), lambda);
  */

  //Compute the flux
  double delta = (diffN * (pixelN - current)) 
    + (diffS * (pixelS - current)) 
    + (diffE * (pixelE - current)) 
    + (diffW * (pixelW - current)) ;
    /*    + 0.5 * ( (diffNW *(pixelNW - current)) 
	      + (diffSW *(pixelSW - current)) 
	      + (diffNE *(pixelNE - current)) 
	      + (diffSE *(pixelSE - current))  ); 
    */
  return delta;
}

/*Function which compute the anysotropic diffusion filter*/
void peronaMalik(Mat& out, const Mat& in, double dt, double tMax, double lambda)
{
  double eps = 1e-8, t = 0;

  Mat imgTemp = Mat(in.rows, in.cols, CV_64F);
  in.copyTo(out);


  while(abs(t - tMax) > eps)
    {
      for(int i = 0; i < in.rows; ++i)
	for(int j = 0; j < in.cols; ++j)
	  {
	    double value = (dt * flux(out, i, j, lambda)) + out.at<double>(i, j);
	    imgTemp.at<double>(i, j) = value;
	  }
      
      imgTemp.copyTo(out);
  
      t += dt;
    }
  
  return;
}

