/*******************************************************************************
*                                                                              *
*                       Brno University of Technology                          *
*                       CPhoto@FIT                                             *
*                                                                              *
*                       Tone Mapping Studio                                    *
*                                                                              *
*                       Brno 2022                                              *
*                                                                              *
*                       Implementation of the TMOLi21 class                *
*                                                                              *
*******************************************************************************/

/*
 * Following code is inspired of original public MATLAB implementation
 * of article:
 *
 *    Domain Transform for Edge-Aware Image and Video Processing
 *    Eduardo S. L. Gastal  and  Manuel M. Oliveira
 *    ACM Transactions on Graphics. Volume 30 (2011), Number 4.
 *    Proceedings of SIGGRAPH 2011, Article 69.
 *
 * Source: http://inf.ufrgs.br/~eslgastal/DomainTransform/
 * Authors: Eduardo S. L. Gastal, Manuel M. Oliveira
 *
 * All rights of original implementation reserved to the authors
 * and any possible similarities are taken as citation of mentioned source.
 */

#include <limits>
#include <float.h>

#include "TMOLi21.h"


using namespace std;
using namespace cv;

/**
 * @brief Detail-Preserving Multi-Exposure Fusion with Edge-Preserving Structural Patch Decomposition
 */


TMOLi21::TMOLi21()
{
   SetName(L"TMOLi21");
   SetDescription(L"Detail-Preserving Multi-Exposure Fusion with Edge-Preserving Structural Patch Decomposition");

   kernelSize.SetName(L"kernel size");
   kernelSize.SetDescription(L"Kernel size X*X: X=");
   kernelSize.SetDefault(3);
   kernelSize = 3;
   this->Register(kernelSize);
   kernelSize.SetRange(3, 20);

   lambda.SetName(L"lambda");
   lambda.SetDescription(L"Lambda param in the equation");
   lambda.SetDefault(0.25);
   lambda = 0.25;
   this->Register(lambda);
   lambda.SetRange(0, 1);
}

TMOLi21::~TMOLi21()
{
}


cv::Mat TMOLi21::SingalStructureS(cv::Mat patch) {
   cv::Mat L = TMOLi21::MeanIntensityL(patch);
   cv::Mat diff = patch-L;
   auto norm = cv::norm(diff);
   return (diff/norm);
}


double inline TMOLi21::SignalStrengthC(cv::Mat patch) {
   cv::Mat L = TMOLi21::MeanIntensityL(patch);
   cv::Mat diff = patch-L;
   double C = cv::norm(diff);
   return C;
}

cv::Mat inline TMOLi21::MeanIntensityL(cv::Mat patch) {
   double a = CalcA(TMOLi21::CalculateVariance(patch), lambda.GetDouble());
   double b = CalcB(TMOLi21::CalculateVariance(patch), lambda.GetDouble(), TMOLi21::CalculateMean(patch));
   cv::Mat L;
   L = patch*a;
   L = L+b;
   return L;
}

double TMOLi21::CalculateVariance(const cv::Mat& patch) {
   cv::Scalar mean, stdDev;
   cv::meanStdDev(patch, mean, stdDev);
   return stdDev[0]*stdDev[0];
}

double TMOLi21::CalculateMean(const cv::Mat& patch) {
   cv::Scalar meanValue = cv::mean(patch);
   return meanValue[0];
}

double inline TMOLi21::CalcA(double var, double lambda) {
   return var/(var+lambda);
}

double inline TMOLi21::CalcB(double var, double lambda, double mean) {
   return (lambda/(var+lambda))*mean;
}

double CalculateGamma() {
   
}

int TMOLi21::Transform()
{
   double* pSourceData = pSrc->GetData();
   double* pDestinationData = pDst->GetData();

   std::vector
   cv::Mat imageMat = cv::Mat(iHeight, iWidth, CV_64F, (void*)pSourceData);
   for (int r = 0; r<iHeight; r += kernelSize.GetInt()) {
      for (int col = 0; col<iWidth; col += kernelSize.GetInt()) {
         cv::Rect roi(col, r, kernelSize.GetInt(), kernelSize.GetInt());
         roi.width = std::min(roi.width, iWidth-col);
         roi.height = std::min(roi.height, iHeight-r);

         cv::Mat patch = imageMat(roi);

         auto l = TMOLi21::MeanIntensityL(patch);
         auto s = TMOLi21::SingalStructureS(patch);
         auto c = TMOLi21::SignalStrengthC(patch);
      }
   }
   return 0;
}