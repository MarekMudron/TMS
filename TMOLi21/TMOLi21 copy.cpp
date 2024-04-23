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
#include <bits/stdc++.h>

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

   kernelSize.SetName(L"kernelsize");
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

   beta.SetName(L"beta");
   beta.SetDescription(L"Beta param in the equation");
   beta.SetDefault(2);
   beta = 2;
   this->Register(beta);
   beta.SetRange(2, 10);

   p.SetName(L"p");
   p.SetDescription(L"p-norm param in the equation");
   p.SetDefault(2);
   p = 2;
   this->Register(p);
   p.SetRange(1, 10);
}

TMOLi21::~TMOLi21()
{}

void fillRGB(cv::Mat& matrix, double* pSourceData, int height, int width) {
   for (int rowI = 0; rowI<height; rowI++)
   {
      for (int colI = 0; colI<width; colI++)
      {
         /** need to store rgb in mat to calculate colour ratio later */
         matrix.at<cv::Vec3d>(rowI, colI)[0] = *pSourceData++;
         matrix.at<cv::Vec3d>(rowI, colI)[1] = *pSourceData++;
         matrix.at<cv::Vec3d>(rowI, colI)[2] = *pSourceData++;
      }
   }
}

using PatchesMatrix = vector<vector<cv::Mat>>;
using CoefsMatrix = vector<vector<cv::Scalar>>;

PatchesMatrix CutIntoPatches(const cv::Mat& matrix, int height, int width, int kernelSize) {
   PatchesMatrix patches;
   for (int r = 0; r<height; r += kernelSize) {
      std::vector<cv::Mat> patchesrow;

      for (int col = 0; col<width; col += kernelSize) {
         cv::Rect roi(col, r, kernelSize, kernelSize);
         roi.width = std::min(roi.width, width-col);
         roi.height = std::min(roi.height, height-r);
         cv::Mat patch = matrix(roi);
         patchesrow.push_back(patch);
      }
      patches.push_back(patchesrow);
   }
   return patches;
}

cv::Scalar CalculateA(cv::Mat patch, double lambda) {
   cv::Scalar meanValue, stdDev;
   cv::meanStdDev(patch, meanValue, stdDev);
   return (stdDev*stdDev)/(stdDev*stdDev+cv::Scalar(lambda, lambda, lambda));
}

cv::Scalar CalculateB(cv::Mat patch, double lambda) {
   cv::Scalar mean, stdDev;
   cv::meanStdDev(patch, mean, stdDev);
   cv::Scalar lambScalar = cv::Scalar(lambda, lambda, lambda);
   return mean*lambScalar/(stdDev*stdDev+lambScalar);
}

cv::Mat MeanIntensityL(cv::Mat patch, double lambda) {
   auto a = CalculateA(patch, lambda);
   //TODO fix
   return CalculateA(patch, lambda)*patch+CalculateB(patch, lambda)*cv::Mat::ones(patch.size(), patch.type());
}

cv::Scalar SignalStrengthC(cv::Mat patch, double lambda) {
   cv::Scalar mean, stdDev;
   cv::meanStdDev(patch, mean, stdDev);
   cv::Scalar lambScalar = cv::Scalar(lambda, lambda, lambda);
   cv::Mat mxMat(patch.size(), patch.type());
   mxMat.setTo(mean);
   return (lambScalar*cv::norm(patch-mxMat))/(lambScalar+stdDev*stdDev);
}

cv::Mat SignalStructureS(cv::Mat patch) {
   cv::Scalar mean, stdDev;
   cv::meanStdDev(patch, mean, stdDev);
   cv::Mat mxMat(patch.size(), patch.type());
   mxMat.setTo(mean);
   cv::Mat result;
   cv::divide(patch-mxMat, cv::norm(patch-mxMat), result);
   return result;
}

cv::Scalar Gamma(cv::Mat patch, cv::Mat l, double lambda, double maxDist, int p, double bottomSum) {
   cv::Scalar lambScalar = cv::Scalar(lambda, lambda, lambda);
   cv::Scalar maxDistScalar = cv::Scalar(maxDist, maxDist, maxDist);
   cv::Scalar meanValue, stdDev;
   cv::meanStdDev(patch, meanValue, stdDev);
   auto first = (lambScalar*maxDistScalar)/(stdDev*stdDev+lambScalar);
   auto second = pow(cv::norm(patch-l), p-1)/bottomSum;
   return first*second;
}

cv::Scalar CalculateBeta(const cv::Mat& intensity, int beta) {
   int center_i = intensity.rows/2;
   int center_j = intensity.cols/2;

   auto centerIntensity = intensity.at<cv::Vec3d>(center_i, center_j)[0];
   double v;
   if (centerIntensity>=0&&centerIntensity<=0.25) {
      v = pow(centerIntensity, beta)*pow(0.25, 1-beta);
   }
   else if (centerIntensity>=0.25&&centerIntensity<=0.5) {
      v = 0.5-pow(0.5-centerIntensity, beta)*pow(0.25, 1-beta);
   }
   else if (centerIntensity>=0.5&&centerIntensity<=0.75) {
      v = 0.5-pow(centerIntensity-0.5, beta)*pow(0.25, 1-beta);
   }
   else {
      v = pow(1-centerIntensity, beta)*pow(0.25, 1-beta);
   }
   return cv::Scalar(v, v, v);
}

void normalize(std::vector<std::vector<cv::Scalar>>& betas) {
   int numChannels = betas[0][0].channels;

   // Initialize a Scalar to hold sums for each channel
   cv::Scalar channelSums(0, 0, 0, 0);

   // Calculate the sum of each channel across all Scalars
   for (const auto& vecScalars:betas) {
      for (const auto& scalar:vecScalars) {
         for (int i = 0; i<numChannels; ++i) {
            channelSums[i] += scalar[i];
         }
      }
   }

   // Normalize each Scalar by the sum of its corresponding channel
   for (auto& vecScalars:betas) {
      for (auto& scalar:vecScalars) {
         for (int i = 0; i<numChannels; ++i) {
            if (channelSums[i]!=0) {  // Avoid division by zero
               scalar[i] /= channelSums[i];
            }
         }
      }
   }
}

cv::Mat mergeTiles(const std::vector<std::vector<cv::Mat>>& imageTiles) {
   // Determine the total size of the merged image
   int totalHeight = 0;
   int totalWidth = 0;
   int type = imageTiles[0][0].type();

   // Calculate the total width and maximum row height
   std::vector<int> rowHeights(imageTiles.size(), 0);
   std::vector<int> colWidths(imageTiles[0].size(), 0);

   for (int i = 0; i<imageTiles.size(); ++i) {
      int currentRowWidth = 0;
      for (int j = 0; j<imageTiles[i].size(); ++j) {
         currentRowWidth += imageTiles[i][j].cols;
         rowHeights[i] = std::max(rowHeights[i], imageTiles[i][j].rows);
      }
      totalWidth = std::max(totalWidth, currentRowWidth);
   }

   // Calculate the total height
   for (int height : rowHeights) {
      totalHeight += height;
   }

   // Create a large image to hold all the tiles
   cv::Mat bigImage(totalHeight, totalWidth, type);

   // Copy each tile into the corresponding position in the big image
   int yOffset = 0;
   for (int i = 0; i<imageTiles.size(); ++i) {
      int xOffset = 0;
      for (int j = 0; j<imageTiles[i].size(); ++j) {
         cv::Rect roi(xOffset, yOffset, imageTiles[i][j].cols, imageTiles[i][j].rows);
         cv::Mat destinationROI = bigImage(roi);
         imageTiles[i][j].copyTo(destinationROI);
         xOffset += imageTiles[i][j].cols;
      }
      yOffset += rowHeights[i];
   }

   return bigImage;
}

int TMOLi21::Transform()
{
   double* pSourceData = pSrc->GetData();
   double* pDestinationData = pDst->GetData();

   // using raw pixel values
   pSrc->Convert(TMO_RGB);
   pDst->Convert(TMO_RGB);

   int kernelSizeValue = kernelSize.GetInt();
   int lamb = lambda.GetDouble();
   int betaVal = beta.GetInt();
   int pVal = p.GetInt();
   cv::Scalar lambScalar = cv::Scalar(lambda, lambda, lambda);

   int height = pSrc->GetHeight();
   int width = pSrc->GetWidth();

   cv::Mat I_RGB(height, width, CV_64FC3);
   fillRGB(I_RGB, pSourceData, height, width);

   PatchesMatrix patches = CutIntoPatches(I_RGB, height, width, kernelSize.GetInt());
   PatchesMatrix ls;
   vector<double> distances;

   double bottomSum = 0;
   for (auto patchRow:patches) {
      std::vector<cv::Mat> lsrow;
      for (auto patch:patchRow) {
         auto l = MeanIntensityL(patch, lamb);
         
         auto c = SignalStructureS(patch);
         auto s = SignalStructureS(patch);
         distances.push_back(cv::norm(patch-l));
         bottomSum += pow(cv::norm(patch-l), pVal);
         lsrow.push_back(l);
      }
      ls.push_back(lsrow);
   }

   double maxDist = *std::max_element(distances.begin(), distances.end());
   std::vector<std::vector<cv::Scalar>> gammas;
   std::vector<std::vector<cv::Scalar>> betas;
   for (int r = 0; r<ls.size(); r++) {
      std::vector<cv::Mat> patchesrow;
      std::vector<cv::Scalar> betasRow;
      std::vector<cv::Scalar> gammasRow;
      for (int col = 0; col<ls[0].size(); col++) {
         auto patch = patches[r][col];
         auto l = ls[r][col];
         auto gamma = Gamma(patch, l, lamb, maxDist, pVal, bottomSum);
         auto beta = CalculateBeta(l, betaVal);
         betasRow.push_back(beta);
         gammasRow.push_back(gamma);
      }
      gammas.push_back(gammasRow);
      betas.push_back(betasRow);
   }
   normalize(betas);

   std::vector<std::vector<cv::Mat>> transpatches;
   for (int r = 0; r<gammas.size(); r++) {
      std::vector<cv::Mat> patchesrow;
      for (int col = 0; col<gammas[0].size(); col++) {

         auto patch = patches[r][col];
         auto gamma = gammas[r][col];
         auto beta = betas[r][col];
         auto l = ls[r][col];
         cv::Mat transPatch = gamma*(patch-l)+beta*l;

         patchesrow.push_back(transPatch);
      }
      transpatches.push_back(patchesrow);
   }
   
   cv::Mat merged = mergeTiles(transpatches);

   for (int j = 0; j<pSrc->GetHeight(); j++)
   {
      for (int i = 0; i<pSrc->GetWidth(); i++)
      {
         /** store results to the destination image */
         *pDestinationData++ = merged.at<cv::Vec3f>(j, i)[0];
         *pDestinationData++ = merged.at<cv::Vec3f>(j, i)[1];
         *pDestinationData++ = merged.at<cv::Vec3f>(j, i)[2];
      }
   }
   return 0;
}