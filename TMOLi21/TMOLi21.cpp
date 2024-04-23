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
   kernelSize.SetDefault(10);
   kernelSize = 10;
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

// OK
double CalculateA(cv::Mat patch, double lambda) {
   cv::Scalar meanValue, stdDev;
   cv::meanStdDev(patch, meanValue, stdDev);
   fprintf(stderr, "exposure mean %f, deviation %f\n", meanValue[0], stdDev[0]);
   return (stdDev[0]*stdDev[0])/(stdDev[0]*stdDev[0]+lambda);
}

// OK

double CalculateB(cv::Mat patch, double lambda) {
   cv::Scalar mean, stdDev;
   cv::meanStdDev(patch, mean, stdDev);
   return mean[0]*lambda/(stdDev[0]*stdDev[0]+lambda);
}

// OK
cv::Mat MeanIntensityL(cv::Mat patch, double lambda) {
   double a = CalculateA(patch, lambda);
   double b = CalculateB(patch, lambda);
   return a*patch+b*cv::Mat::ones(patch.size(), patch.type());
}

// OK
cv::Mat SignalStructureS(cv::Mat patch) {
   cv::Scalar mean, stdDev;
   cv::meanStdDev(patch, mean, stdDev);
   cv::Mat mxMat(patch.size(), patch.type());
   mxMat.setTo(mean[0]);
   cv::Mat result;
   cv::divide(patch-mxMat, cv::norm(patch-mxMat), result);
   return result;
}

// OK
double SignalStrengthC(cv::Mat patch, double lambda) {
   cv::Scalar mean, stdDev;
   cv::meanStdDev(patch, mean, stdDev);
   cv::Mat mxMat(patch.size(), patch.type());
   mxMat.setTo(mean[0]);
   return (lambda*cv::norm(patch-mxMat))/(lambda+stdDev[0]*stdDev[0]);
}
// OK
double Gamma(cv::Mat patch, cv::Mat l, double lambda, double maxDist, int p, double bottomSum) {
   cv::Scalar meanValue, stdDev;
   cv::meanStdDev(patch, meanValue, stdDev);
   auto first = (lambda*maxDist)/(stdDev[0]*stdDev[0]+lambda);
   auto second = pow(cv::norm(patch-l), p-1)/bottomSum;
   return first*second;
}

using PatchesMatrix = vector<vector<cv::Mat>>;

// OK
PatchesMatrix CutIntoPatches(const cv::Mat& image, int height, int width, int kernelSize) {
   PatchesMatrix tiles;
   // Calculate the number of vertical and horizontal tiles
   int numVerticalTiles = (height+kernelSize-1)/kernelSize;
   int numHorizontalTiles = (width+kernelSize-1)/kernelSize;

   for (int i = 0; i<numVerticalTiles; ++i) {
      std::vector<cv::Mat> rowTiles;
      for (int j = 0; j<numHorizontalTiles; ++j) {
         int startY = i*kernelSize;
         int startX = j*kernelSize;
         int tileHeight = std::min(kernelSize, height-startY);
         int tileWidth = std::min(kernelSize, width-startX);

         // Extract the tile using cv::Rect and push it into the row vector
         cv::Rect roi(startX, startY, tileWidth, tileHeight);
         rowTiles.push_back(image(roi));
      }
      // Add the row of tiles to the tiles vector
      tiles.push_back(rowTiles);
   }
   return tiles;
}

// OK
cv::Mat normalizeMat(cv::Mat mat) {
   cv::Mat normalizedMat;
   double minVal, maxVal;
   cv::minMaxLoc(mat, &minVal, &maxVal); // Find minimum and maximum values
   cv::normalize(mat, normalizedMat, 0, 1, cv::NORM_MINMAX);
   return normalizedMat;
}

// OK
double CalculateBeta(const cv::Mat& fullIntensity, int beta) {
   int center_i = fullIntensity.rows/2;
   int center_j = fullIntensity.cols/2;

   //auto intensity = normalizeMat(fullIntensity);
   auto intensity = fullIntensity;
   auto centerIntensity = intensity.at<double>(center_i, center_j);
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
   return v;
}

void normalize(std::vector<std::vector<double>>& vec) {
   double sum = 0.0;

   // Calculate the sum of all elements
   for (const auto& row:vec) {
      for (double val : row) {
         sum += val;
      }
   }
   // Check for zero sum to avoid division by zero
   if (sum==0) return;

   // Normalize each element
   for (auto& row:vec) {
      for (double& val:row) {
         val /= sum;
      }
   }
}



cv::Mat mergeTiles(const std::vector<std::vector<cv::Mat>>& tiles) {
   if (tiles.empty()||tiles[0].empty()) return cv::Mat();

   int totalRows = 0;
   int totalCols = 0;
   int numRows = tiles.size();
   int numCols = tiles[0].size();

   // Calculate the total number of rows and columns in the merged image
   for (int i = 0; i<numRows; ++i) {
      totalRows += tiles[i][0].rows;
   }
   for (int j = 0; j<numCols; ++j) {
      totalCols += tiles[0][j].cols;
   }

   // Create a large matrix to store the merged image
   cv::Mat mergedImage(totalRows, totalCols, tiles[0][0].type());
   int startY = 0;

   // Copy each tile into the correct location in the large matrix
   for (int i = 0; i<numRows; ++i) {
      int startX = 0;
      int currentRowHeight = tiles[i][0].rows;

      for (int j = 0; j<numCols; ++j) {
         tiles[i][j].copyTo(mergedImage(cv::Rect(startX, startY, tiles[i][j].cols, currentRowHeight)));
         startX += tiles[i][j].cols;
      }
      startY += currentRowHeight;
   }

   return mergedImage;
}

cv::Mat TMOLi21::transformOneChannel(cv::Mat imageChannel) {
   int kernelSizeValue = kernelSize.GetInt();
   double lamb = lambda.GetDouble();
   int betaVal = beta.GetInt();
   int pVal = p.GetInt();

   int height = pSrc->GetHeight();
   int width = pSrc->GetWidth();
   PatchesMatrix patches = CutIntoPatches(imageChannel, height, width, kernelSizeValue);
   PatchesMatrix ls;

   double bottomSum = 0;
   vector<double> distances;

   for (auto patchRow:patches) {
      std::vector<cv::Mat> lsrow;
      for (auto patch:patchRow) {
         auto l = MeanIntensityL(patch, lamb);
         auto s = SignalStructureS(patch);
         auto c = SignalStrengthC(patch, lamb);
         distances.push_back(cv::norm(patch-l));
         bottomSum += pow(cv::norm(patch-l), pVal);
         lsrow.push_back(l);
      }
      ls.push_back(lsrow);
   }

   double maxDist = *std::max_element(distances.begin(), distances.end());
   std::vector<std::vector<double>> gammas;
   std::vector<std::vector<double>> betas;

   for (int r = 0; r<ls.size(); r++) {
      std::vector<double> betasRow;
      std::vector<double> gammasRow;
      for (int col = 0; col<ls[0].size(); col++) {
         auto patch = patches[r][col];
         auto l = ls[r][col];
         auto gamma = Gamma(patch, l, lamb, maxDist, pVal, bottomSum);
         auto beta = CalculateBeta(l, betaVal);
         //fprintf(stderr, "%f %f %f\n", beta, gamma, bottomSum);
         betasRow.push_back(beta);
         gammasRow.push_back(gamma);
      }
      gammas.push_back(gammasRow);
      betas.push_back(betasRow);
   }

   normalize(betas);

   PatchesMatrix transpatches;
   for (int r = 0; r<gammas.size(); r++) {
      std::vector<cv::Mat> patchesrow;
      for (int col = 0; col<gammas[0].size(); col++) {
         auto patch = patches[r][col];
         auto gamma = gammas[r][col];
         auto beta = betas[r][col];
         auto l = ls[r][col];

         cv::Mat transPatch = gamma*(patch-l)+beta*l;
         fprintf(stderr, "gamma: %f, beta: %f\n", gamma, beta);
         patchesrow.push_back(transPatch);
      }
      transpatches.push_back(patchesrow);
   }
   cv::Mat merged = mergeTiles(transpatches);
   return merged;
}


int TMOLi21::Transform()
{
   double* pSourceData = pSrc->GetData();
   double* pDestinationData = pDst->GetData();

   int kernelSizeValue = kernelSize.GetInt();

   int height = pSrc->GetHeight();
   int width = pSrc->GetWidth();

   cv::Mat R;
   cv::Mat G;
   cv::Mat B;

   R = cv::Mat::zeros(height, width, CV_64F);
   G = cv::Mat::zeros(height, width, CV_64F);
   B = cv::Mat::zeros(height, width, CV_64F);

   int j;
   for (j = 0; j<height; j++)
   {
      for (int i = 0; i<width; i++)
      {
         R.at<double>(j, i) = *pSourceData++;
         G.at<double>(j, i) = *pSourceData++;
         B.at<double>(j, i) = *pSourceData++;
         //fprintf(stderr, "%f\n",R.at<double>(j, i ));
      }
   }

   cv::Mat Rt;
   cv::Mat Gt;
   cv::Mat Bt;

   Rt = transformOneChannel(R);
   Gt = transformOneChannel(G);
   Bt = transformOneChannel(B);
   fprintf(stderr, "------------------------------");
   for (int j = 0; j<height; j++)
   {
      for (int i = 0; i<width; i++)
      {													// simple variables
         fprintf(stderr, "%f\n",Rt.at<double>(j, i));
         *pDestinationData++ = Rt.at<double>(j, i); // + (detailChan[2]).at<float>(j,i)) / 256.0;
         *pDestinationData++ = Gt.at<double>(j, i);		// + (detailChan[1]).at<float>(j,i)) / 256.0;
         *pDestinationData++ = Bt.at<double>(j, i);
      }
   }
   return 0;
}