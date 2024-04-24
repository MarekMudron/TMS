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
   //fprintf(stderr, "exposure mean %f\n", meanValue[0]);
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
   // fprintf(stderr, "exposure %f\n", centerIntensity);
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

std::vector<cv::Mat> convertHDRtoLDR(const cv::Mat& hdrImage, int K) {
   double minExposure = 0.125;  // This might simulate an underexposure
   double maxExposure = 8.0;    // This might simulate an overexposure
   double exposureRange = maxExposure/minExposure;

   // Compute the exposure increment factor logarithmically
   double exposureIncrement = std::pow(exposureRange, 1.0/(K-1));

   double currentExposure = minExposure;
   std::vector<cv::Mat> exposureImages;
   for (int i = 0; i<K; ++i) {
      cv::Mat exposedImage;
      // Apply the exposure adjustment
      hdrImage.convertTo(exposedImage, CV_64F, currentExposure);
      exposureImages.push_back(exposedImage);
      currentExposure *= exposureIncrement;
   }
   return exposureImages;
}

std::vector<std::vector<std::vector<cv::Mat>>> transposePatches(const std::vector<PatchesMatrix>& images) {
   if (images.empty()) return {};

   // Determine the size of the grid of patches from the first image
   int numRows = images[0].size();
   int numCols = images[0].empty() ? 0 : images[0][0].size();

   // Prepare the output structure with the required dimensions
   std::vector<std::vector<std::vector<cv::Mat>>> transposed(numRows, std::vector<std::vector<cv::Mat>>(numCols));

   // Populate the transposed structure
   for (const auto& image:images) { // Loop over each image
      for (int i = 0; i<numRows; ++i) { // Loop over each row of patches
         for (int j = 0; j<numCols; ++j) { // Loop over each column of patches
            transposed[i][j].push_back(image[i][j]); // Add the patch at (i, j) to the list of patches at the same position across all images
         }
      }
   }

   return transposed;
}

std::vector<double> normalizeVector(const std::vector<double>& input) {
   std::vector<double> normalized;
   if (input.empty()) return normalized;  // Return empty vector if input is empty

   double sum = std::accumulate(input.begin(), input.end(), 0.0);  // Calculate the sum of all elements
   if (sum==0) return normalized;  // If sum is 0, return empty vector to avoid division by zero

   normalized.reserve(input.size());  // Reserve memory to avoid reallocation

   // Divide each element by the sum and store in the normalized vector
   for (double num : input) {
      normalized.push_back(num/sum);
   }

   return normalized;
}

int TMOLi21::Transform()
{
   double* pSourceData = pSrc->GetData();
   double* pDestinationData = pDst->GetData();

   pSrc->Convert(TMO_Yxy);
   pDst->Convert(TMO_Yxy);

   int kernelSizeValue = kernelSize.GetInt();
   double lamb = lambda.GetDouble();
   int betaVal = beta.GetInt();
   int pVal = p.GetInt();

   int height = pSrc->GetHeight();
   int width = pSrc->GetWidth();

   cv::Mat Y;
   cv::Mat x;
   cv::Mat y;

   Y = cv::Mat::zeros(height, width, CV_64F);
   x = cv::Mat::zeros(height, width, CV_64F);
   y = cv::Mat::zeros(height, width, CV_64F);

   int K = 3;

   int j;
   for (j = 0; j<height; j++)
   {
      for (int i = 0; i<width; i++)
      {
         Y.at<double>(j, i) = *pSourceData++;
         x.at<double>(j, i) = *pSourceData++; /** getting separate RGB channels */
         y.at<double>(j, i) = *pSourceData++;
      }
   }
   auto ldrs = convertHDRtoLDR(Y, K);
   vector<PatchesMatrix> patchesMatrices;
   for (const auto& mat:ldrs) {
      PatchesMatrix patches = CutIntoPatches(mat, height, width, kernelSizeValue);
      patchesMatrices.push_back(patches);
   }
   vector<vector<vector<cv::Mat>>> transposedPatches = transposePatches(patchesMatrices);

   PatchesMatrix transPatches;
   for (auto patchRow:transposedPatches) {
      std::vector<cv::Mat> lsrow;
      std::vector<cv::Mat> transPatchesRow;
      for (auto patchesK:patchRow) {
         vector<double> distances;
         vector<double> cs;
         vector<cv::Mat> ss;
         vector<cv::Mat> ls;
         double bottomSum = 0;

         for (auto patch:patchesK) {
            auto l = MeanIntensityL(patch, lamb);
            auto s = SignalStructureS(patch);
            auto c = SignalStrengthC(patch, lamb);
            distances.push_back(cv::norm(patch-l));
            bottomSum += pow(cv::norm(patch-l), pVal);
            ls.push_back(l);
            cs.push_back(c);
            ss.push_back(s);
         }
         double maxDist = *std::max_element(distances.begin(), distances.end());
         std::vector<double> gammas;
         std::vector<double> betas;

         for (int kI = 0; kI<patchesK.size();kI++) {
            auto patch = patchesK[kI];
            auto l = ls[kI];
            auto gamma = Gamma(patch, l, lamb, maxDist, pVal, bottomSum);
            auto beta = CalculateBeta(l, betaVal);
            gammas.push_back(gamma);
            betas.push_back(beta);
         }
         betas = normalizeVector(betas);

         cv::Mat transPatch = cv::Mat::zeros(kernelSizeValue, kernelSizeValue, CV_64F);
         for (int kI = 0; kI<patchesK.size();kI++) {
            auto patch = patchesK[kI];
            auto gamma = gammas[kI];
            auto beta = betas[kI];
            auto l = ls[kI];
            transPatch += gamma*(patch-l)+beta*l;
         }
         transPatchesRow.push_back(transPatch);
      }
      transPatches.push_back(transPatchesRow);
   }

   cv::Mat merged = mergeTiles(transPatches);
   for (int j = 0; j<height; j++)
   {
      for (int i = 0; i<width; i++)
      {													// simple variables
         //fprintf(stderr, "final %f\n", merged.at<double>(j, i));

         *pDestinationData++ = merged.at<double>(j, i); // + (detailChan[2]).at<float>(j,i)) / 256.0;
         *pDestinationData++ = x.at<double>(j, i);		// + (detailChan[1]).at<float>(j,i)) / 256.0;
         *pDestinationData++ = y.at<double>(j, i);
      }
   }
   pDst->Convert(TMO_RGB);
   return 0;
}