#include "TMO.h"
#include "opencv2/core.hpp"
class TMOLi21 : public TMO
{
public:
	TMOLi21();
	virtual ~TMOLi21();
    virtual int Transform();

protected:
    TMOInt kernelSize;
    TMODouble lambda;

private:
    double SignalStrengthC(cv::Mat patch);
    cv::Mat SingalStructureS(cv::Mat patch);
    cv::Mat MeanIntensityL(cv::Mat patch);

    double CalcA(double var, double lambda);
    double CalcB(double var, double lambda, double mean);

    double CalculateVariance(const cv::Mat& patch);
    double CalculateMean(const cv::Mat& patch);

    double CalculateGamma();
    double CalculateBeta();
};
