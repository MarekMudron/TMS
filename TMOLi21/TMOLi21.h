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
    TMOInt beta;
    TMOInt p;

};
