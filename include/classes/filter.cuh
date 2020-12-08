#ifndef FILTER_CUH
#define FILTER_CUH

class Filter
{
public:
    virtual void applyCriteria(Visibilities *v) = 0;
    virtual void configure(void *params) = 0;
};

#endif //FILTER_CUH
