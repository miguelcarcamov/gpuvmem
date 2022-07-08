#ifndef IMAGE_CUH
#define IMAGE_CUH

typedef struct functionMap {
  void (*newP)(float*, float*, float, int);
  void (*evaluateXt)(float*, float*, float*, float, int);
} imageMap;

class Image {
 public:
  Image(float* image, int image_count) {
    this->image = image;
    this->image_count = image_count;
  };
  int getImageCount() { return image_count; };
  float* getImage() { return image; };
  float* getErrorImage() { return error_image; };
  imageMap* getFunctionMapping() { return functionMapping; };
  void setImageCount(int i) { this->image_count = i; };
  void setErrorImage(float* f) { this->error_image = f; };
  void setImage(float* i) { this->image = i; };
  void setFunctionMapping(imageMap* f) { this->functionMapping = f; };

 private:
  int image_count;
  float* image;
  float* error_image;
  imageMap* functionMapping;
};

#endif
