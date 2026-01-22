#ifndef FACEREC_H
#define FACEREC_H

#ifdef __cplusplus
extern "C" {
#endif

#include <stdint.h>

// Opaque handle to the face recognizer
typedef void* facerec;

// Image structure for passing images from Go
typedef struct {
    uint8_t* data;
    int width;
    int height;
    int stride;
} image;

// Rectangle structure for face locations
typedef struct {
    long left;
    long top;
    long right;
    long bottom;
} rect;

// Point structure for landmarks
typedef struct {
    long x;
    long y;
} point;

// Initialize face recognizer with model directory
facerec facerec_init(const char* model_dir);

// Free resources
void facerec_free(facerec rec);

// Get last error message (NULL if no error)
const char* facerec_get_error(facerec rec);

// Free error string
void facerec_free_error(const char* err);

// Detect faces in an image
// Returns array of rectangles, sets num_faces to count
// use_cnn: 0 for HOG, 1 for CNN
rect* facerec_detect(facerec rec, image img, int upsample_times, int use_cnn, int* num_faces);

// Get facial landmarks for detected faces
// Returns array of points (num_faces * points_per_face)
// use_small: 0 for 68-point model, 1 for 5-point model
point* facerec_landmarks(facerec rec, image img, rect* faces, int num_faces, int use_small);

// Compute face encodings from landmarks
// Returns array of doubles (num_faces * 128)
double* facerec_encode(facerec rec, image img, point* landmarks, int num_faces, int points_per_face, int num_jitters);

#ifdef __cplusplus
}
#endif

#endif // FACEREC_H
