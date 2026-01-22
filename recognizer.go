package gofacerecognition

/*
#cgo CXXFLAGS: -std=c++14 -Wall -O3 -DNDEBUG
#cgo LDFLAGS: -ldlib -lstdc++ -lm

#cgo linux LDFLAGS: -lopenblas -llapack -ljpeg -lpng
#cgo windows LDFLAGS: -static

#include <stdlib.h>
#include <stdint.h>
#include "facerec.h"
*/
import "C"
import (
	"path/filepath"
	"sync"
	"unsafe"
)

// FaceRecognizer is the main struct for face recognition operations
type FaceRecognizer struct {
	rec         C.facerec
	modelPaths  ModelPaths
	initialized bool
	mu          sync.RWMutex
}

// NewFaceRecognizer creates a new FaceRecognizer with the given configuration
func NewFaceRecognizer(config Config) (*FaceRecognizer, error) {
	if err := config.ModelPaths.ValidateRequired(); err != nil {
		return nil, err
	}

	fr := &FaceRecognizer{
		modelPaths: config.ModelPaths,
	}

	// Get model directory
	modelDir := filepath.Dir(config.ModelPaths.ShapePredictor68)
	if modelDir == "" {
		modelDir = "."
	}
	// Ensure trailing separator for C code
	modelDir = filepath.ToSlash(modelDir) + "/"

	cModelDir := C.CString(modelDir)
	defer C.free(unsafe.Pointer(cModelDir))

	fr.rec = C.facerec_init(cModelDir)

	errStr := C.facerec_get_error(fr.rec)
	if errStr != nil {
		defer C.facerec_free_error(errStr)
		return nil, &ModelNotFoundError{
			ModelName: "dlib models",
			Path:      C.GoString(errStr),
		}
	}

	fr.initialized = true
	return fr, nil
}

// Close releases resources held by the FaceRecognizer
func (fr *FaceRecognizer) Close() {
	fr.mu.Lock()
	defer fr.mu.Unlock()

	if fr.initialized {
		C.facerec_free(fr.rec)
		fr.initialized = false
	}
}

// FaceLocations detects faces in an image and returns their bounding boxes
func (fr *FaceRecognizer) FaceLocations(img *ImageMatrix, upsampleTimes int, model DetectionModel) ([]Rectangle, error) {
	fr.mu.RLock()
	defer fr.mu.RUnlock()

	if !fr.initialized {
		return nil, &RecognizerNotInitializedError{}
	}

	if upsampleTimes < 1 {
		upsampleTimes = 1
	}

	// Convert image to C format
	cImg := imageMatrixToC(img)
	defer C.free(unsafe.Pointer(cImg.data))

	useCNN := 0
	if model == CNN {
		useCNN = 1
	}

	// Call C function
	var numFaces C.int
	cRects := C.facerec_detect(fr.rec, cImg, C.int(upsampleTimes), C.int(useCNN), &numFaces)

	if numFaces == 0 {
		return []Rectangle{}, nil
	}

	defer C.free(unsafe.Pointer(cRects))

	// Convert results
	rects := make([]Rectangle, int(numFaces))
	cRectsSlice := (*[1 << 28]C.rect)(unsafe.Pointer(cRects))[:numFaces:numFaces]

	for i, r := range cRectsSlice {
		rects[i] = Rectangle{
			Top:    int(r.top),
			Right:  int(r.right),
			Bottom: int(r.bottom),
			Left:   int(r.left),
		}
		// Trim to image bounds
		rects[i] = trimRectToBounds(rects[i], img.Height, img.Width)
	}

	return rects, nil
}

// FaceLandmarksDetect detects facial landmarks for faces in an image
func (fr *FaceRecognizer) FaceLandmarksDetect(img *ImageMatrix, faceLocations []Rectangle, model LandmarkModel) ([]RawLandmarks, error) {
	fr.mu.RLock()
	defer fr.mu.RUnlock()

	if !fr.initialized {
		return nil, &RecognizerNotInitializedError{}
	}

	// If no face locations provided, detect them first
	if faceLocations == nil {
		fr.mu.RUnlock()
		var err error
		faceLocations, err = fr.FaceLocations(img, 1, HOG)
		fr.mu.RLock()
		if err != nil {
			return nil, err
		}
	}

	if len(faceLocations) == 0 {
		return []RawLandmarks{}, nil
	}

	// Convert image to C format
	cImg := imageMatrixToC(img)
	defer C.free(unsafe.Pointer(cImg.data))

	// Convert face locations
	cRects := make([]C.rect, len(faceLocations))
	for i, r := range faceLocations {
		cRects[i] = C.rect{
			left:   C.long(r.Left),
			top:    C.long(r.Top),
			right:  C.long(r.Right),
			bottom: C.long(r.Bottom),
		}
	}

	useSmall := 0
	if model == LandmarkSmall {
		useSmall = 1
	}

	numPoints := 68
	if useSmall == 1 {
		numPoints = 5
	}

	// Call C function
	cLandmarks := C.facerec_landmarks(
		fr.rec,
		cImg,
		&cRects[0],
		C.int(len(faceLocations)),
		C.int(useSmall),
	)

	if cLandmarks == nil {
		return []RawLandmarks{}, nil
	}
	defer C.free(unsafe.Pointer(cLandmarks))

	// Convert results
	landmarks := make([]RawLandmarks, len(faceLocations))
	cLandmarksSlice := (*[1 << 28]C.point)(unsafe.Pointer(cLandmarks))[:len(faceLocations)*numPoints : len(faceLocations)*numPoints]

	for i := 0; i < len(faceLocations); i++ {
		landmarks[i].Points = make([]Point, numPoints)
		for j := 0; j < numPoints; j++ {
			idx := i*numPoints + j
			landmarks[i].Points[j] = Point{
				X: int(cLandmarksSlice[idx].x),
				Y: int(cLandmarksSlice[idx].y),
			}
		}
	}

	return landmarks, nil
}

// FaceLandmarks returns structured facial landmarks for the "large" model
func (fr *FaceRecognizer) FaceLandmarks(img *ImageMatrix, faceLocations []Rectangle) ([]FaceLandmarks, error) {
	raw, err := fr.FaceLandmarksDetect(img, faceLocations, LandmarkLarge)
	if err != nil {
		return nil, err
	}

	landmarks := make([]FaceLandmarks, len(raw))
	for i, r := range raw {
		if len(r.Points) < 68 {
			continue
		}
		landmarks[i] = FaceLandmarks{
			Chin:         r.Points[0:17],
			LeftEyebrow:  r.Points[17:22],
			RightEyebrow: r.Points[22:27],
			NoseBridge:   r.Points[27:31],
			NoseTip:      r.Points[31:36],
			LeftEye:      r.Points[36:42],
			RightEye:     r.Points[42:48],
			TopLip:       append(r.Points[48:55], r.Points[64], r.Points[63], r.Points[62], r.Points[61], r.Points[60]),
			BottomLip:    append(append(r.Points[54:60], r.Points[48]), r.Points[60], r.Points[67], r.Points[66], r.Points[65], r.Points[64]),
		}
	}

	return landmarks, nil
}

// FaceLandmarksSmallModel returns structured facial landmarks for the "small" model
func (fr *FaceRecognizer) FaceLandmarksSmallModel(img *ImageMatrix, faceLocations []Rectangle) ([]FaceLandmarksSmall, error) {
	raw, err := fr.FaceLandmarksDetect(img, faceLocations, LandmarkSmall)
	if err != nil {
		return nil, err
	}

	landmarks := make([]FaceLandmarksSmall, len(raw))
	for i, r := range raw {
		if len(r.Points) < 5 {
			continue
		}
		landmarks[i] = FaceLandmarksSmall{
			NoseTip:  []Point{r.Points[4]},
			LeftEye:  r.Points[2:4],
			RightEye: r.Points[0:2],
		}
	}

	return landmarks, nil
}

// FaceEncodings computes 128-dimensional face encodings for faces in an image
func (fr *FaceRecognizer) FaceEncodings(img *ImageMatrix, faceLocations []Rectangle, numJitters int, model LandmarkModel) ([]FaceEncoding, error) {
	fr.mu.RLock()
	defer fr.mu.RUnlock()

	if !fr.initialized {
		return nil, &RecognizerNotInitializedError{}
	}

	if numJitters < 1 {
		numJitters = 1
	}

	// Get landmarks first (need to unlock for this call)
	fr.mu.RUnlock()
	raw, err := fr.FaceLandmarksDetect(img, faceLocations, model)
	fr.mu.RLock()
	if err != nil {
		return nil, err
	}

	if len(raw) == 0 {
		return []FaceEncoding{}, nil
	}

	// Convert image to C format
	cImg := imageMatrixToC(img)
	defer C.free(unsafe.Pointer(cImg.data))

	numPoints := 68
	if model == LandmarkSmall {
		numPoints = 5
	}

	// Flatten landmarks for C
	cPoints := make([]C.point, len(raw)*numPoints)
	for i, r := range raw {
		for j, p := range r.Points {
			cPoints[i*numPoints+j] = C.point{
				x: C.long(p.X),
				y: C.long(p.Y),
			}
		}
	}

	// Call C function
	cEncodings := C.facerec_encode(
		fr.rec,
		cImg,
		&cPoints[0],
		C.int(len(raw)),
		C.int(numPoints),
		C.int(numJitters),
	)

	if cEncodings == nil {
		return []FaceEncoding{}, nil
	}
	defer C.free(unsafe.Pointer(cEncodings))

	// Convert results
	encodings := make([]FaceEncoding, len(raw))
	cEncodingsSlice := (*[1 << 28]C.double)(unsafe.Pointer(cEncodings))[:len(raw)*128 : len(raw)*128]

	for i := 0; i < len(raw); i++ {
		for j := 0; j < 128; j++ {
			encodings[i][j] = float64(cEncodingsSlice[i*128+j])
		}
	}

	return encodings, nil
}

// DetectAndEncode detects faces and computes encodings in one call
func (fr *FaceRecognizer) DetectAndEncode(img *ImageMatrix, upsampleTimes int, numJitters int) ([]Face, error) {
	locations, err := fr.FaceLocations(img, upsampleTimes, HOG)
	if err != nil {
		return nil, err
	}

	landmarks, err := fr.FaceLandmarks(img, locations)
	if err != nil {
		return nil, err
	}

	encodings, err := fr.FaceEncodings(img, locations, numJitters, LandmarkLarge)
	if err != nil {
		return nil, err
	}

	faces := make([]Face, len(locations))
	for i := range locations {
		faces[i] = Face{
			Rectangle: locations[i],
			Encoding:  encodings[i],
		}
		if i < len(landmarks) {
			faces[i].Landmarks = landmarks[i]
		}
	}

	return faces, nil
}

// Helper functions

func trimRectToBounds(rect Rectangle, height, width int) Rectangle {
	return Rectangle{
		Top:    max(rect.Top, 0),
		Right:  min(rect.Right, width),
		Bottom: min(rect.Bottom, height),
		Left:   max(rect.Left, 0),
	}
}

// C helper types and conversions (these match facerec.h)
func imageMatrixToC(img *ImageMatrix) C.image {
	cData := C.CBytes(img.Pixels)
	return C.image{
		data:   (*C.uint8_t)(cData),
		width:  C.int(img.Width),
		height: C.int(img.Height),
		stride: C.int(img.Stride),
	}
}
