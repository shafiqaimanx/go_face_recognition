package gofacerecognition

import (
	"os"
	"path/filepath"
	"runtime"
)

// ModelPaths holds the paths to all required model files
type ModelPaths struct {
	// ShapePredictor68 is the path to shape_predictor_68_face_landmarks.dat
	ShapePredictor68 string
	// ShapePredictor5 is the path to shape_predictor_5_face_landmarks.dat
	ShapePredictor5 string
	// FaceRecognitionModel is the path to dlib_face_recognition_resnet_model_v1.dat
	FaceRecognitionModel string
	// CNNFaceDetector is the path to mmod_human_face_detector.dat
	CNNFaceDetector string
}

// DefaultModelPaths returns ModelPaths with default model file names
// You need to set the directory where models are located
func DefaultModelPaths(modelDir string) ModelPaths {
	return ModelPaths{
		ShapePredictor68:     filepath.Join(modelDir, "shape_predictor_68_face_landmarks.dat"),
		ShapePredictor5:      filepath.Join(modelDir, "shape_predictor_5_face_landmarks.dat"),
		FaceRecognitionModel: filepath.Join(modelDir, "dlib_face_recognition_resnet_model_v1.dat"),
		CNNFaceDetector:      filepath.Join(modelDir, "mmod_human_face_detector.dat"),
	}
}

// Validate checks if all model files exist
func (m ModelPaths) Validate() error {
	files := map[string]string{
		"shape_predictor_68":     m.ShapePredictor68,
		"shape_predictor_5":      m.ShapePredictor5,
		"face_recognition_model": m.FaceRecognitionModel,
		"cnn_face_detector":      m.CNNFaceDetector,
	}

	for name, path := range files {
		if path == "" {
			continue // Optional model
		}
		if _, err := os.Stat(path); os.IsNotExist(err) {
			return &ModelNotFoundError{ModelName: name, Path: path}
		}
	}

	return nil
}

// ValidateRequired checks if the required model files exist (excludes CNN detector)
func (m ModelPaths) ValidateRequired() error {
	files := map[string]string{
		"shape_predictor_68":     m.ShapePredictor68,
		"face_recognition_model": m.FaceRecognitionModel,
	}

	for name, path := range files {
		if path == "" {
			return &ModelNotFoundError{ModelName: name, Path: "not specified"}
		}
		if _, err := os.Stat(path); os.IsNotExist(err) {
			return &ModelNotFoundError{ModelName: name, Path: path}
		}
	}

	return nil
}

// EmbeddedModelsDir returns the path to the models directory bundled with this package.
// This uses runtime.Caller to find the package location.
// Deprecated: Use DefaultModelsDir() instead for the standard models location.
func EmbeddedModelsDir() string {
	_, filename, _, ok := runtime.Caller(0)
	if !ok {
		return ""
	}
	return filepath.Join(filepath.Dir(filename), "models")
}
