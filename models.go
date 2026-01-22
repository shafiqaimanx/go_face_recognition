package gofacerecognition

import (
	"os"
	"path/filepath"
)

type ModelPaths struct {
	ShapePredictor68     string // shape_predictor_68_face_landmarks.dat
	ShapePredictor5      string // shape_predictor_5_face_landmarks.dat
	FaceRecognitionModel string // dlib_face_recognition_resnet_model_v1.dat
	CNNFaceDetector      string // mmod_human_face_detector.dat
}

func DefaultModelPaths(modeldir string) ModelPaths {
	return ModelPaths{
		ShapePredictor68:     filepath.Join(modeldir, "shape_predictor_68_face_landmarks.dat"),
		ShapePredictor5:      filepath.Join(modeldir, "shape_predictor_5_face_landmarks.dat"),
		FaceRecognitionModel: filepath.Join(modeldir, "dlib_face_recognition_resnet_model_v1.dat"),
		CNNFaceDetector:      filepath.Join(modeldir, "mmod_human_face_detector.dat"),
	}
}

// Validate: Checks if all model files exist
func (m ModelPaths) Validate() error {
	files := map[string]string{
		"shape_predictor_68":     m.ShapePredictor68,
		"shape_predictor_5":      m.ShapePredictor5,
		"face_recognition_model": m.FaceRecognitionModel,
		"cnn_face_detector":      m.CNNFaceDetector,
	}

	for name, path := range files {
		if _, err := os.Stat(path); os.IsNotExist(err) {
			return &ModelNotFoundError{
				ModelName: name,
				Path:      path,
			}
		}
	}

	return nil
}

func (m ModelPaths) ValidateRequired() error {
	files := map[string]string{
		"shape_predictor_68":     m.ShapePredictor68,
		"face_recognition_model": m.FaceRecognitionModel,
	}

	for name, path := range files {
		if _, err := os.Stat(path); os.IsNotExist(err) {
			return &ModelNotFoundError{
				ModelName: name,
				Path: path,
			}
		}
	}

	return nil
}
