package gofacerecognition

import "fmt"

// ModelNotFoundError is returned when a required model file is not found
type ModelNotFoundError struct {
	ModelName string
	Path      string
}

func (e *ModelNotFoundError) Error() string {
	return fmt.Sprintf("model '%s' not found at path: %s", e.ModelName, e.Path)
}

// ImageLoadError is returned when an image cannot be loaded
type ImageLoadError struct {
	Path string
	Err  error
}

func (e *ImageLoadError) Error() string {
	return fmt.Sprintf("failed to load image '%s': %v", e.Path, e.Err)
}

func (e *ImageLoadError) Unwrap() error {
	return e.Err
}

// NoFaceFoundError is returned when no face is found in an image
type NoFaceFoundError struct{}

func (e *NoFaceFoundError) Error() string {
	return "no face found in image"
}

// InvalidModelError is returned when an invalid model type is specified
type InvalidModelError struct {
	Model string
	Valid []string
}

func (e *InvalidModelError) Error() string {
	return fmt.Sprintf("invalid model '%s', valid options are: %v", e.Model, e.Valid)
}

// RecognizerNotInitializedError is returned when the recognizer is used before initialization
type RecognizerNotInitializedError struct{}

func (e *RecognizerNotInitializedError) Error() string {
	return "face recognizer not initialized, call Init() first"
}
