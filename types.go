package gofacerecognition

// Rectangle represents a face bounding box in CSS order (top, right, bottom, left)
type Rectangle struct {
	Top    int
	Right  int
	Bottom int
	Left   int
}

// Width returns the width of the rectangle
func (r Rectangle) Width() int {
	return r.Right - r.Left
}

// Height returns the height of the rectangle
func (r Rectangle) Height() int {
	return r.Bottom - r.Top
}

// Point represents a 2D point (x, y)
type Point struct {
	X int
	Y int
}

// FaceEncoding represents a 128-dimensional face encoding vector
type FaceEncoding [128]float64

// FaceLandmarks represents facial landmarks for the "large" model (68 points)
type FaceLandmarks struct {
	Chin         []Point // 17 points
	LeftEyebrow  []Point // 5 points
	RightEyebrow []Point // 5 points
	NoseBridge   []Point // 4 points
	NoseTip      []Point // 5 points
	LeftEye      []Point // 6 points
	RightEye     []Point // 6 points
	TopLip       []Point // 12 points
	BottomLip    []Point // 12 points
}

// FaceLandmarksSmall represents facial landmarks for the "small" model (5 points)
type FaceLandmarksSmall struct {
	NoseTip  []Point // 1 point
	LeftEye  []Point // 2 points
	RightEye []Point // 2 points
}

// DetectionModel specifies the face detection model to use
type DetectionModel string

const (
	// HOG is the Histogram of Oriented Gradients model (faster, less accurate)
	HOG DetectionModel = "hog"
	// CNN is the Convolutional Neural Network model (slower, more accurate, GPU accelerated)
	CNN DetectionModel = "cnn"
)

// LandmarkModel specifies the face landmark model to use
type LandmarkModel string

const (
	// LandmarkLarge uses the 68-point model
	LandmarkLarge LandmarkModel = "large"
	// LandmarkSmall uses the 5-point model (faster)
	LandmarkSmall LandmarkModel = "small"
)

// Face represents a detected face with all its properties
type Face struct {
	Rectangle Rectangle
	Landmarks interface{} // FaceLandmarks or FaceLandmarksSmall
	Encoding  FaceEncoding
}

// RawLandmarks represents raw landmark points before conversion
type RawLandmarks struct {
	Points []Point
}
