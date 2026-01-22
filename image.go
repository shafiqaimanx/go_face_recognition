package gofacerecognition

import (
	"image"
	"image/color"
	_ "image/gif"
	_ "image/jpeg"
	_ "image/png"
	"os"

	_ "golang.org/x/image/bmp"
	_ "golang.org/x/image/webp"
)

// ImageMatrix represents an image as a 3D matrix (height x width x channels)
// This is similar to numpy array representation used in the Python version
type ImageMatrix struct {
	Pixels []byte
	Width  int
	Height int
	// Stride is the number of bytes per row
	Stride int
}

// NewImageMatrix creates an ImageMatrix from width and height
func NewImageMatrix(width, height int) *ImageMatrix {
	stride := width * 3 // RGB
	return &ImageMatrix{
		Pixels: make([]byte, height*stride),
		Width:  width,
		Height: height,
		Stride: stride,
	}
}

// Shape returns the image shape as (height, width, channels)
func (im *ImageMatrix) Shape() (int, int, int) {
	return im.Height, im.Width, 3
}

// At returns the RGB values at position (x, y)
func (im *ImageMatrix) At(x, y int) (r, g, b byte) {
	offset := y*im.Stride + x*3
	return im.Pixels[offset], im.Pixels[offset+1], im.Pixels[offset+2]
}

// Set sets the RGB values at position (x, y)
func (im *ImageMatrix) Set(x, y int, r, g, b byte) {
	offset := y*im.Stride + x*3
	im.Pixels[offset] = r
	im.Pixels[offset+1] = g
	im.Pixels[offset+2] = b
}

// LoadImageFile loads an image file and converts it to RGB format
// Supports: JPEG, PNG, GIF, BMP, WebP
func LoadImageFile(path string) (*ImageMatrix, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, &ImageLoadError{Path: path, Err: err}
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, &ImageLoadError{Path: path, Err: err}
	}

	return ImageToMatrix(img), nil
}

// LoadImageFileGrayscale loads an image file and converts it to grayscale
func LoadImageFileGrayscale(path string) (*ImageMatrix, error) {
	file, err := os.Open(path)
	if err != nil {
		return nil, &ImageLoadError{Path: path, Err: err}
	}
	defer file.Close()

	img, _, err := image.Decode(file)
	if err != nil {
		return nil, &ImageLoadError{Path: path, Err: err}
	}

	return ImageToGrayscaleMatrix(img), nil
}

// ImageToMatrix converts a Go image.Image to ImageMatrix (RGB format)
func ImageToMatrix(img image.Image) *ImageMatrix {
	bounds := img.Bounds()
	width := bounds.Max.X - bounds.Min.X
	height := bounds.Max.Y - bounds.Min.Y

	matrix := NewImageMatrix(width, height)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b, _ := img.At(x+bounds.Min.X, y+bounds.Min.Y).RGBA()
			// Convert from 16-bit to 8-bit
			matrix.Set(x, y, byte(r>>8), byte(g>>8), byte(b>>8))
		}
	}

	return matrix
}

// ImageToGrayscaleMatrix converts a Go image.Image to grayscale ImageMatrix
func ImageToGrayscaleMatrix(img image.Image) *ImageMatrix {
	bounds := img.Bounds()
	width := bounds.Max.X - bounds.Min.X
	height := bounds.Max.Y - bounds.Min.Y

	matrix := NewImageMatrix(width, height)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			gray := color.GrayModel.Convert(img.At(x+bounds.Min.X, y+bounds.Min.Y)).(color.Gray)
			matrix.Set(x, y, gray.Y, gray.Y, gray.Y)
		}
	}

	return matrix
}

// ToGoImage converts ImageMatrix back to image.Image
func (im *ImageMatrix) ToGoImage() image.Image {
	img := image.NewRGBA(image.Rect(0, 0, im.Width, im.Height))
	for y := 0; y < im.Height; y++ {
		for x := 0; x < im.Width; x++ {
			r, g, b := im.At(x, y)
			img.Set(x, y, color.RGBA{R: r, G: g, B: b, A: 255})
		}
	}
	return img
}

// Crop returns a cropped portion of the image based on a Rectangle
func (im *ImageMatrix) Crop(rect Rectangle) *ImageMatrix {
	// Clamp values to image bounds
	left := max(0, rect.Left)
	top := max(0, rect.Top)
	right := min(im.Width, rect.Right)
	bottom := min(im.Height, rect.Bottom)

	width := right - left
	height := bottom - top

	if width <= 0 || height <= 0 {
		return NewImageMatrix(0, 0)
	}

	cropped := NewImageMatrix(width, height)

	for y := 0; y < height; y++ {
		for x := 0; x < width; x++ {
			r, g, b := im.At(x+left, y+top)
			cropped.Set(x, y, r, g, b)
		}
	}

	return cropped
}
