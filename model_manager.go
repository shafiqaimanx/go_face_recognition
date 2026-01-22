package gofacerecognition

import (
	"compress/bzip2"
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
)

// Model file URLs (from GitHub releases)
const (
	// GitHub releases URL base
	GitHubReleasesBase = "https://github.com/shafiqaimanx/go_face_recognition/releases/download/models/"

	ShapePredictor68URL = GitHubReleasesBase + "shape_predictor_68_face_landmarks.dat"
	ShapePredictor5URL  = GitHubReleasesBase + "shape_predictor_5_face_landmarks.dat"
	FaceRecognitionURL  = GitHubReleasesBase + "dlib_face_recognition_resnet_model_v1.dat"
)

// Model file names
const (
	ShapePredictor68File  = "shape_predictor_68_face_landmarks.dat"
	ShapePredictor5File   = "shape_predictor_5_face_landmarks.dat"
	FaceRecognitionFile   = "dlib_face_recognition_resnet_model_v1.dat"
)

// RequiredModels lists the models required for basic operation
var RequiredModels = []ModelInfo{
	{Name: ShapePredictor68File, URL: ShapePredictor68URL, Required: true},
	{Name: FaceRecognitionFile, URL: FaceRecognitionURL, Required: true},
}

// OptionalModels lists optional models
var OptionalModels = []ModelInfo{
	{Name: ShapePredictor5File, URL: ShapePredictor5URL, Required: false},
}

// AllModels combines required and optional models
var AllModels = append(RequiredModels, OptionalModels...)

// ModelInfo describes a model file
type ModelInfo struct {
	Name     string
	URL      string
	Required bool
}

// DefaultModelsDir returns the default directory for storing models
// Linux/macOS: ~/.goface_recognition/models/
// Windows: %USERPROFILE%\.goface_recognition\models\
func DefaultModelsDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		// Fallback to current directory
		return filepath.Join(".", ".goface_recognition", "models")
	}
	return filepath.Join(home, ".goface_recognition", "models")
}

// ModelsExist checks if all required models exist in the given directory
func ModelsExist(dir string) bool {
	for _, model := range RequiredModels {
		path := filepath.Join(dir, model.Name)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			return false
		}
	}
	return true
}

// ModelExists checks if a specific model exists
func ModelExists(dir, modelName string) bool {
	path := filepath.Join(dir, modelName)
	_, err := os.Stat(path)
	return err == nil
}

// ResolveModelsDir finds or creates a directory with models
// Priority:
// 1. User-specified path (if provided and models exist)
// 2. Default location (~/.goface_recognition/models/)
// 3. Auto-download to default location
func ResolveModelsDir(userPath string, autoDownload bool) (string, error) {
	// 1. Check user-specified path
	if userPath != "" {
		if ModelsExist(userPath) {
			return userPath, nil
		}
		// User specified a path but models don't exist
		if !autoDownload {
			return "", &ModelNotFoundError{
				ModelName: "required models",
				Path:      userPath,
			}
		}
		// Download to user-specified path
		if err := EnsureModels(userPath); err != nil {
			return "", err
		}
		return userPath, nil
	}

	// 2. Check default location
	defaultDir := DefaultModelsDir()
	if ModelsExist(defaultDir) {
		return defaultDir, nil
	}

	// 3. Auto-download if enabled
	if autoDownload {
		fmt.Printf("Models not found. Downloading to %s...\n", defaultDir)
		if err := EnsureModels(defaultDir); err != nil {
			return "", err
		}
		return defaultDir, nil
	}

	return "", &ModelNotFoundError{
		ModelName: "required models",
		Path:      defaultDir,
	}
}

// EnsureModels downloads any missing required models to the specified directory
func EnsureModels(dir string) error {
	// Create directory if needed
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create models directory: %w", err)
	}

	for _, model := range RequiredModels {
		if !ModelExists(dir, model.Name) {
			fmt.Printf("Downloading %s...\n", model.Name)
			if err := DownloadModel(model.URL, filepath.Join(dir, model.Name)); err != nil {
				return fmt.Errorf("failed to download %s: %w", model.Name, err)
			}
			fmt.Printf("Downloaded %s\n", model.Name)
		}
	}

	return nil
}

// EnsureAllModels downloads all models (required + optional)
func EnsureAllModels(dir string) error {
	if err := os.MkdirAll(dir, 0755); err != nil {
		return fmt.Errorf("failed to create models directory: %w", err)
	}

	for _, model := range AllModels {
		if !ModelExists(dir, model.Name) {
			fmt.Printf("Downloading %s...\n", model.Name)
			if err := DownloadModel(model.URL, filepath.Join(dir, model.Name)); err != nil {
				if model.Required {
					return fmt.Errorf("failed to download %s: %w", model.Name, err)
				}
				fmt.Printf("Warning: failed to download optional model %s: %v\n", model.Name, err)
				continue
			}
			fmt.Printf("Downloaded %s\n", model.Name)
		}
	}

	return nil
}

// DownloadModel downloads and extracts a model file
// Handles .bz2 compression automatically if URL ends with .bz2
func DownloadModel(url, destPath string) error {
	// Create HTTP client with redirect following
	client := &http.Client{
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return nil // Follow redirects
		},
	}

	// Create HTTP request
	resp, err := client.Get(url)
	if err != nil {
		return fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP status %d: %s", resp.StatusCode, resp.Status)
	}

	// Create destination file
	destFile, err := os.Create(destPath)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer destFile.Close()

	// Check if URL ends with .bz2 (for dlib.net fallback)
	var reader io.Reader = resp.Body
	if filepath.Ext(url) == ".bz2" {
		reader = bzip2.NewReader(resp.Body)
	}

	// Copy with progress indicator
	written, err := copyWithProgress(destFile, reader, resp.ContentLength)
	if err != nil {
		os.Remove(destPath) // Clean up partial file
		return fmt.Errorf("download failed: %w", err)
	}

	fmt.Printf("\nDownloaded %.2f MB\n", float64(written)/(1024*1024))
	return nil
}

// copyWithProgress copies data with a progress indicator
func copyWithProgress(dst io.Writer, src io.Reader, total int64) (int64, error) {
	var written int64
	buf := make([]byte, 32*1024) // 32KB buffer

	for {
		nr, er := src.Read(buf)
		if nr > 0 {
			nw, ew := dst.Write(buf[0:nr])
			if nw < 0 || nr < nw {
				nw = 0
				if ew == nil {
					ew = fmt.Errorf("invalid write result")
				}
			}
			written += int64(nw)
			if ew != nil {
				return written, ew
			}
			if nr != nw {
				return written, io.ErrShortWrite
			}

			// Print progress (only on terminals)
			if total > 0 && isTerminal() {
				pct := float64(written) / float64(total) * 100
				fmt.Printf("\r  Progress: %.1f%%", pct)
			}
		}
		if er != nil {
			if er != io.EOF {
				return written, er
			}
			break
		}
	}

	return written, nil
}

// isTerminal checks if stdout is a terminal
func isTerminal() bool {
	if runtime.GOOS == "windows" {
		return true // Assume terminal on Windows
	}
	fi, err := os.Stdout.Stat()
	if err != nil {
		return false
	}
	return (fi.Mode() & os.ModeCharDevice) != 0
}

// GetModelPath returns the full path to a model file
func GetModelPath(dir, modelName string) string {
	return filepath.Join(dir, modelName)
}

// PrintModelsStatus prints the status of all models in a directory
func PrintModelsStatus(dir string) {
	fmt.Printf("Models directory: %s\n", dir)
	fmt.Println()
	fmt.Println("Required models:")
	for _, model := range RequiredModels {
		status := "MISSING"
		if ModelExists(dir, model.Name) {
			status = "OK"
		}
		fmt.Printf("  [%s] %s\n", status, model.Name)
	}
	fmt.Println()
	fmt.Println("Optional models:")
	for _, model := range OptionalModels {
		status := "MISSING"
		if ModelExists(dir, model.Name) {
			status = "OK"
		}
		fmt.Printf("  [%s] %s\n", status, model.Name)
	}
}
