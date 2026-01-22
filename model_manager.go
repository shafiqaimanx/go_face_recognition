package gofacerecognition

import (
	"fmt"
	"io"
	"net/http"
	"os"
	"path/filepath"
	"runtime"
)

const (
	GitHubReleasesBase = "https://github.com/shafiqaimanx/go_face_recognition/releases/download/v1/"

	ShapePredictor68URL = GitHubReleasesBase + "shape_predictor_68_face_landmarks.dat"
	ShapePredictor5URL  = GitHubReleasesBase + "shape_predictor_5_face_landmarks.dat"
	FaceRecognitionURL  = GitHubReleasesBase + "dlib_face_recognition_resnet_model_v1.dat"
)

const (
	ShapePredictor68File = "shape_predictor_68_face_landmarks.dat"
	ShapePredictor5File  = "shape_predictor_5_face_landmarks.dat"
	FaceRecognitionFile  = "dlib_face_recognition_resnet_model_v1.dat"
)

type ModelInfo struct {
	Name     string
	URL      string
	Required bool
}

var AllModels = []ModelInfo{
	{Name: ShapePredictor68File, URL: ShapePredictor68URL, Required: true},
	{Name: ShapePredictor5File, URL: ShapePredictor5URL, Required: true},
	{Name: FaceRecognitionFile, URL: FaceRecognitionURL, Required: true},
}

// DefaultModelsDir: Returns the default directory for storing models
// Linux/macOS: ~/.goface_recognition/models/
// Windows: %USERPROFILE%\.goface_recognition\models\
func DefaultModelsDir() string {
	home, err := os.UserHomeDir()
	if err != nil {
		// fallback to current directory
		return filepath.Join(".", ".go_face_recognition", "models")
	}
	return filepath.Join(home, ".go_face_recognition", "models")
}

func ModelsExist(dir string) bool {
	for _, model := range AllModels {
		path := filepath.Join(dir, model.Name)
		if _, err := os.Stat(path); os.IsNotExist(err) {
			return false
		}
	}
	return true
}

func ModelExists(dir, modelName string) bool {
	path := filepath.Join(dir, modelName)
	_, err := os.Stat(path)
	return err == nil
}

func EnsureModels(dir string) error {
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
				continue
			}
		}
	}
	return nil
}

func DownloadModel(url, destpath string) error {
	client := &http.Client{
		CheckRedirect: func(req *http.Request, via []*http.Request) error {
			return nil
		},
	}

	resp, err := client.Get(url)
	if err != nil {
		return fmt.Errorf("HTTP request failed: %w", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		return fmt.Errorf("HTTP status %d: %s", resp.StatusCode, resp.Status)
	}

	destFile, err := os.Create(destpath)
	if err != nil {
		return fmt.Errorf("failed to create file: %w", err)
	}
	defer destFile.Close()

	written, err := copyWithProgress(destFile, resp.Body, resp.ContentLength)
	if err != nil {
		os.Remove(destpath)
		return fmt.Errorf("download failed: %w", err)
	}

	fmt.Printf("\nDownloaded %.2f MB\n", float64(written)/(1024*1024))
	return nil
}

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

func isTerminal() bool {
	if runtime.GOOS == "windows" {
		return true
	}
	fi, err := os.Stdout.Stat()
	if err != nil {
		return false
	}
	return (fi.Mode() & os.ModeCharDevice) != 0
}
