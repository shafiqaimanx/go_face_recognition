package gofacerecognition

// Config holds configuration options for the FaceRecognizer
type Config struct {
	// ModelPaths contains paths to all model files
	ModelPaths ModelPaths
	// UseGPU enables GPU acceleration if available
	UseGPU bool
	// NumJitters is the number of times to re-sample the face (higher = more accurate but slower)
	NumJitters int
}

// DefaultConfig returns a Config with models from the specified directory
// Use this when you know exactly where your models are located
func DefaultConfig(modelDir string) Config {
	return Config{
		ModelPaths: DefaultModelPaths(modelDir),
		UseGPU:     false,
		NumJitters: 1,
	}
}

// DefaultConfigAuto returns a Config that automatically finds or downloads models
// Priority:
// 1. Check user-specified path (if provided)
// 2. Check default location (~/.goface_recognition/models/)
// 3. Auto-download to default location if autoDownload is true
func DefaultConfigAuto(userPath string, autoDownload bool) (Config, error) {
	modelDir, err := ResolveModelsDir(userPath, autoDownload)
	if err != nil {
		return Config{}, err
	}

	return Config{
		ModelPaths: DefaultModelPaths(modelDir),
		UseGPU:     false,
		NumJitters: 1,
	}, nil
}

// DefaultConfigWithDownload returns a Config, downloading models if needed
// This is the easiest way to get started - just call this and it handles everything
func DefaultConfigWithDownload() (Config, error) {
	return DefaultConfigAuto("", true)
}

// DefaultConfigEmbedded returns a Config using the default models location
// Does NOT auto-download - models must already exist
func DefaultConfigEmbedded() Config {
	return Config{
		ModelPaths: DefaultModelPaths(DefaultModelsDir()),
		UseGPU:     false,
		NumJitters: 1,
	}
}
