package gofacerecognition

type Config struct {
	ModelPaths ModelPaths
	UseGPU     bool
	NumJitters int // Number of times to re-sample the face (higher = more accurate but slower)
}

func NewConfig() (Config, error) {
	modelDir := DefaultModelsDir()

	if err := EnsureModels(modelDir); err != nil {
		return Config{}, err
	}

	return Config{
		ModelPaths: DefaultModelPaths(modelDir),
		UseGPU:     false,
		NumJitters: 1,
	}, nil
}
