package gofacerecognition

import "math"

// FaceDistance calculates the Euclidean distance between two face encodings
// Lower distance means more similar faces
func FaceDistance(encoding1, encoding2 FaceEncoding) float64 {
	var sum float64
	for i := 0; i < 128; i++ {
		diff := encoding1[i] - encoding2[i]
		sum += diff * diff
	}
	return math.Sqrt(sum)
}

// FaceDistances calculates the Euclidean distance between a face encoding and a list of encodings
// Returns a slice of distances in the same order as the input encodings
func FaceDistances(encodings []FaceEncoding, faceToCompare FaceEncoding) []float64 {
	if len(encodings) == 0 {
		return []float64{}
	}

	distances := make([]float64, len(encodings))
	for i, encoding := range encodings {
		distances[i] = FaceDistance(encoding, faceToCompare)
	}
	return distances
}

// CompareFaces compares a list of face encodings against a candidate encoding
// Returns a slice of booleans indicating which faces match (distance <= tolerance)
// Default tolerance is 0.6
func CompareFaces(knownEncodings []FaceEncoding, faceToCheck FaceEncoding, tolerance float64) []bool {
	if tolerance <= 0 {
		tolerance = 0.6 // Default tolerance
	}

	distances := FaceDistances(knownEncodings, faceToCheck)
	matches := make([]bool, len(distances))

	for i, distance := range distances {
		matches[i] = distance <= tolerance
	}

	return matches
}

// CompareFacesWithDistances compares faces and returns both matches and distances
func CompareFacesWithDistances(knownEncodings []FaceEncoding, faceToCheck FaceEncoding, tolerance float64) ([]bool, []float64) {
	if tolerance <= 0 {
		tolerance = 0.6
	}

	distances := FaceDistances(knownEncodings, faceToCheck)
	matches := make([]bool, len(distances))

	for i, distance := range distances {
		matches[i] = distance <= tolerance
	}

	return matches, distances
}

// FindBestMatch finds the best matching face from a list of known encodings
// Returns the index of the best match and the distance, or -1 if no match found within tolerance
func FindBestMatch(knownEncodings []FaceEncoding, faceToCheck FaceEncoding, tolerance float64) (int, float64) {
	if len(knownEncodings) == 0 {
		return -1, 0
	}

	if tolerance <= 0 {
		tolerance = 0.6
	}

	distances := FaceDistances(knownEncodings, faceToCheck)

	bestIndex := -1
	bestDistance := tolerance + 1 // Start with value higher than tolerance

	for i, distance := range distances {
		if distance <= tolerance && distance < bestDistance {
			bestIndex = i
			bestDistance = distance
		}
	}

	if bestIndex == -1 {
		return -1, 0
	}

	return bestIndex, bestDistance
}

// AverageEncoding calculates the average of multiple face encodings
// Useful for creating a more robust encoding from multiple images of the same person
func AverageEncoding(encodings []FaceEncoding) FaceEncoding {
	if len(encodings) == 0 {
		return FaceEncoding{}
	}

	var avg FaceEncoding
	for _, encoding := range encodings {
		for i := 0; i < 128; i++ {
			avg[i] += encoding[i]
		}
	}

	n := float64(len(encodings))
	for i := 0; i < 128; i++ {
		avg[i] /= n
	}

	return avg
}
