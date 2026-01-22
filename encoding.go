package gofacerecognition

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"io"
)

// EncodingToBytes converts a FaceEncoding to a byte slice
// Uses little-endian binary format for efficient storage
func EncodingToBytes(encoding FaceEncoding) []byte {
	buf := new(bytes.Buffer)
	for _, v := range encoding {
		binary.Write(buf, binary.LittleEndian, v)
	}
	return buf.Bytes()
}

// BytesToEncoding converts a byte slice back to a FaceEncoding
func BytesToEncoding(data []byte) (FaceEncoding, error) {
	var encoding FaceEncoding
	buf := bytes.NewReader(data)

	for i := 0; i < 128; i++ {
		var v float64
		if err := binary.Read(buf, binary.LittleEndian, &v); err != nil {
			return encoding, err
		}
		encoding[i] = v
	}

	return encoding, nil
}

// WriteEncodings writes multiple face encodings to a writer in binary format
// Format: [count uint32][encoding1][encoding2]...
func WriteEncodings(w io.Writer, encodings []FaceEncoding) error {
	// Write count
	if err := binary.Write(w, binary.LittleEndian, uint32(len(encodings))); err != nil {
		return err
	}

	// Write each encoding
	for _, enc := range encodings {
		for _, v := range enc {
			if err := binary.Write(w, binary.LittleEndian, v); err != nil {
				return err
			}
		}
	}

	return nil
}

// ReadEncodings reads multiple face encodings from a reader
func ReadEncodings(r io.Reader) ([]FaceEncoding, error) {
	// Read count
	var count uint32
	if err := binary.Read(r, binary.LittleEndian, &count); err != nil {
		return nil, err
	}

	encodings := make([]FaceEncoding, count)

	// Read each encoding
	for i := uint32(0); i < count; i++ {
		for j := 0; j < 128; j++ {
			if err := binary.Read(r, binary.LittleEndian, &encodings[i][j]); err != nil {
				return nil, err
			}
		}
	}

	return encodings, nil
}

// EncodingToJSON converts a FaceEncoding to JSON
func EncodingToJSON(encoding FaceEncoding) ([]byte, error) {
	return json.Marshal(encoding)
}

// JSONToEncoding converts JSON back to a FaceEncoding
func JSONToEncoding(data []byte) (FaceEncoding, error) {
	var encoding FaceEncoding
	err := json.Unmarshal(data, &encoding)
	return encoding, err
}

// NamedEncoding pairs a name with a face encoding for serialization
type NamedEncoding struct {
	Name     string        `json:"name"`
	Encoding FaceEncoding  `json:"encoding"`
	Metadata interface{}   `json:"metadata,omitempty"`
}

// EncodeNamedEncodings serializes named encodings to JSON
func EncodeNamedEncodings(encodings []NamedEncoding) ([]byte, error) {
	return json.MarshalIndent(encodings, "", "  ")
}

// DecodeNamedEncodings deserializes named encodings from JSON
func DecodeNamedEncodings(data []byte) ([]NamedEncoding, error) {
	var encodings []NamedEncoding
	err := json.Unmarshal(data, &encodings)
	return encodings, err
}

// NormalizeEncoding normalizes a face encoding to unit length
// This can sometimes improve comparison results
func NormalizeEncoding(encoding FaceEncoding) FaceEncoding {
	var sum float64
	for _, v := range encoding {
		sum += v * v
	}

	if sum == 0 {
		return encoding
	}

	var norm FaceEncoding
	magnitude := 1.0 / sum // Avoid sqrt for efficiency
	for i, v := range encoding {
		norm[i] = v * magnitude
	}

	return norm
}
