package main

import (
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"time"

	"github.com/baabaaox/go-webrtcvad"
	"github.com/cryptix/wav"
)

func main() {
	flag.Parse()

	// Default to test.wav in the test directory if no argument provided
	var filename string
	if flag.NArg() < 1 {
		// Try to find test file in parent/test directory
		defaultPath := "../test/test2.wav"
		if _, err := os.Stat(defaultPath); err == nil {
			filename = defaultPath
		} else {
			log.Fatal("usage: example [infile.wav] or place test.wav in ../test/ directory")
		}
	} else {
		filename = flag.Arg(0)
	}

	info, err := os.Stat(filename)
	if err != nil {
		log.Fatal(err)
	}

	file, err := os.Open(filename)
	if err != nil {
		log.Fatal(err)
	}

	wavReader, err := wav.NewReader(file, info.Size())
	if err != nil {
		log.Fatal(err)
	}
	reader, err := wavReader.GetDumbReader()
	if err != nil {
		log.Fatal(err)
	}

	wavInfo := wavReader.GetFile()
	originalRate := int(wavInfo.SampleRate)
	channels := int(wavInfo.Channels)
	bitsPerSample := int(wavInfo.SignificantBits)

	if bitsPerSample != 16 {
		log.Fatalf("expected 16-bit audio, got %d-bit", bitsPerSample)
	}

	// Convert stereo to mono if needed
	var monoReader io.Reader = reader
	if channels == 2 {
		log.Printf("Converting stereo to mono (taking left channel)")
		monoReader = &stereoToMonoReader{reader: reader}
	} else if channels != 1 {
		log.Fatalf("expected mono or stereo file, got %d channels", channels)
	}

	// Check if sample rate is supported and resample if needed
	validRates := []int{8000, 16000, 32000, 48000}
	rate := originalRate
	rateValid := false
	for _, r := range validRates {
		if rate == r {
			rateValid = true
			break
		}
	}

	var finalReader io.Reader = monoReader
	if !rateValid {
		// Find the nearest supported rate
		nearestRate := validRates[0]
		minDiff := abs(rate - nearestRate)
		for _, r := range validRates {
			diff := abs(rate - r)
			if diff < minDiff {
				minDiff = diff
				nearestRate = r
			}
		}
		log.Printf("Resampling from %d Hz to %d Hz", originalRate, nearestRate)
		finalReader = &resampler{
			reader:     monoReader,
			inputRate:  originalRate,
			outputRate: nearestRate,
		}
		rate = nearestRate
	}

	vad := webrtcvad.Create()
	defer webrtcvad.Free(vad)

	if err := webrtcvad.Init(vad); err != nil {
		log.Fatal(err)
	}

	if err := webrtcvad.SetMode(vad, 2); err != nil {
		log.Fatal(err)
	}

	// Calculate frame size: use 20ms frames (10ms, 20ms, or 30ms are valid)
	// frameLength in samples = rate * frameDurationMs / 1000
	frameDurationMs := 20
	frameLength := rate * frameDurationMs / 1000
	frameSize := frameLength * 2 // 16-bit = 2 bytes per sample

	// Validate frame length
	if ok := webrtcvad.ValidRateAndFrameLength(rate, frameLength); !ok {
		// Try 10ms frame
		frameDurationMs = 10
		frameLength = rate * frameDurationMs / 1000
		frameSize = frameLength * 2
		if ok := webrtcvad.ValidRateAndFrameLength(rate, frameLength); !ok {
			// Try 30ms frame
			frameDurationMs = 30
			frameLength = rate * frameDurationMs / 1000
			frameSize = frameLength * 2
			if ok := webrtcvad.ValidRateAndFrameLength(rate, frameLength); !ok {
				log.Fatalf("invalid rate/frame combination: rate=%d, frameLength=%d", rate, frameLength)
			}
		}
	}

	frame := make([]byte, frameSize)
	log.Printf("Using %d Hz, %dms frames (%d samples, %d bytes)", rate, frameDurationMs, frameLength, frameSize)

	var isActive bool
	var offset int

	report := func() {
		// offset is in bytes, convert to samples (2 bytes per sample for 16-bit)
		samples := offset / 2
		t := time.Duration(samples) * time.Second / time.Duration(rate)
		fmt.Printf("isActive = %v, t = %v\n", isActive, t)
	}

	for {
		_, err := io.ReadFull(finalReader, frame)
		if err == io.EOF || err == io.ErrUnexpectedEOF {
			break
		}
		if err != nil {
			log.Fatal(err)
		}

		frameActive, err := webrtcvad.Process(vad, rate, frame, frameLength)
		if err != nil {
			log.Fatal(err)
		}

		if isActive != frameActive || offset == 0 {
			isActive = frameActive
			report()
		}

		offset += len(frame)
	}

	report()
}

// stereoToMonoReader converts stereo 16-bit audio to mono by taking the left channel
type stereoToMonoReader struct {
	reader io.Reader
	buffer []byte
}

func (r *stereoToMonoReader) Read(p []byte) (n int, err error) {
	// Read stereo data (2 channels * 2 bytes per sample)
	stereoSize := len(p) * 2
	if len(r.buffer) < stereoSize {
		r.buffer = make([]byte, stereoSize)
	}

	// Read stereo frame
	stereoN, err := r.reader.Read(r.buffer[:stereoSize])
	if err != nil && err != io.EOF {
		return 0, err
	}

	// Convert to mono by taking left channel (every 4th byte starting from 0)
	// Stereo 16-bit: [L1_LSB, L1_MSB, R1_LSB, R1_MSB, L2_LSB, L2_MSB, ...]
	// Mono: [L1_LSB, L1_MSB, L2_LSB, L2_MSB, ...]
	for i := 0; i < stereoN/4; i++ {
		// Copy left channel (first 2 bytes of each stereo sample pair)
		p[i*2] = r.buffer[i*4]
		p[i*2+1] = r.buffer[i*4+1]
	}

	n = stereoN / 2
	return n, err
}

// resampler converts audio from one sample rate to another using linear interpolation
type resampler struct {
	reader     io.Reader
	inputRate  int
	outputRate int
	buffer     []byte
}

func abs(x int) int {
	if x < 0 {
		return -x
	}
	return x
}

func (r *resampler) Read(p []byte) (n int, err error) {
	// Calculate how many input samples we need
	outputSamples := len(p) / 2 // 16-bit = 2 bytes per sample
	inputSamples := int(float64(outputSamples) * float64(r.inputRate) / float64(r.outputRate))
	inputSize := inputSamples * 2

	// Ensure buffer is large enough
	if len(r.buffer) < inputSize {
		r.buffer = make([]byte, inputSize)
	}

	// Read input samples
	inputN, err := r.reader.Read(r.buffer[:inputSize])
	if err != nil && err != io.EOF {
		return 0, err
	}
	if inputN == 0 {
		return 0, err
	}

	// Resample using linear interpolation
	inputSampleCount := inputN / 2
	ratio := float64(r.inputRate) / float64(r.outputRate)

	for i := 0; i < outputSamples && i*2 < len(p); i++ {
		// Calculate position in input buffer
		srcPos := float64(i) * ratio

		// Get integer and fractional parts
		idx := int(srcPos)
		frac := srcPos - float64(idx)

		if idx+1 < inputSampleCount {
			// Linear interpolation between two samples
			sample1 := int16(r.buffer[idx*2]) | int16(r.buffer[idx*2+1])<<8
			sample2 := int16(r.buffer[(idx+1)*2]) | int16(r.buffer[(idx+1)*2+1])<<8

			// Interpolate
			interpolated := float64(sample1)*(1.0-frac) + float64(sample2)*frac
			sample := int16(interpolated)

			// Write to output (little-endian)
			p[i*2] = byte(sample)
			p[i*2+1] = byte(sample >> 8)
		} else if idx < inputSampleCount {
			// Last sample, no interpolation needed
			p[i*2] = r.buffer[idx*2]
			p[i*2+1] = r.buffer[idx*2+1]
		} else {
			// Past end of input
			break
		}
	}

	n = outputSamples * 2
	if n > len(p) {
		n = len(p)
	}
	return n, err
}
