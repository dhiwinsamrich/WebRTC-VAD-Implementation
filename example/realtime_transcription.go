package main

import (
	"bufio"
	"bytes"
	"context"
	"encoding/binary"
	"flag"
	"fmt"
	"io"
	"log"
	"os"
	"os/exec"
	"strings"
	"sync"
	"time"

	"github.com/baabaaox/go-webrtcvad"
	"google.golang.org/genai"
)

const (
	// VAD Configuration
	sampleRate    = 16000 // 16kHz is good for speech
	frameDuration = 20    // 20ms frames
	vadMode       = 2     // Aggressive mode (0=quality, 3=lowest false positive rate)

	// Audio buffer configuration
	minAudioSize   = 8000 // Minimum bytes before sending (0.5 seconds at 16kHz)
	maxSilenceTime = 1500 // Milliseconds of silence before flushing
)

var (
	frameSize  = sampleRate / 1000 * frameDuration
	frameBytes = frameSize * 2 // 16-bit = 2 bytes per sample
)

func main() {
	// Parse command line flags
	apiKey := flag.String("key", "", "Gemini API key (or set GEMINI_API_KEY env var)")
	micDevice := flag.String("mic", "audio=Microphone Array (Intel® Smart Sound Technology for Digital Microphones)", "Microphone device name for ffmpeg")
	language := flag.String("lang", "en", "Language code (e.g., en, hi, ta)")
	model := flag.String("model", "gemini-2.0-flash", "Gemini model name")
	flag.Parse()

	// Get API key
	geminiKey := *apiKey
	if geminiKey == "" {
		geminiKey = os.Getenv("GEMINI_API_KEY")
	}
	if geminiKey == "" {
		log.Fatal("❌ Gemini API key required. Use -key flag or set GEMINI_API_KEY environment variable")
	}

	fmt.Println("🎤 Starting Real-time Transcription with WebRTC VAD + Gemini")
	fmt.Printf("📝 Language: %s, Model: %s\n", *language, *model)
	fmt.Println("🎙️ Listening... speak now!\n")

	// Initialize WebRTC VAD
	vad := webrtcvad.Create()
	defer webrtcvad.Free(vad)

	if err := webrtcvad.Init(vad); err != nil {
		log.Fatalf("Failed to initialize VAD: %v", err)
	}

	if err := webrtcvad.SetMode(vad, vadMode); err != nil {
		log.Fatalf("Failed to set VAD mode: %v", err)
	}

	// Validate frame size
	if !webrtcvad.ValidRateAndFrameLength(sampleRate, frameSize) {
		log.Fatalf("Invalid rate/frame combination: rate=%d, frameLength=%d", sampleRate, frameSize)
	}

	// Initialize Gemini client
	ctx := context.Background()
	client, err := genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  geminiKey,
		Backend: genai.BackendGeminiAPI,
	})
	if err != nil {
		log.Fatalf("Failed to create Gemini client: %v", err)
	}
	// Note: genai.Client doesn't have Close() method

	// Start microphone capture with ffmpeg
	cmd := exec.Command("ffmpeg",
		"-f", "dshow",
		"-i", *micDevice,
		"-ar", fmt.Sprintf("%d", sampleRate),
		"-ac", "1",
		"-f", "s16le",
		"-",
	)

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		log.Fatalf("Failed to get ffmpeg output: %v", err)
	}

	if err := cmd.Start(); err != nil {
		log.Fatalf("Failed to start ffmpeg: %v", err)
	}
	defer cmd.Process.Kill()

	// Audio buffer for voice activity
	var (
		audioBuffer    []byte
		bufferMutex    sync.Mutex
		lastSpeechTime time.Time
		isProcessing   bool
	)

	// Goroutine: Process audio and detect voice activity
	go func() {
		reader := bufio.NewReader(stdout)
		frame := make([]byte, frameBytes)

		for {
			// Read one frame
			_, err := io.ReadFull(reader, frame)
			if err == io.EOF {
				break
			}
			if err != nil {
				log.Printf("Audio read error: %v", err)
				break
			}

			// Process frame with VAD
			frameActive, err := webrtcvad.Process(vad, sampleRate, frame, frameSize)
			if err != nil {
				log.Printf("VAD process error: %v", err)
				continue
			}

			bufferMutex.Lock()

			if frameActive {
				// Voice detected - add to buffer
				audioBuffer = append(audioBuffer, frame...)
				lastSpeechTime = time.Now()
				bufferMutex.Unlock()
				fmt.Print("🎤") // Visual indicator of voice activity
				continue
			}

			var chunk []byte
			shouldProcess := false

			if len(audioBuffer) > 0 {
				silenceDuration := time.Since(lastSpeechTime)
				if silenceDuration > time.Duration(maxSilenceTime)*time.Millisecond && len(audioBuffer) >= minAudioSize && !isProcessing {
					// Copy buffered audio before releasing the lock
					chunk = append([]byte(nil), audioBuffer...)
					audioBuffer = audioBuffer[:0] // Clear buffer
					isProcessing = true
					shouldProcess = true
				}
			}

			bufferMutex.Unlock()

			if shouldProcess {
				go func(data []byte) {
					processAudio(ctx, client, data, *language, *model)
					bufferMutex.Lock()
					isProcessing = false
					bufferMutex.Unlock()
				}(chunk)
			}

			fmt.Print(".") // Visual indicator of silence
		}
	}()

	// Keep main goroutine alive
	select {}
}

// processAudio sends audio to Gemini API for transcription
func processAudio(ctx context.Context, client *genai.Client, audioData []byte, language, modelName string) {
	if len(audioData) < minAudioSize {
		return
	}

	// Create WAV header
	wavData := createWAVHeader(audioData, sampleRate, 1, 16)
	wavData = append(wavData, audioData...)

	// Build language-specific prompt
	languageMap := map[string]string{
		"en": "English",
		"hi": "Hindi",
		"ta": "Tamil",
		"te": "Telugu",
		"kn": "Kannada",
		"ml": "Malayalam",
		"bn": "Bengali",
		"mr": "Marathi",
		"gu": "Gujarati",
		"pa": "Punjabi",
		"es": "Spanish",
		"fr": "French",
		"de": "German",
		"it": "Italian",
		"pt": "Portuguese",
		"ru": "Russian",
		"ja": "Japanese",
		"ko": "Korean",
		"zh": "Chinese",
		"ar": "Arabic",
	}

	langCode := language
	if len(langCode) > 2 {
		langCode = langCode[:2]
	}

	languageName := languageMap[langCode]
	if languageName == "" {
		languageName = "multilingual"
	}

	systemInstruction := fmt.Sprintf("You are an audio transcription service. You MUST transcribe ONLY in %s language. Transcribe only clear speech accurately. Ignore background noise, breathing sounds, and non-speech audio. Never translate or use any other language.", languageName)
	promptText := fmt.Sprintf("Listen to this audio and transcribe it in %s ONLY. Return only the spoken words in %s, nothing else. If there is no clear speech, return empty. Do not translate to any other language.", languageName, languageName)

	// Create chat for transcription
	tempChat, err := client.Chats.Create(ctx, modelName, &genai.GenerateContentConfig{
		SystemInstruction: &genai.Content{
			Parts: []*genai.Part{genai.NewPartFromText(systemInstruction)},
			Role:  genai.RoleModel,
		},
	}, nil)

	if err != nil {
		log.Printf("❌ Failed to create Gemini chat: %v", err)
		return
	}

	// Create parts
	textPart := genai.Part{
		Text: promptText,
	}

	audioPart := genai.Part{
		InlineData: &genai.Blob{
			MIMEType: "audio/wav",
			Data:     wavData,
		},
	}

	// Send audio with prompt
	iter := tempChat.SendMessageStream(ctx, textPart, audioPart)

	transcript := ""
	for resp, err := range iter {
		if err != nil {
			log.Printf("❌ Gemini transcription error: %v", err)
			return
		}

		for _, cand := range resp.Candidates {
			if cand.Content != nil {
				for _, part := range cand.Content.Parts {
					if part.Text != "" {
						transcript += part.Text
					}
				}
			}
		}
	}

	transcript = strings.TrimSpace(transcript)

	// Filter out very short transcripts
	if transcript == "" || len(transcript) < 2 {
		return
	}

	// Display transcription
	fmt.Printf("\n📝 Transcript: %s\n", transcript)
}

// createWAVHeader creates a proper WAV header for raw PCM audio
func createWAVHeader(audioData []byte, sampleRate, channels, bitsPerSample int) []byte {
	dataSize := int32(len(audioData))
	var buf bytes.Buffer

	// RIFF header
	buf.WriteString("RIFF")
	binary.Write(&buf, binary.LittleEndian, dataSize+36) // File size - 8
	buf.WriteString("WAVE")

	// fmt subchunk
	buf.WriteString("fmt ")
	binary.Write(&buf, binary.LittleEndian, int32(16)) // Subchunk1Size
	binary.Write(&buf, binary.LittleEndian, int16(1))  // AudioFormat (1 = PCM)
	binary.Write(&buf, binary.LittleEndian, int16(channels))
	binary.Write(&buf, binary.LittleEndian, int32(sampleRate))
	binary.Write(&buf, binary.LittleEndian, int32(sampleRate*channels*bitsPerSample/8)) // ByteRate
	binary.Write(&buf, binary.LittleEndian, int16(channels*bitsPerSample/8))            // BlockAlign
	binary.Write(&buf, binary.LittleEndian, int16(bitsPerSample))

	// data subchunk
	buf.WriteString("data")
	binary.Write(&buf, binary.LittleEndian, dataSize)

	return buf.Bytes()
}
