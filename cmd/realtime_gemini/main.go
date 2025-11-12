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

	// Audio buffer configuration for live transcription
	minAudioSize       = 8000  // Minimum bytes before sending (0.5 seconds at 16kHz)
	liveChunkSize      = 16000 // Send chunks every ~1 second for live transcription
	maxSilenceTime     = 800   // Milliseconds of silence before final flush
	liveUpdateInterval = 1000  // Send live updates every 1 second of speech
)

var (
	frameSize  = sampleRate / 1000 * frameDuration
	frameBytes = frameSize * 2 // 16-bit = 2 bytes per sample
)

func main() {
	// Parse command line flags
	apiKey := flag.String("key", "", "Gemini API key (or set GEMINI_API_KEY env var)")
	micDevice := flag.String("mic", "audio=Microphone Array (Realtek(R) Audio)", "Microphone device name for ffmpeg")
	language := flag.String("lang", "en", "Language code (e.g., en, hi, ta)")
	model := flag.String("model", "gemini-2.0-flash", "Gemini model name")
	duration := flag.Int("duration", 10, "Recording duration in seconds")
	debug := flag.Bool("debug", false, "Enable debug output")
	liveOutput := flag.Bool("live", true, "Show streaming transcription while recording")
	flag.Parse()

	// Get API key
	geminiKey := *apiKey
	if geminiKey == "" {
		geminiKey = os.Getenv("GEMINI_API_KEY")
	}
	if geminiKey == "" {
		log.Fatal("❌ Gemini API key required. Use -key flag or set GEMINI_API_KEY environment variable")
	}

	if *debug {
		fmt.Printf("🎤 Recording for %d seconds, then transcribing...\n", *duration)
		fmt.Printf("📝 Language: %s, Model: %s\n", *language, *model)
		fmt.Printf("🔧 Debug mode enabled\n")
		fmt.Printf("⚙️  VAD Mode: %d, Sample Rate: %d Hz, Frame Size: %d samples\n", vadMode, sampleRate, frameSize)
		fmt.Printf("🎙️ Recording... speak now! (Recording will stop in %d seconds)\n\n", *duration)
	} else {
		// Silent mode - just show minimal info
		fmt.Printf("Recording for %d seconds...\n", *duration)
	}

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

	// Normalize microphone device name (remove spaces after audio=)
	micName := strings.TrimSpace(*micDevice)
	if strings.HasPrefix(micName, "audio=") {
		// Remove any spaces after audio=
		parts := strings.SplitN(micName, "=", 2)
		if len(parts) == 2 {
			micName = "audio=" + strings.TrimSpace(parts[1])
		}
	} else if !strings.HasPrefix(micName, "audio=") {
		// If user didn't include "audio=" prefix, add it
		micName = "audio=" + strings.TrimSpace(micName)
	}

	fmt.Printf("🎙️ Using microphone: %s\n", micName)

	// Start microphone capture with ffmpeg
	cmd := exec.Command("ffmpeg",
		"-f", "dshow",
		"-i", micName,
		"-ar", fmt.Sprintf("%d", sampleRate),
		"-ac", "1",
		"-f", "s16le",
		"-loglevel", "error", // Suppress ffmpeg logs
		"-",
	)

	// Capture stderr for debugging
	stderr, err := cmd.StderrPipe()
	if err != nil {
		log.Printf("Warning: Could not get stderr: %v", err)
	} else {
		go func() {
			scanner := bufio.NewScanner(stderr)
			for scanner.Scan() {
				log.Printf("FFmpeg: %s", scanner.Text())
			}
		}()
	}

	stdout, err := cmd.StdoutPipe()
	if err != nil {
		log.Fatalf("❌ Failed to get ffmpeg output: %v", err)
	}

	if err := cmd.Start(); err != nil {
		log.Fatalf("❌ Failed to start ffmpeg: %v\n\n💡 Tips:\n   1. List available devices: ffmpeg -list_devices true -f dshow -i dummy\n   2. Use exact device name: -mic \"Headset (OnePlus Buds Pro 2)\" (no 'audio=' needed)\n   3. Or use: -mic \"audio=Headset (OnePlus Buds Pro 2)\" (no space after =)", err)
	}
	defer cmd.Process.Kill()

	// Audio buffers for recording
	var (
		audioBuffer       []byte
		liveBuffer        []byte
		bufferMutex       sync.Mutex
		recordingStart    = time.Now()
		recordingDone     = make(chan bool)
		frameCount        int
		activeFrames      int
		liveProcessing    bool
		lastLiveSend      time.Time
		lastSpeechTime    time.Time
		currentTranscript string
		liveWG            sync.WaitGroup
	)

	// Goroutine: Record audio for specified duration
	go func() {
		reader := bufio.NewReader(stdout)
		frame := make([]byte, frameBytes)

		if *debug {
			fmt.Println("✅ Recording started...")
		}

		for {
			// Check if recording time is up
			elapsed := time.Since(recordingStart)
			if elapsed >= time.Duration(*duration)*time.Second {
				recordingDone <- true
				break
			}

			// Read one frame
			n, err := io.ReadFull(reader, frame)
			if err == io.EOF {
				log.Println("Audio stream ended")
				recordingDone <- true
				break
			}
			if err != nil {
				log.Printf("⚠️ Audio read error: %v (read %d bytes)", err, n)
				time.Sleep(10 * time.Millisecond)
				continue
			}

			frameCount++

			// Process frame with VAD
			frameActive, err := webrtcvad.Process(vad, sampleRate, frame, frameSize)
			if err != nil {
				log.Printf("⚠️ VAD process error: %v", err)
				continue
			}

			bufferMutex.Lock()
			now := time.Now()
			audioBuffer = append(audioBuffer, frame...)

			if frameActive {
				activeFrames++
				lastSpeechTime = now
			}

			sendLive := false
			var liveChunk []byte

			if *liveOutput && (frameActive || len(liveBuffer) > 0) {
				liveBuffer = append(liveBuffer, frame...)
			}

			if *liveOutput {
				intervalDuration := time.Duration(liveUpdateInterval) * time.Millisecond
				silenceDuration := time.Duration(maxSilenceTime) * time.Millisecond

				if len(liveBuffer) >= liveChunkSize && !liveProcessing {
					sendLive = true
				} else if !liveProcessing && len(liveBuffer) >= 4000 {
					if lastLiveSend.IsZero() || now.Sub(lastLiveSend) >= intervalDuration {
						sendLive = true
					}
				}

				if !liveProcessing && !frameActive && !lastSpeechTime.IsZero() && now.Sub(lastSpeechTime) >= silenceDuration && len(liveBuffer) >= minAudioSize {
					sendLive = true
				}

				if sendLive && len(liveBuffer) > 0 && !liveProcessing {
					liveChunk = append([]byte(nil), liveBuffer...)
					liveBuffer = liveBuffer[:0]
					liveProcessing = true
					lastLiveSend = now
				}
			}

			bufferMutex.Unlock()

			if liveChunk != nil {
				liveWG.Add(1)
				go func(data []byte) {
					defer liveWG.Done()
					processAudioLive(ctx, client, data, *language, *model, true, &currentTranscript, *liveOutput)
					bufferMutex.Lock()
					liveProcessing = false
					bufferMutex.Unlock()
				}(liveChunk)
			}

			// Show progress only in debug mode
			if *debug {
				remaining := *duration - int(elapsed.Seconds())
				if frameCount%250 == 0 && remaining > 0 {
					fmt.Printf("\r⏱️  Recording... %d seconds remaining | Frames: %d | Voice: %d (%.1f%%)",
						remaining, frameCount, activeFrames, float64(activeFrames)/float64(frameCount)*100)
				}
			}

			if *liveOutput {
				if frameActive {
					fmt.Print("🎤")
				} else {
					fmt.Print(".")
				}
			}
		}
	}()

	// Wait for recording to complete
	<-recordingDone

	// Stop ffmpeg
	cmd.Process.Kill()

	// Flush any remaining live buffer
	bufferMutex.Lock()
	var tailChunk []byte
	if *liveOutput && !liveProcessing && len(liveBuffer) >= 4000 {
		tailChunk = append([]byte(nil), liveBuffer...)
		liveBuffer = liveBuffer[:0]
		liveProcessing = true
		lastLiveSend = time.Now()
	}
	bufferMutex.Unlock()

	if tailChunk != nil {
		liveWG.Add(1)
		go func(data []byte) {
			defer liveWG.Done()
			processAudioLive(ctx, client, data, *language, *model, true, &currentTranscript, *liveOutput)
			bufferMutex.Lock()
			liveProcessing = false
			bufferMutex.Unlock()
		}(tailChunk)
	}

	liveWG.Wait()

	if *debug {
		fmt.Printf("\n\n✅ Recording complete! Processing %d frames (%.2f seconds of audio)...\n",
			frameCount, float64(len(audioBuffer))/(float64(sampleRate)*2))
	}

	// Get the final audio buffer
	bufferMutex.Lock()
	finalAudio := append([]byte(nil), audioBuffer...)
	audioBuffer = audioBuffer[:0]
	bufferMutex.Unlock()

	// Move live transcript to a new line before final result
	if *liveOutput && currentTranscript != "" {
		fmt.Println()
	}

	// Transcribe the complete recording
	if len(finalAudio) >= minAudioSize {
		if *debug {
			fmt.Println("📤 Sending audio to Gemini for transcription...")
		}
		var wg sync.WaitGroup
		wg.Add(1)
		go func(data []byte) {
			defer wg.Done()
			processAudioLive(ctx, client, data, *language, *model, false, nil, *liveOutput)
		}(finalAudio)
		wg.Wait()
	} else {
		if *debug {
			fmt.Printf("⚠️ Not enough audio recorded (%d bytes, need %d bytes)\n", len(finalAudio), minAudioSize)
		}
	}
}

// processAudioLive sends audio to Gemini API for transcription
func processAudioLive(ctx context.Context, client *genai.Client, audioData []byte, language, modelName string, isLive bool, currentTranscript *string, showOutput bool) {
	if len(audioData) < minAudioSize && !isLive {
		return
	}

	if isLive {
		// For live updates, we can send smaller chunks
		if len(audioData) < 4000 {
			return
		}
	}

	// Skip if no audio data
	if len(audioData) == 0 {
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
	responseCount := 0
	for resp, err := range iter {
		if err != nil {
			if isLive {
				// Don't log errors for live updates to avoid spam
				return
			}
			log.Printf("❌ Gemini transcription error: %v", err)
			return
		}

		responseCount++

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
	if transcript == "" || len(transcript) < 1 {
		return
	}

	if isLive && currentTranscript != nil {
		// Live update - append to current transcript
		newTranscript := strings.TrimSpace(*currentTranscript + " " + transcript)
		*currentTranscript = newTranscript
		if showOutput {
			// Clear line and show live transcript (with padding to clear previous text)
			fmt.Printf("\r🔴 LIVE: %-80s", newTranscript)
		}
	} else {
		// Final transcript - show only the text
		if currentTranscript != nil && *currentTranscript != "" {
			finalText := *currentTranscript
			if showOutput {
				fmt.Println(strings.TrimSpace(finalText))
			} else {
				fmt.Println(strings.TrimSpace(finalText))
			}
			*currentTranscript = ""
		} else {
			fmt.Println(strings.TrimSpace(transcript))
		}
	}
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
