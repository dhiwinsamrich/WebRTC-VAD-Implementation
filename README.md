# go-webrtcvad

[![Go Tests](https://github.com/baabaaox/go-webrtcvad/actions/workflows/go-tests.yml/badge.svg)](https://github.com/baabaaox/go-webrtcvad/actions/workflows/go-tests.yml)

I have ported VAD from latest WebRTC lkgr at 20250207.

This project rewrite from [maxhawkins/go-webrtcvad](https://github.com/maxhawkins/go-webrtcvad).

The WebRTC source code download form [WebRTC lkgr commit [8e55dca89f4e39241f9e3ecd25ab0ebbf5d1ab37]](https://webrtc.googlesource.com/src/+/8e55dca89f4e39241f9e3ecd25ab0ebbf5d1ab37).

## Installation

```shell
go get github.com/baabaaox/go-webrtcvad
```

## Project layout

```
.
├── cmd/
│   ├── offline_vad/          # Analyse recorded WAV files with the Go binding
│   └── realtime_gemini/      # Live transcription with WebRTC VAD + Gemini
├── example/voicebot/         # Larger voicebot reference implementation
├── test/                     # Sample audio clips for quick experiments
├── vad.go                    # Go binding to the WebRTC VAD C library
└── vad_test.go               # Binding smoke tests
```

The binaries under `cmd/` are intentionally tiny and showcase different ways to consume the binding. Feel free to copy them into your own projects as a starting point.

## Usage

### Offline analysis

```
go run ./cmd/offline_vad ./test/test2.wav
```

This loads a WAV file, keeps only the left channel (if stereo), resamples when necessary, and prints when speech is detected.

### Real-time Gemini transcription

```
export GEMINI_API_KEY=your-key
go run ./cmd/realtime_gemini -mic "audio=Microphone Array (Realtek(R) Audio)" -duration 15
```

You can disable the streaming preview if you only want the final transcript:

```
go run ./cmd/realtime_gemini -live=false
```

Flags available:

- `-mic` – exact DirectShow microphone name (run `ffmpeg -list_devices true -f dshow -i dummy` to inspect)
- `-lang` – Gemini language hint (defaults to `en`)
- `-model` – Gemini model (`gemini-2.0-flash` by default)
- `-duration` – recording length in seconds
- `-debug` – verbose progress output
- `-live` – enable or disable streaming transcripts while recording

### Library import

If you only need the VAD binding, add the module to your `go.mod` and call the package directly:

```go
vad := webrtcvad.Create()
defer webrtcvad.Free(vad)
if err := webrtcvad.Init(vad); err != nil {
	log.Fatal(err)
}
if err := webrtcvad.SetMode(vad, 2); err != nil {
	log.Fatal(err)
}
speech, err := webrtcvad.Process(vad, 16000, frame, frameLength)
```

See the binaries under `cmd/` for complete, runnable examples.

### Run my Application:

STEP:1 -> Navigate to the Folder
```
cd/cmd/realtime_gemini
```

STEP:2 -> Build the GO
```
go build -o main.exe main.go
```

STEP:3 -> Run the Application
```
.\main.exe -key "API-KEY"
```