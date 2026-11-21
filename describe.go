package main

import (
	"encoding/json"
	"errors"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strconv"
	"strings"
)

// Media vectors can't be ranked against text: measured on Workers AI, a plain
// sine tone outscored real matches for every query. So each attachment is
// turned into text (a caption, a transcript, or both), and that text is what
// default search embeds and ranks. Claude can also read it without opening
// the file.

const (
	imagePrompt = "Describe this image for a searchable engineering journal in 2-4 sentences: " +
		"name the app, screen, error, chart or scene. Then write a line 'Text:' and transcribe " +
		"all visible text verbatim and in full, every line. No preamble."
	videoPrompt = "These are %d frames from one %s video, in order and evenly spaced. " +
		"Describe what the video shows for a searchable engineering journal. " +
		"Transcribe visible text that matters. 2-4 sentences, no preamble."
	videoFrames = 8
)

// Describe returns a text description of an image, video or audio file.
func (e *AI) Describe(a Asset) (string, error) {
	switch a.Modality() {
	case "image":
		return e.Caption([]string{a.Path}, imagePrompt)
	case "audio":
		return e.describeAudio(a.Path)
	case "video":
		return e.describeVideo(a.Path)
	}
	return "", permanent{fmt.Errorf("can't describe %s", a.Mime)}
}

func (e *AI) describeAudio(path string) (string, error) {
	dir, err := os.MkdirTemp("", "jrnl-audio")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(dir)
	wav := filepath.Join(dir, "a.wav")
	if err := ffmpeg("-i", path, "-vn", "-ac", "1", "-ar", "16000", wav); err != nil {
		return "", err
	}
	t, err := transcribe(dir, wav)
	if err != nil {
		return "", err
	}
	if t == "" {
		return "Audio with no speech.", nil
	}
	return "Transcript:\n" + t, nil
}

func (e *AI) describeVideo(path string) (string, error) {
	dir, err := os.MkdirTemp("", "jrnl-video")
	if err != nil {
		return "", err
	}
	defer os.RemoveAll(dir)

	dur := probeDuration(path)
	rate := "1"
	length := "short"
	if dur > 0 {
		rate = strconv.FormatFloat(float64(videoFrames)/dur, 'f', 4, 64)
		length = fmt.Sprintf("%.0f-second", dur)
	}
	if err := ffmpeg("-i", path, "-vf", "fps="+rate+",scale=640:-2", "-frames:v", strconv.Itoa(videoFrames),
		filepath.Join(dir, "f%02d.jpg")); err != nil {
		return "", err
	}
	frames, _ := filepath.Glob(filepath.Join(dir, "f*.jpg"))
	if len(frames) == 0 {
		return "", permanent{errors.New("no frames could be read from the video")}
	}
	desc, err := e.Caption(frames, fmt.Sprintf(videoPrompt, len(frames), length))
	if err != nil {
		return "", err
	}
	// The sound track, if any, is transcribed too.
	wav := filepath.Join(dir, "a.wav")
	if ffmpeg("-i", path, "-vn", "-ac", "1", "-ar", "16000", wav) == nil {
		// A failed transcript must not throw away a good caption.
		var ns notSetUp
		if t, err := transcribe(dir, wav); errors.As(err, &ns) {
			// Not set up here; `jrnl doctor` says so.
		} else if err != nil {
			fmt.Fprintf(os.Stderr, "jrnl: video sound track not transcribed: %v\n", err)
		} else if t != "" {
			desc += "\n\nTranscript:\n" + t
		}
	}
	return desc, nil
}

const parakeetModel = "mlx-community/parakeet-tdt-0.6b-v3"

// transcribe runs NVIDIA Parakeet locally on Apple silicon, the same setup as
// the video-review skill. An 11-minute narration took 10.6 s in one call,
// complete to the last sentence; a pure tone gave no sentences. Each sentence
// keeps its start time, so a long recording can be opened at the right spot.
func transcribe(dir, wav string) (string, error) {
	// The uvx shim can point at a missing uv binary; mise always resolves it.
	cmd := []string{"mise", "exec", "uv", "--", "uvx"}
	if _, err := exec.LookPath("mise"); err != nil {
		if _, err := exec.LookPath("uvx"); err != nil {
			return "", notSetUp{errors.New("transcription needs mise or uvx (for parakeet-mlx)")}
		}
		cmd = []string{"uvx"}
	}
	cmd = append(cmd, "parakeet-mlx", "--model", parakeetModel, "--output-format", "json", "--output-dir", dir, wav)
	if out, err := exec.Command(cmd[0], cmd[1:]...).CombinedOutput(); err != nil {
		return "", permanent{fmt.Errorf("parakeet: %s", truncate(strings.TrimSpace(string(out)), 300))}
	}
	data, err := os.ReadFile(strings.TrimSuffix(wav, filepath.Ext(wav)) + ".json")
	if err != nil {
		return "", permanent{fmt.Errorf("parakeet wrote no transcript: %w", err)}
	}
	var res struct {
		Sentences []struct {
			Text  string  `json:"text"`
			Start float64 `json:"start"`
		} `json:"sentences"`
	}
	if err := json.Unmarshal(data, &res); err != nil {
		return "", permanent{fmt.Errorf("parakeet output: %w", err)}
	}
	var lines []string
	for _, st := range res.Sentences {
		if t := strings.TrimSpace(st.Text); t != "" {
			sec := int(st.Start)
			lines = append(lines, fmt.Sprintf("[%02d:%02d] %s", sec/60, sec%60, t))
		}
	}
	return strings.Join(lines, "\n"), nil
}

// ffmpeg failures mean the file can't be decoded; retrying won't help.
func ffmpeg(args ...string) error {
	if _, err := exec.LookPath("ffmpeg"); err != nil {
		return notSetUp{errors.New("ffmpeg is not installed")}
	}
	out, err := exec.Command("ffmpeg", append([]string{"-loglevel", "error", "-y"}, args...)...).CombinedOutput()
	if err != nil {
		return permanent{fmt.Errorf("ffmpeg: %s", truncate(strings.TrimSpace(string(out)), 200))}
	}
	return nil
}

func probeDuration(path string) float64 {
	out, err := exec.Command("ffprobe", "-v", "error", "-show_entries", "format=duration",
		"-of", "default=noprint_wrappers=1:nokey=1", path).Output()
	if err != nil {
		return 0
	}
	d, _ := strconv.ParseFloat(strings.TrimSpace(string(out)), 64)
	return d
}
