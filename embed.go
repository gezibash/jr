package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"time"
)

const (
	embedModel   = "@cf/google/embeddinggemma-2" // media vectors, for search --like
	captionModel = "@cf/google/gemma-4-26b-a4b-it"
	// Text is embedded locally. On the 20,000-entry test, Ollama answered in
	// 40 ms (worst 0.85 s) against 1.5 s (worst 10 s, with 429s) on Workers AI,
	// and found as many right answers (21-22 of 30 against 21).
	ollamaModel = "embeddinggemma-2:270m"
	// Full size. On a 20,000-entry test, 256 dims put a reworded question's
	// answer first 2 times in 30 (typical rank 160th); 768 did 10 in 30
	// (typical rank 5th).
	embedDims = 768
	// The model's window is 8,192 tokens; this keeps long entries well inside it.
	maxDocChars = 24000
)

// Each vector records the embedder that made it, and search compares only
// vectors from the same one. Vectors from different embedders, or the same
// model reached another way, are not comparable (measured: cosine 0.74-0.81
// for one text sent two ways), and nothing else would catch the mix-up.
const (
	textEmbedder  = "ollama/" + ollamaModel + "/768"
	mediaEmbedder = "workers-ai/embeddinggemma-2/768"
)

// Input is one thing to embed: text, or a media file.
type Input struct {
	Text       string
	File, Mime string
}

// Task prefixes from the EmbeddingGemma 2 model card. Media gets no prefix.
func docText(title, body string) string {
	t := "title: " + title + " | text: " + body
	if len(t) > maxDocChars {
		t = t[:maxDocChars]
	}
	return t
}

func queryText(q string) string { return "task: search result | query: " + q }

// AI reaches the models jr uses: Ollama on this machine for text
// embeddings, Workers AI for captions, reranking and media embeddings.
type AI struct {
	base, token string
	cfErr       error // set when Workers AI credentials are missing
	ollama      string
	hc          *http.Client
	attempts    int
}

// Interactive returns a client for a search the user is waiting on: one
// attempt, hard deadline. Measured over 61 searches, 1 rerank call in 10
// took 7-13 s and the worst 22 s; past the deadline, search uses its
// fallback instead of waiting.
func (e *AI) Interactive(deadline time.Duration) *AI {
	c := *e
	c.hc = &http.Client{Timeout: deadline}
	c.attempts = 1
	return &c
}

// NewAI reads CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN for Workers AI,
// and OLLAMA_HOST (default localhost:11434). Missing Cloudflare credentials
// only disable the cloud calls; text embedding still works. JR_AI_URL
// replaces the Workers AI base URL (tests point it at a local server).
func NewAI() (*AI, error) {
	e := &AI{token: os.Getenv("CLOUDFLARE_API_TOKEN"), hc: &http.Client{Timeout: 3 * time.Minute}, attempts: 4}
	e.base = os.Getenv("JR_AI_URL")
	if e.base == "" {
		acct := os.Getenv("CLOUDFLARE_ACCOUNT_ID")
		if acct == "" || e.token == "" {
			e.cfErr = errors.New("set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN to use Workers AI")
		}
		e.base = "https://api.cloudflare.com/client/v4/accounts/" + acct + "/ai/run"
	}
	e.ollama = os.Getenv("OLLAMA_HOST")
	if e.ollama == "" {
		e.ollama = "localhost:11434"
	}
	if !strings.Contains(e.ollama, "://") {
		e.ollama = "http://" + e.ollama
	}
	return e, nil
}

type apiErrors struct {
	Success bool `json:"success"`
	Errors  []struct {
		Message string `json:"message"`
	} `json:"errors"`
}

func (r apiErrors) err() error {
	if r.Success {
		return nil
	}
	var m []string
	for _, er := range r.Errors {
		m = append(m, er.Message)
	}
	return fmt.Errorf("workers ai: %s", strings.Join(m, "; "))
}

// Embed sends text to Ollama and media to Workers AI. The two kinds of vector
// live in different spaces and are never compared with each other.
func (e *AI) Embed(ins []Input) ([][]float32, error) {
	var texts, media []int
	for i, in := range ins {
		if in.File == "" {
			texts = append(texts, i)
		} else {
			media = append(media, i)
		}
	}
	out := make([][]float32, len(ins))
	if len(texts) > 0 {
		input := make([]string, len(texts))
		for j, i := range texts {
			input[j] = ins[i].Text
		}
		vecs, err := e.embedText(input)
		if err != nil {
			return nil, err
		}
		for j, i := range texts {
			out[i] = vecs[j]
		}
	}
	if len(media) > 0 {
		msgs := make([]any, len(media))
		for j, i := range media {
			content, err := messageContent(ins[i])
			if err != nil {
				return nil, err
			}
			msgs[j] = []any{map[string]any{"role": "user", "content": content}}
		}
		vecs, err := e.embedRequest(map[string]any{"messages": msgs, "dimensions": embedDims}, len(media))
		if err != nil {
			return nil, err
		}
		for j, i := range media {
			out[i] = vecs[j]
		}
	}
	return out, nil
}

func (e *AI) embedText(input []string) ([][]float32, error) {
	body, _ := json.Marshal(map[string]any{
		"model": ollamaModel,
		"input": input,
		// Ollama's default window is 2,048 tokens and it cuts longer text
		// without saying so; 8,192 is the model's full window.
		"options":    map[string]any{"num_ctx": 8192},
		"keep_alive": "30m", // reloading after an unload took about 1 s
	})
	req, _ := http.NewRequest("POST", e.ollama+"/api/embed", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	resp, err := e.hc.Do(req)
	if err != nil {
		return nil, fmt.Errorf("ollama is not reachable at %s (start it with `ollama serve`): %w", e.ollama, err)
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	switch {
	case resp.StatusCode == 404:
		return nil, fmt.Errorf("ollama has no %s model (run `ollama pull %s`)", ollamaModel, ollamaModel)
	case resp.StatusCode == 400:
		return nil, permanent{fmt.Errorf("ollama: %s", truncate(string(data), 200))}
	case resp.StatusCode != 200:
		return nil, fmt.Errorf("ollama: HTTP %d: %s", resp.StatusCode, truncate(string(data), 200))
	}
	var res struct {
		Embeddings [][]float32 `json:"embeddings"`
	}
	if err := json.Unmarshal(data, &res); err != nil {
		return nil, err
	}
	if len(res.Embeddings) != len(input) {
		return nil, fmt.Errorf("ollama returned %d embeddings for %d inputs", len(res.Embeddings), len(input))
	}
	for i := range res.Embeddings {
		res.Embeddings[i] = normalize(res.Embeddings[i])
	}
	return res.Embeddings, nil
}

func (e *AI) embedRequest(req map[string]any, n int) ([][]float32, error) {
	body, err := json.Marshal(req)
	if err != nil {
		return nil, err
	}

	var res struct {
		apiErrors
		Result struct {
			Data []struct {
				Index     int       `json:"index"`
				Embedding []float32 `json:"embedding"`
			} `json:"data"`
		} `json:"result"`
	}
	if err := e.post(embedModel, body, &res); err != nil {
		return nil, err
	}
	if err := res.err(); err != nil {
		return nil, err
	}
	out := make([][]float32, n)
	for _, d := range res.Result.Data {
		if d.Index < 0 || d.Index >= len(out) {
			return nil, fmt.Errorf("workers ai: embedding index %d out of range", d.Index)
		}
		out[d.Index] = normalize(d.Embedding)
	}
	for i, v := range out {
		if len(v) == 0 {
			return nil, fmt.Errorf("workers ai: no embedding for input %d", i)
		}
	}
	return out, nil
}

// post retries rate limits, server errors and network failures with backoff,
// so a burst from many parallel sessions slows down instead of failing.
func (e *AI) post(model string, body []byte, out any) error {
	if e.cfErr != nil {
		return e.cfErr
	}
	var last error
	for attempt := 0; attempt < e.attempts; attempt++ {
		if attempt > 0 {
			time.Sleep(time.Duration(1<<(attempt-1)) * time.Second)
		}
		req, _ := http.NewRequest("POST", e.base+"/"+model, bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		if e.token != "" {
			req.Header.Set("Authorization", "Bearer "+e.token)
		}
		resp, err := e.hc.Do(req)
		if err != nil {
			last = err
			continue
		}
		data, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode == 429 || resp.StatusCode >= 500 {
			last = fmt.Errorf("workers ai: HTTP %d", resp.StatusCode)
			continue
		}
		if err != nil {
			return err
		}
		if resp.StatusCode != 200 {
			err := fmt.Errorf("workers ai: HTTP %d: %s", resp.StatusCode, truncate(string(data), 300))
			// Only statuses about the input itself are permanent. A bad token
			// (401/403) or a missing model (404) must stay retryable, or every
			// entry written meanwhile would be skipped by reindex forever.
			switch resp.StatusCode {
			case 400, 413, 415, 422:
				return permanent{err}
			}
			return err
		}
		return json.Unmarshal(data, out)
	}
	return last
}

func messageContent(in Input) (any, error) {
	if in.File == "" {
		return in.Text, nil
	}
	b, err := os.ReadFile(in.File)
	if err != nil {
		return nil, permanent{err}
	}
	enc := base64.StdEncoding.EncodeToString(b)
	uri := "data:" + in.Mime + ";base64," + enc
	switch {
	case strings.HasPrefix(in.Mime, "image/"):
		return []any{map[string]any{"type": "image_url", "image_url": map[string]string{"url": uri}}}, nil
	case strings.HasPrefix(in.Mime, "video/"):
		return []any{map[string]any{"type": "video_url", "video_url": map[string]string{"url": uri}}}, nil
	case strings.HasPrefix(in.Mime, "audio/"):
		return []any{map[string]any{"type": "audio_url", "audio_url": map[string]string{"url": uri}}}, nil
	}
	return nil, permanent{fmt.Errorf("cannot embed %s", in.Mime)}
}

// Caption asks Gemma 4 to describe one or more images (video frames are sent
// as images; the Workers AI build of Gemma 4 takes text and images only).
func (e *AI) Caption(images []string, prompt string) (string, error) {
	var content []any
	for _, path := range images {
		b, err := os.ReadFile(path)
		if err != nil {
			return "", permanent{err}
		}
		mt := mediaTypes[strings.ToLower(filepath.Ext(path))]
		content = append(content, map[string]any{"type": "image_url",
			"image_url": map[string]string{"url": "data:" + mt + ";base64," + base64.StdEncoding.EncodeToString(b)}})
	}
	content = append(content, map[string]any{"type": "text", "text": prompt})
	out, err := e.chat(content, 2048) // room to transcribe a full screen of text
	if err == nil && out == "" {
		err = errors.New("workers ai: empty caption")
	}
	return out, err
}

// chat sends one user message to Gemma 4 and returns the reply text.
func (e *AI) chat(content any, maxTokens int) (string, error) {
	body, err := json.Marshal(map[string]any{
		"messages":              []any{map[string]any{"role": "user", "content": content}},
		"max_completion_tokens": maxTokens,
		// With thinking on, the reasoning used the whole token budget and the
		// answer came back empty.
		"chat_template_kwargs": map[string]any{"enable_thinking": false},
	})
	if err != nil {
		return "", err
	}
	var res struct {
		apiErrors
		Result struct {
			Choices []struct {
				FinishReason string `json:"finish_reason"`
				Message      struct {
					Content string `json:"content"`
				} `json:"message"`
			} `json:"choices"`
		} `json:"result"`
	}
	if err := e.post(captionModel, body, &res); err != nil {
		return "", err
	}
	if err := res.err(); err != nil {
		return "", err
	}
	if len(res.Result.Choices) == 0 {
		return "", errors.New("workers ai: no reply")
	}
	c := res.Result.Choices[0]
	if c.FinishReason == "length" {
		fmt.Fprintf(os.Stderr, "jr: the reply hit its %d-token limit and is cut off\n", maxTokens)
	}
	return strings.TrimSpace(c.Message.Content), nil
}

// Rerank asks Gemma 4 to order candidate entries by how well they answer the
// query, and returns up to k candidate positions, best first. On the 20,000
// entry test, this put the right entry first for 21 of 30 reworded questions,
// against 11 for keyword+vector fusion alone.
//
// k <= 0 asks for every relevant entry instead: for "the latest on X", search
// then sorts those by date. Asking for the 5 best of ~60 near-identical
// matches and sorting those missed the newest one on the 20,000-entry test.
func (e *AI) Rerank(query string, items []string, k int) ([]int, error) {
	var b strings.Builder
	fmt.Fprintf(&b, "Query: %s\n\nJournal entries:\n", query)
	for i, it := range items {
		fmt.Fprintf(&b, "[%d] %s\n", i, it)
	}
	if k > 0 {
		fmt.Fprintf(&b, "\nRank the entries by how well they answer the query. Reply with only a JSON array "+
			"of the %d best entry numbers, best first, for example [3, 0, 7].", k)
	} else {
		b.WriteString("\nWhich entries are about the query? Reply with only a JSON array of the numbers " +
			"of every such entry, for example [3, 0, 7], or [] if none are.")
	}
	out, err := e.chat(b.String(), 400)
	if err != nil {
		return nil, err
	}
	var order []int
	seen := map[int]bool{}
	for _, f := range strings.FieldsFunc(out, func(r rune) bool { return r < '0' || r > '9' }) {
		n, err := strconv.Atoi(f)
		if err == nil && n < len(items) && !seen[n] {
			seen[n] = true
			order = append(order, n)
		}
	}
	// "[]" means none of the candidates answers the query. On the 20,000
	// entry test it came back when the right entry was not among them.
	if len(order) == 0 && strings.TrimSpace(out) != "[]" {
		return nil, fmt.Errorf("reranker gave no usable order: %q", truncate(out, 120))
	}
	return order, nil
}

// permanent marks an input that will fail the same way on every retry: the
// API rejected it, or the file is unreadable or of an unsupported type.
type permanent struct{ error }

func isPermanent(err error) bool {
	var p permanent
	return errors.As(err, &p)
}

// EmbedEach embeds a batch in one request. If the batch fails, it retries each
// input alone, so one bad attachment cannot block the others. Result i is
// either vecs[i] or errs[i].
//
// Text (local Ollama) and media (Workers AI) are embedded separately with
// separate errors, so a cloud outage never holds back text that Ollama has
// already embedded.
func (e *AI) EmbedEach(ins []Input) (vecs [][]float32, errs []error) {
	vecs, errs = make([][]float32, len(ins)), make([]error, len(ins))
	var text, media []int
	for i, in := range ins {
		if in.File == "" {
			text = append(text, i)
		} else {
			media = append(media, i)
		}
	}
	for _, group := range [][]int{text, media} {
		if len(group) == 0 {
			continue
		}
		sub := make([]Input, len(group))
		for j, i := range group {
			sub[j] = ins[i]
		}
		gv, ge := e.embedGroup(sub)
		for j, i := range group {
			vecs[i], errs[i] = gv[j], ge[j]
		}
	}
	return vecs, errs
}

func (e *AI) embedGroup(ins []Input) (vecs [][]float32, errs []error) {
	vecs, errs = make([][]float32, len(ins)), make([]error, len(ins))
	all, err := e.Embed(ins)
	if err == nil {
		return all, errs
	}
	// A transient failure (outage, rate limit after retries) hits every input
	// alike; retrying them one by one would only multiply the wait.
	if len(ins) == 1 || !isPermanent(err) {
		for i := range errs {
			errs[i] = err
		}
		return vecs, errs
	}
	for i, in := range ins {
		v, err := e.Embed([]Input{in})
		if err != nil {
			errs[i] = err
			continue
		}
		vecs[i] = v[0]
	}
	return vecs, errs
}

func normalize(v []float32) []float32 {
	var sum float64
	for _, f := range v {
		sum += float64(f) * float64(f)
	}
	if sum == 0 {
		return v
	}
	n := float32(1 / math.Sqrt(sum))
	for i := range v {
		v[i] *= n
	}
	return v
}

func dot(a, b []float32) float32 {
	var s float32
	for i := range a {
		if i >= len(b) {
			break
		}
		s += a[i] * b[i]
	}
	return s
}

func truncate(s string, n int) string {
	r := []rune(s)
	if len(r) <= n {
		return s
	}
	return string(r[:n]) + "…"
}
