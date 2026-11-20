package main

import (
	"bytes"
	"encoding/base64"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"math"
	"net"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"sort"
	"strconv"
	"strings"
	"syscall"
	"time"
)

const (
	embedModel   = "@cf/google/embeddinggemma-2"   // media vectors, for search --like
	captionModel = "@cf/google/gemma-4-26b-a4b-it" // chat model of the Cloudflare preset
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
// for one text sent two ways), and nothing else would catch the mix-up. A
// text embedder is its model and the host serving it (embedderID).
const mediaEmbedder = "workers-ai/embeddinggemma-2/768"

// embedderID names a text embedder: the model and the host that serves it.
func embedderID(base, model string) string {
	host := base
	if u, err := url.Parse(base); err == nil && u.Host != "" {
		host = u.Host
	}
	h, port, err := net.SplitHostPort(host)
	if err == nil && (h == "127.0.0.1" || h == "::1") {
		host = net.JoinHostPort("localhost", port)
	}
	return model + "@" + host
}

// Input is one thing to embed: a document (Title and Text), a search query
// (Query set), or a media file.
type Input struct {
	Title, Text string
	Query       bool
	File, Mime  string
}

// format writes text the way the embedding model was trained to read it.
// Each family has its own convention, from its model card.
func format(model string, in Input) string {
	m := strings.ToLower(model)
	var t string
	switch {
	case strings.Contains(m, "embeddinggemma"):
		t = "title: " + in.Title + " | text: " + in.Text
		if in.Query {
			t = "task: search result | query: " + in.Text
		}
	case strings.Contains(m, "qwen3-embedding"):
		// Qwen3 instructs the query and leaves the document plain.
		t = in.Title + "\n\n" + in.Text
		if in.Query {
			t = "Instruct: Given a question about a software project, retrieve the journal entries that answer it\nQuery: " + in.Text
		}
	default:
		t = in.Title + "\n\n" + in.Text
		if in.Query {
			t = in.Text
		}
	}
	if len(t) > maxDocChars {
		t = t[:maxDocChars]
	}
	return t
}

// AI reaches the models jr uses, all optional:
//   - text embeddings: any OpenAI-compatible /embeddings API; by default
//     Ollama on this machine.
//   - chat, for captions, reranking and summaries: any OpenAI-compatible
//     /chat/completions API; Cloudflare Workers AI when only its credentials
//     are set.
//   - raw media vectors (search --like): Workers AI's own API, which no
//     standard covers.
type AI struct {
	embedURL, embedKey, embedModel, embedID string
	embedSet                                bool // JR_EMBED_URL was given

	chatURL, chatKey, chatModel string

	// A dedicated reranker with a Cohere-style /rerank API, optional.
	rerankURL, rerankKey, rerankModel string
	chatExtra                         map[string]any // merged into each chat request
	chatErr                           error          // set when no chat API is set up

	cfBase, cfToken string
	cfErr           error // set when Workers AI credentials are missing

	hc       *http.Client
	attempts int
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

// NewAI reads its settings from the environment:
//
//	JR_EMBED_URL, JR_EMBED_KEY, JR_EMBED_MODEL  text embeddings
//	    (default: Ollama at OLLAMA_HOST or localhost:11434, embeddinggemma-2)
//	JR_CHAT_URL, JR_CHAT_KEY, JR_CHAT_MODEL     chat
//	JR_CHAT_EXTRA                               JSON fields added to each chat request
//	JR_RERANK_URL, JR_RERANK_KEY, JR_RERANK_MODEL  a reranker with a
//	    Cohere-style /rerank API (Cohere, Voyage, Jina, Vercel AI Gateway)
//	CLOUDFLARE_ACCOUNT_ID, CLOUDFLARE_API_TOKEN Workers AI: search --like, and
//	    chat with Gemma 4 when JR_CHAT_URL is not set
//
// JR_AI_URL replaces the Workers AI root, .../accounts/<id>/ai (tests point
// it at a local server).
// ollamaURL is the OpenAI-compatible API of Ollama at OLLAMA_HOST.
func ollamaURL() string {
	host := os.Getenv("OLLAMA_HOST")
	if host == "" {
		host = "localhost:11434"
	}
	if !strings.Contains(host, "://") {
		host = "http://" + host
	}
	return strings.TrimSuffix(host, "/") + "/v1"
}

func NewAI() (*AI, error) {
	e := &AI{hc: &http.Client{Timeout: 3 * time.Minute}, attempts: 4}

	e.embedURL = strings.TrimSuffix(os.Getenv("JR_EMBED_URL"), "/")
	e.embedSet = e.embedURL != ""
	if !e.embedSet {
		e.embedURL = ollamaURL()
	}
	e.embedKey = os.Getenv("JR_EMBED_KEY")
	e.embedModel = os.Getenv("JR_EMBED_MODEL")
	if e.embedModel == "" {
		e.embedModel = ollamaModel
	}
	e.embedID = embedderID(e.embedURL, e.embedModel)

	root := os.Getenv("JR_AI_URL")
	e.cfToken = os.Getenv("CLOUDFLARE_API_TOKEN")
	if root == "" {
		acct := os.Getenv("CLOUDFLARE_ACCOUNT_ID")
		if acct == "" || e.cfToken == "" {
			e.cfErr = errNoWorkersAI
		}
		root = "https://api.cloudflare.com/client/v4/accounts/" + acct + "/ai"
	}
	e.cfBase = root + "/run"

	e.chatURL = strings.TrimSuffix(os.Getenv("JR_CHAT_URL"), "/")
	switch {
	case e.chatURL != "":
		e.chatKey, e.chatModel = os.Getenv("JR_CHAT_KEY"), os.Getenv("JR_CHAT_MODEL")
		if e.chatModel == "" {
			return nil, errors.New("JR_CHAT_URL is set, so set JR_CHAT_MODEL too")
		}
	case e.cfErr == nil:
		// Gemma 4 on Workers AI. With thinking on, its reasoning used the
		// whole token budget and the answer came back empty.
		e.chatURL, e.chatKey, e.chatModel = root+"/v1", e.cfToken, captionModel
		e.chatExtra = map[string]any{"chat_template_kwargs": map[string]any{"enable_thinking": false}}
	default:
		e.chatErr = errNoChat
	}
	e.rerankURL = strings.TrimSuffix(os.Getenv("JR_RERANK_URL"), "/")
	e.rerankKey, e.rerankModel = os.Getenv("JR_RERANK_KEY"), os.Getenv("JR_RERANK_MODEL")
	if e.rerankURL != "" && e.rerankModel == "" {
		return nil, errors.New("JR_RERANK_URL is set, so set JR_RERANK_MODEL too")
	}
	if x := os.Getenv("JR_CHAT_EXTRA"); x != "" {
		e.chatExtra = nil
		if err := json.Unmarshal([]byte(x), &e.chatExtra); err != nil {
			return nil, fmt.Errorf("JR_CHAT_EXTRA must be a JSON object: %w", err)
		}
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
			input[j] = format(e.embedModel, ins[i])
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
	body, _ := json.Marshal(map[string]any{"model": e.embedModel, "input": input})
	var res struct {
		Data []struct {
			Index     int       `json:"index"`
			Embedding []float32 `json:"embedding"`
		} `json:"data"`
	}
	err := e.post(e.embedURL+"/embeddings", e.embedKey, body, &res)
	var down unreachable
	var he httpError
	switch {
	case errors.As(err, &down):
		return nil, embedDown{e.embedURL, e.embedSet, err}
	case errors.As(err, &he) && he.status == 404 && !e.embedSet:
		return nil, fmt.Errorf("ollama has no %s model (run `ollama pull %s`)", e.embedModel, e.embedModel)
	case err != nil:
		return nil, fmt.Errorf("embeddings at %s: %w", e.embedURL, err)
	}
	out := make([][]float32, len(input))
	for _, d := range res.Data {
		if d.Index >= 0 && d.Index < len(out) {
			out[d.Index] = normalize(d.Embedding)
		}
	}
	for i, v := range out {
		if len(v) == 0 {
			return nil, fmt.Errorf("embeddings at %s: no vector for input %d", e.embedURL, i)
		}
	}
	return out, nil
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
	if e.cfErr != nil {
		return nil, e.cfErr
	}
	if err := e.post(e.cfBase+"/"+embedModel, e.cfToken, body, &res); err != nil {
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

// httpError is a reply other than 200, with the API's own message.
type httpError struct {
	status int
	msg    string
}

func (h httpError) Error() string { return fmt.Sprintf("HTTP %d: %s", h.status, h.msg) }

// unreachable means nothing listens at the address. Retrying won't help,
// and a missing local Ollama must not cost seconds of backoff on every add.
type unreachable struct{ error }

// post sends JSON to an API and decodes the reply. It retries rate limits,
// server errors and network failures with backoff, so a burst from many
// parallel sessions slows down instead of failing.
func (e *AI) post(url, key string, body []byte, out any) error {
	var last error
	for attempt := 0; attempt < e.attempts; attempt++ {
		if attempt > 0 {
			time.Sleep(time.Duration(1<<(attempt-1)) * time.Second)
		}
		req, _ := http.NewRequest("POST", url, bytes.NewReader(body))
		req.Header.Set("Content-Type", "application/json")
		if key != "" {
			req.Header.Set("Authorization", "Bearer "+key)
		}
		resp, err := e.hc.Do(req)
		if errors.Is(err, syscall.ECONNREFUSED) {
			return unreachable{err}
		}
		if err != nil {
			last = err
			continue
		}
		data, err := io.ReadAll(resp.Body)
		resp.Body.Close()
		if resp.StatusCode == 429 || resp.StatusCode >= 500 {
			last = httpError{resp.StatusCode, apiMessage(data)}
			continue
		}
		if err != nil {
			return err
		}
		if resp.StatusCode != 200 {
			err := httpError{resp.StatusCode, apiMessage(data)}
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

// apiMessage pulls the error message out of a reply in the shapes OpenAI,
// Cloudflare and Ollama use, or else returns the start of the body.
func apiMessage(data []byte) string {
	var r struct {
		Error  json.RawMessage `json:"error"`
		Errors []struct {
			Message string `json:"message"`
		} `json:"errors"`
	}
	if json.Unmarshal(data, &r) == nil {
		var m struct {
			Message string `json:"message"`
		}
		var str string
		switch {
		case json.Unmarshal(r.Error, &m) == nil && m.Message != "":
			return truncate(m.Message, 300)
		case json.Unmarshal(r.Error, &str) == nil && str != "":
			return truncate(str, 300)
		case len(r.Errors) > 0:
			return truncate(r.Errors[0].Message, 300)
		}
	}
	return truncate(strings.TrimSpace(string(data)), 300)
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

// Caption asks the chat model to describe one or more images (video frames are sent
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
		err = errors.New("empty caption")
	}
	return out, err
}

// chat sends one user message to the chat model and returns the reply text.
func (e *AI) chat(content any, maxTokens int) (string, error) {
	if e.chatErr != nil {
		return "", e.chatErr
	}
	req := map[string]any{
		"model":                 e.chatModel,
		"messages":              []any{map[string]any{"role": "user", "content": content}},
		"max_completion_tokens": maxTokens,
	}
	for k, v := range e.chatExtra {
		req[k] = v
	}
	body, err := json.Marshal(req)
	if err != nil {
		return "", err
	}
	var res struct {
		Choices []struct {
			FinishReason string `json:"finish_reason"`
			Message      struct {
				Content string `json:"content"`
			} `json:"message"`
		} `json:"choices"`
	}
	if err := e.post(e.chatURL+"/chat/completions", e.chatKey, body, &res); err != nil {
		return "", fmt.Errorf("chat at %s: %w", e.chatURL, err)
	}
	if len(res.Choices) == 0 {
		return "", fmt.Errorf("chat at %s: no reply", e.chatURL)
	}
	c := res.Choices[0]
	if c.FinishReason == "length" {
		fmt.Fprintf(os.Stderr, "jr: the reply hit its %d-token limit and is cut off\n", maxTokens)
	}
	return strings.TrimSpace(c.Message.Content), nil
}

// Rerank asks the chat model to order candidate entries by how well they answer the
// query, and returns up to k candidate positions, best first. On the 20,000
// entry test, this put the right entry first for 21 of 30 reworded questions,
// against 11 for keyword+vector fusion alone.
//
// k <= 0 asks for every relevant entry instead: for "the latest on X", search
// then sorts those by date. Asking for the 5 best of ~60 near-identical
// matches and sorting those missed the newest one on the 20,000-entry test.
func (e *AI) Rerank(query string, items []string, k int) ([]int, error) {
	// A dedicated reranker scores every candidate, but can't say that none
	// of them answers; "every relevant entry" (k <= 0) stays with chat.
	if e.rerankURL != "" && k > 0 {
		return e.rerankAPI(query, items, k)
	}
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

// rerankAPI calls a Cohere-style /rerank endpoint. Cohere, Jina and Vercel
// answer with "results", Voyage with "data"; both carry index and score.
func (e *AI) rerankAPI(query string, items []string, k int) ([]int, error) {
	body, _ := json.Marshal(map[string]any{"model": e.rerankModel, "query": query, "documents": items})
	type scored struct {
		Index int     `json:"index"`
		Score float64 `json:"relevance_score"`
	}
	var res struct {
		Results []scored `json:"results"`
		Data    []scored `json:"data"`
	}
	if err := e.post(e.rerankURL+"/rerank", e.rerankKey, body, &res); err != nil {
		return nil, fmt.Errorf("rerank at %s: %w", e.rerankURL, err)
	}
	got := append(res.Results, res.Data...)
	if len(got) == 0 {
		return nil, fmt.Errorf("rerank at %s: no scores", e.rerankURL)
	}
	sort.SliceStable(got, func(a, b int) bool { return got[a].Score > got[b].Score })
	var order []int
	seen := map[int]bool{}
	for _, g := range got {
		if g.Index >= 0 && g.Index < len(items) && !seen[g.Index] && len(order) < k {
			seen[g.Index] = true
			order = append(order, g.Index)
		}
	}
	return order, nil
}

// notSetUp marks a part of jr that this machine has not been set up for:
// no chat API, no Workers AI credentials, no ffmpeg, no parakeet. The work
// waits, quietly, for `jr reindex` after setup; `jr doctor` says how to turn
// each part on.
type notSetUp struct{ error }

var (
	errNoChat = notSetUp{errors.New("no chat model is set up (set JR_CHAT_URL, JR_CHAT_KEY and JR_CHAT_MODEL, " +
		"or CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN)")}
	errNoWorkersAI = notSetUp{errors.New("Workers AI is not set up (set CLOUDFLARE_ACCOUNT_ID and CLOUDFLARE_API_TOKEN)")}
)

// embedDown means nothing answered at the embeddings address. With no
// JR_EMBED_URL, no ollama installed and no text embedding in the journal yet,
// that is a machine without Ollama, and jr stays quiet; otherwise it was set
// up and is now broken.
type embedDown struct {
	addr string
	set  bool // JR_EMBED_URL was given
	err  error
}

func (d embedDown) Error() string {
	if d.set {
		return "nothing answers at JR_EMBED_URL " + d.addr
	}
	return "ollama is not reachable at " + strings.TrimSuffix(d.addr, "/v1") + " (start it with `ollama serve`)"
}
func (d embedDown) Unwrap() error { return d.err }

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
