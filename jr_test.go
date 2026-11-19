package main

// Tests drive the real jr binary against a fake Workers AI server, the same
// boundary Claude uses.

import (
	"database/sql"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"os"
	"os/exec"
	"path/filepath"
	"regexp"
	"slices"
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

var bin string

func TestMain(m *testing.M) {
	dir, err := os.MkdirTemp("", "jr-test-bin")
	if err != nil {
		panic(err)
	}
	bin = filepath.Join(dir, "jr")
	if out, err := exec.Command("go", "build", "-o", bin, ".").CombinedOutput(); err != nil {
		panic(string(out))
	}
	code := m.Run()
	os.RemoveAll(dir)
	os.Exit(code)
}

// fakeAI stands in for the two Workers AI models jr calls.
//
// Embeddings are hand-built on concept axes, so a text and an image can be
// "about" the same thing without sharing a word:
//
//	axis 0: signing in   (words: sign, login, authentication, credentials)
//	axis 1: images       (words: screenshot, picture)
//	axis 9: anything else
//
// Queries point straight along their concept. Documents only half-match
// theirs (weight 0.5, the rest on a noise axis), and every raw image is a
// "hub" that sits 0.577 from every concept. That reproduces what Workers AI
// measured: a meaningless tone outscored real text matches for every query.
//
// Captions (Gemma) describe a fake image as "A screenshot showing <bytes after
// the PNG magic>".
type fakeAI struct {
	fail         atomic.Bool
	unauth       atomic.Bool  // answer 401, like a revoked token
	maxMedia     atomic.Int64 // largest raw media payload sent for embedding
	rerankFail   atomic.Bool  // reranking calls answer 500
	rerankDelay  atomic.Int64 // reranking calls stall this many seconds first
	summaryCalls atomic.Int64
	mu           sync.Mutex
	texts        []string
	dims         []int
	ollamaReqs   []string // "model ctx=N" per Ollama request
	requests     atomic.Int64
}

func (f *fakeAI) vector(content any) []float32 {
	v := make([]float32, 256)
	if parts, ok := content.([]any); ok {
		// Raw media: the hub vector.
		b, _ := json.Marshal(parts)
		if n := int64(len(b)); n > f.maxMedia.Load() {
			f.maxMedia.Store(n)
		}
		v[0], v[1], v[9] = 0.577, 0.577, 0.577
		return v
	}
	text := strings.ToLower(content.(string))
	f.mu.Lock()
	f.texts = append(f.texts, content.(string))
	f.mu.Unlock()
	axis := 9
	for _, w := range []string{"sign", "login", "authentication", "credentials"} {
		if strings.Contains(text, w) {
			axis = 0
		}
	}
	for _, w := range []string{"screenshot", "picture"} {
		if strings.Contains(text, w) {
			axis = 1
		}
	}
	if strings.HasPrefix(text, "task: search result") {
		v[axis] = 1
		return v
	}
	v[axis], v[20+len(text)%200] = 0.5, 0.866
	return v
}

func (f *fakeAI) ServeHTTP(w http.ResponseWriter, r *http.Request) {
	f.requests.Add(1)
	// fail and unauth model Workers AI trouble; Ollama is local and unaffected.
	if r.URL.Path == "/api/embed" { // Ollama, on this machine
		var req struct {
			Model   string         `json:"model"`
			Input   []string       `json:"input"`
			Options map[string]any `json:"options"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		f.mu.Lock()
		f.ollamaReqs = append(f.ollamaReqs, fmt.Sprintf("%s ctx=%v", req.Model, req.Options["num_ctx"]))
		f.mu.Unlock()
		var embs [][]float32
		for _, t := range req.Input {
			embs = append(embs, f.vector(t))
		}
		json.NewEncoder(w).Encode(map[string]any{"embeddings": embs})
		return
	}
	if f.fail.Load() {
		http.Error(w, "boom", 500)
		return
	}
	if f.unauth.Load() {
		http.Error(w, `{"success":false,"errors":[{"message":"Authentication error"}]}`, 401)
		return
	}
	reply := func(result any) {
		json.NewEncoder(w).Encode(map[string]any{"success": true, "result": result})
	}
	switch {
	case strings.HasSuffix(r.URL.Path, "/gemma-4-26b-a4b-it"):
		var req struct {
			Messages []struct {
				Content any `json:"content"`
			} `json:"messages"`
		}
		json.NewDecoder(r.Body).Decode(&req)
		if prompt, ok := req.Messages[0].Content.(string); ok && (strings.HasPrefix(prompt, "Summarize") || strings.HasPrefix(prompt, "Combine")) {
			// Summaries: one bullet citing every [ID] the prompt contains,
			// so a test can see whether any entry was dropped.
			f.summaryCalls.Add(1)
			// One bullet per entry number in the prompt (or per citation in
			// part summaries being combined), plus a number that matches no
			// entry, which jr must drop.
			var lines []string
			for _, m := range regexp.MustCompile(`(?m)^\[(\d+)\]|- item \[(\d+)\]`).FindAllStringSubmatch(prompt, -1) {
				lines = append(lines, "- item ["+m[1]+m[2]+"]")
			}
			lines = append(lines, "- invented [99999]")
			reply(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": strings.Join(lines, "\n")}}}})
			return
		}
		if prompt, ok := req.Messages[0].Content.(string); ok {
			// Reranking: entries containing "rerank-me" go first, the rest
			// keep their order.
			if f.rerankFail.Load() {
				http.Error(w, "boom", 500)
				return
			}
			time.Sleep(time.Duration(f.rerankDelay.Load()) * time.Second)
			var first, rest []int
			for _, line := range strings.Split(prompt, "\n") {
				var i int
				if _, err := fmt.Sscanf(line, "[%d]", &i); err == nil {
					if strings.Contains(strings.ToLower(line), "rerank-me") {
						first = append(first, i)
					} else {
						rest = append(rest, i)
					}
				}
			}
			order, _ := json.Marshal(append(first, rest...))
			reply(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": string(order)}}}})
			return
		}
		parts := req.Messages[0].Content.([]any)
		url := parts[0].(map[string]any)["image_url"].(map[string]any)["url"].(string)
		raw, _ := base64.StdEncoding.DecodeString(url[strings.IndexByte(url, ',')+1:])
		caption := "A screenshot showing " + strings.TrimPrefix(string(raw), "\x89PNG ")
		reply(map[string]any{"choices": []any{map[string]any{"message": map[string]any{"content": caption}}}})
		return
	case !strings.HasSuffix(r.URL.Path, "/embeddinggemma-2"):
		http.Error(w, "unknown model "+r.URL.Path, 404)
		return
	}
	var req struct {
		Input      []string           `json:"input"`
		Messages   [][]map[string]any `json:"messages"`
		Dimensions int                `json:"dimensions"`
	}
	if err := json.NewDecoder(r.Body).Decode(&req); err != nil {
		http.Error(w, err.Error(), 400)
		return
	}
	// Text sent through "messages" embeds differently from text sent through
	// "input" (cosine 0.74-0.81 on the real API), so jr must never do it.
	for _, m := range req.Messages {
		if _, isText := m[0]["content"].(string); isText {
			http.Error(w, `{"success":false,"errors":[{"message":"text sent on the messages path"}]}`, 400)
			return
		}
	}
	// Like the real API rejecting an input it can't decode: any request
	// carrying a video fails as a whole with a 400.
	for _, m := range req.Messages {
		if parts, ok := m[0]["content"].([]any); ok && parts[0].(map[string]any)["type"] == "video_url" {
			http.Error(w, `{"success":false,"errors":[{"message":"cannot decode video"}]}`, 400)
			return
		}
	}
	f.mu.Lock()
	f.dims = append(f.dims, req.Dimensions)
	f.mu.Unlock()
	type item struct {
		Index     int       `json:"index"`
		Embedding []float32 `json:"embedding"`
	}
	var data []item
	for i, t := range req.Input {
		data = append(data, item{i, f.vector(t)})
	}
	for i, m := range req.Messages {
		data = append(data, item{i, f.vector(m[0]["content"])})
	}
	reply(map[string]any{"data": data})
}

type env struct {
	t      *testing.T
	home   string
	ai     *fakeAI
	url    string
	ollama string // OLLAMA_HOST; the fake server unless a test changes it
}

func newEnv(t *testing.T) *env {
	ai := &fakeAI{}
	srv := httptest.NewServer(ai)
	t.Cleanup(srv.Close)
	return &env{t: t, home: t.TempDir(), ai: ai, url: srv.URL, ollama: srv.URL}
}

func (e *env) run(stdin string, args ...string) (string, string, error) {
	cmd := exec.Command(bin, append(args, "--project", "test")...)
	cmd.Env = append(os.Environ(), "JR_HOME="+e.home, "JR_AI_URL="+e.url, "OLLAMA_HOST="+e.ollama,
		// A fixed clock and zone: Thursday 2026-11-19, 15:30 in Berlin (UTC+1).
		"TZ=Europe/Berlin", "JR_NOW=2026-11-19T15:30:00+01:00", "CLAUDECODE=", "CLAUDE_CODE_SESSION_ID=sess-1")
	if stdin != "" {
		cmd.Stdin = strings.NewReader(stdin)
	}
	var out, errb strings.Builder
	cmd.Stdout, cmd.Stderr = &out, &errb
	err := cmd.Run()
	return out.String(), errb.String(), err
}

func (e *env) must(stdin string, args ...string) string {
	e.t.Helper()
	out, errOut, err := e.run(stdin, args...)
	if err != nil {
		e.t.Fatalf("jr %v: %v\n%s", args, err, errOut)
	}
	return out
}

func firstID(out string) string { return strings.Fields(out)[0] }

// Many sessions writing to one project at once must not lose or reject a
// single entry, including the very first runs that create the database.
func TestParallelSessionsLoseNothing(t *testing.T) {
	e := newEnv(t)
	const procs, each = 20, 25
	var wg sync.WaitGroup
	errs := make(chan string, procs*each)
	for p := 0; p < procs; p++ {
		wg.Add(1)
		go func(p int) {
			defer wg.Done()
			for i := 0; i < each; i++ {
				if _, errOut, err := e.run("", "add", "--kind", "log", fmt.Sprintf("session %d entry %d", p, i)); err != nil {
					errs <- errOut
				}
			}
		}(p)
	}
	wg.Wait()
	close(errs)
	failed := 0
	for msg := range errs {
		if failed == 0 {
			t.Errorf("first failure: %s", msg)
		}
		failed++
	}
	if failed > 0 {
		t.Fatalf("%d of %d adds failed", failed, procs*each)
	}
	out := e.must("", "list", "--limit", "100000")
	if n := strings.Count(out, "\n"); n != procs*each {
		t.Fatalf("listed %d entries, want %d", n, procs*each)
	}
}

func TestSearchFindsByKeywordAndByMeaning(t *testing.T) {
	e := newEnv(t)
	// No word in common with the query below; only meaning links them.
	auth := firstID(e.must("Users could not sign in after the token refresh.", "add", "--kind", "bug", "--tag", "auth"))
	busy := firstID(e.must("Writers hit SQLITE_BUSY when two sessions started together.", "add", "--kind", "bug"))
	firstID(e.must("Chose Go over Bun for the CLI.", "add", "--kind", "decision"))

	out := e.must("", "search", "authentication failure")
	if !strings.HasPrefix(out, auth) {
		t.Errorf("semantic query: want %s first, got:\n%s", auth, out)
	}
	out = e.must("", "search", "SQLITE_BUSY")
	if !strings.HasPrefix(out, busy) || !strings.Contains(strings.SplitN(out, "\n", 2)[0], "keyword") {
		t.Errorf("keyword query: want %s first via keyword, got:\n%s", busy, out)
	}
	// Filters apply to both rankings.
	out = e.must("", "search", "--kind", "decision", "authentication failure")
	if strings.Contains(out, auth) {
		t.Errorf("--kind decision returned a bug entry:\n%s", out)
	}
	// Spec: documents carry the document prefix, queries the query prefix,
	// and both use the same dimension.
	for _, txt := range e.ai.texts {
		if !strings.HasPrefix(txt, "title: ") && !strings.HasPrefix(txt, "task: search result | query: ") {
			t.Errorf("text sent without a task prefix: %q", txt)
		}
	}
	// Text goes to the local model with its full 8,192-token window (Ollama's
	// default of 2,048 cuts long text without saying so).
	if len(e.ai.ollamaReqs) == 0 {
		t.Error("no text was embedded through Ollama")
	}
	for _, r := range e.ai.ollamaReqs {
		if r != "embeddinggemma-2:270m ctx=8192" {
			t.Errorf("ollama request %q, want embeddinggemma-2:270m ctx=8192", r)
		}
	}
}

func TestTextQueryFindsImage(t *testing.T) {
	e := newEnv(t)
	img := filepath.Join(t.TempDir(), "broken-header.png")
	os.WriteFile(img, []byte("\x89PNG fake bytes"), 0o644)
	id := firstID(e.must("Header overlaps the nav on mobile.", "add", "--attach", img+"::header overlapping nav"))
	e.must("Bumped the Go version.", "add")

	out := e.must("", "search", "picture")
	first := strings.SplitN(out, "\n", 2)[0]
	if !strings.HasPrefix(first, id) {
		t.Fatalf("want %s first, got:\n%s", id, out)
	}
	if !strings.Contains(out, "matched image ") || !strings.Contains(out, filepath.Join(e.home, "test", "assets")) {
		t.Errorf("search should name the matching attachment's stored path:\n%s", out)
	}
	show := e.must("", "show", id[:12])
	if !strings.Contains(show, "header overlapping nav") {
		t.Errorf("show is missing the caption:\n%s", show)
	}
	if !strings.Contains(show, "description: A screenshot showing fake bytes") {
		t.Errorf("show is missing the generated description:\n%s", show)
	}
}

// A Workers AI outage must not hold back text: the entry is searchable by
// meaning at once (Ollama embedded it), and its attachment is described and
// made searchable by reindex once the cloud is back.
func TestCloudOutageDoesNotBlockText(t *testing.T) {
	e := newEnv(t)
	img := filepath.Join(t.TempDir(), "shot.png")
	os.WriteFile(img, []byte("\x89PNG settings page"), 0o644)
	e.ai.fail.Store(true)
	out, errOut, err := e.run("Login form rejects valid credentials.", "add", "--attach", img)
	if err != nil {
		t.Fatalf("add failed during outage: %v\n%s", err, errOut)
	}
	id := firstID(out)
	if !strings.Contains(errOut, "reindex") {
		t.Errorf("add should say the attachment waits for reindex: %q", errOut)
	}
	if out := e.must("", "search", "authentication problem"); !strings.HasPrefix(out, id) {
		t.Fatalf("text not searchable by meaning during a cloud outage:\n%s", out)
	}
	if out := e.must("", "search", "picture"); strings.Contains(out, "matched image") {
		t.Fatalf("image matched before it was described:\n%s", out)
	}

	e.ai.fail.Store(false)
	if out := e.must("", "reindex"); !strings.Contains(out, "described 1 file(s)") {
		t.Fatalf("reindex: %s", out)
	}
	if out := e.must("", "search", "picture"); !strings.Contains(out, "matched image") {
		t.Fatalf("image not searchable after reindex:\n%s", out)
	}
}

// One attachment the API rejects must not leave the rest of the entry
// unsearchable, and reindex must not retry it forever.
func TestRejectedAttachmentDoesNotBlockOthers(t *testing.T) {
	e := newEnv(t)
	dir := t.TempDir()
	img, vid, txt := filepath.Join(dir, "shot.png"), filepath.Join(dir, "clip.mp4"), filepath.Join(dir, "notes.txt")
	os.WriteFile(img, []byte("\x89PNG fake"), 0o644)
	os.WriteFile(vid, []byte("fake mp4"), 0o644)
	os.WriteFile(txt, []byte("plain notes"), 0o644)

	out, errOut, err := e.run("Login button does nothing.", "add", "--attach", img, "--attach", vid, "--attach", txt)
	if err != nil {
		t.Fatalf("add: %v\n%s", err, errOut)
	}
	id := firstID(out)
	if !strings.Contains(errOut, "clip.mp4 can't be embedded") {
		t.Errorf("stderr should name the rejected video: %q", errOut)
	}
	if !strings.Contains(errOut, "notes.txt stored but not searchable") {
		t.Errorf("stderr should warn about the unsupported type: %q", errOut)
	}
	// Text (axis 0, "login") and image (axis 1) were still embedded.
	if out := e.must("", "search", "authentication"); !strings.HasPrefix(out, id) {
		t.Errorf("text not searchable after a rejected sibling:\n%s", out)
	}
	if out := e.must("", "search", "picture"); !strings.Contains(out, "matched image") {
		t.Errorf("image not searchable after a rejected sibling:\n%s", out)
	}
	before := e.ai.requests.Load()
	if out := e.must("", "reindex"); strings.TrimSpace(out) != "described 0 file(s), embedded 0 pending chunk(s)" {
		t.Errorf("reindex should have nothing left to try, got %q", out)
	}
	if n := e.ai.requests.Load() - before; n != 0 {
		t.Errorf("reindex sent %d requests for a permanently rejected input", n)
	}
}

// Raw media vectors sit close to every query (the fake's hub, the real
// model's tone). Default search must rank on text and descriptions only, and
// --like must still find media by media.
func TestRawMediaDoesNotDrownText(t *testing.T) {
	e := newEnv(t)
	img := filepath.Join(t.TempDir(), "chart.png")
	os.WriteFile(img, []byte("\x89PNG quarterly chart"), 0o644)
	withImage := firstID(e.must("Bumped the Go version.", "add", "--attach", img))
	login := firstID(e.must("Users could not sign in after deploy.", "add"))

	out := e.must("", "search", "authentication")
	if !strings.HasPrefix(out, login) {
		t.Fatalf("text match should rank first, not the attachment hub:\n%s", out)
	}
	out = e.must("", "search", "--like", img)
	if !strings.HasPrefix(out, withImage) || strings.Contains(out, login) {
		t.Fatalf("--like should return only the entry with a similar image:\n%s", out)
	}
}

// A revoked token is not the input's fault. Entries written while it was bad
// must be processed once the token works again.
func TestBadTokenDoesNotPoisonEntries(t *testing.T) {
	e := newEnv(t)
	img := filepath.Join(t.TempDir(), "login.png")
	os.WriteFile(img, []byte("\x89PNG login dialog"), 0o644)
	e.ai.unauth.Store(true)
	id := firstID(e.must("Session expired banner shows twice.", "add", "--attach", img))
	e.ai.unauth.Store(false)
	out := e.must("", "reindex")
	if !strings.Contains(out, "described 1 file(s)") || strings.Contains(out, "never") {
		t.Fatalf("reindex after the token was fixed: %q", out)
	}
	if out := e.must("", "search", "picture"); !strings.HasPrefix(out, id) {
		t.Fatalf("entry not searchable after the token was fixed:\n%s", out)
	}
}

// A huge file must not be base64'd into one embedding request (a 941 MB
// video peaked at 7.6 GB of memory), but must stay findable by description.
func TestHugeAttachmentSkipsRawVector(t *testing.T) {
	e := newEnv(t)
	big := filepath.Join(t.TempDir(), "huge.png")
	f, _ := os.Create(big)
	f.WriteString("\x89PNG giant diagram")
	f.Truncate(51 << 20) // sparse: cheap to create
	f.Close()
	out, errOut, err := e.run("Architecture overview.", "add", "--attach", big)
	if err != nil {
		t.Fatalf("add: %v\n%s", err, errOut)
	}
	if n := e.ai.maxMedia.Load(); n > 0 {
		t.Errorf("sent %d bytes of raw media for embedding; want none over the cap", n)
	}
	if got := e.must("", "search", "picture"); !strings.HasPrefix(got, firstID(out)) {
		t.Errorf("huge attachment should still be found by its description:\n%s", got)
	}
}

// Common words must not count as keyword matches: with OR between terms,
// "a beeping tone" made every entry containing "a" a keyword hit.
func TestStopwordsAreNotKeywordHits(t *testing.T) {
	e := newEnv(t)
	e.must("Fixed a crash in the parser.", "add")
	e.must("Moved the docs to a new folder.", "add")
	out := e.must("", "search", "what is the a of zzzqx")
	if strings.Contains(out, "keyword") {
		t.Fatalf("stopwords alone produced keyword hits:\n%s", out)
	}
}

// Search must follow the reranker's order, and fall back to keyword+vector
// order when the reranker fails.
func TestSearchFollowsRerankerAndSurvivesItsFailure(t *testing.T) {
	e := newEnv(t)
	login := firstID(e.must("Login rejects valid credentials after deploy.", "add"))
	other := firstID(e.must("Moved the docs folder (rerank-me).", "add"))

	out := e.must("", "search", "login credentials")
	if !strings.HasPrefix(out, other) || !strings.Contains(strings.SplitN(out, "\n", 2)[0], "reranked") {
		t.Fatalf("reranker's pick should come first, marked reranked:\n%s", out)
	}

	e.ai.rerankFail.Store(true)
	out, errOut, err := e.run("", "search", "login credentials")
	if err != nil {
		t.Fatalf("search failed when the reranker did: %v\n%s", err, errOut)
	}
	if !strings.HasPrefix(out, login) || !strings.Contains(errOut, "not reranked") {
		t.Fatalf("want fused order with a warning; stderr %q, got:\n%s", errOut, out)
	}
}

// A stalled or failing reranker must not hold a search hostage: no retries
// with backoff, and a deadline after which the keyword+vector order is used.
func TestSearchDoesNotWaitOnSlowReranker(t *testing.T) {
	e := newEnv(t)
	login := firstID(e.must("Login rejects valid credentials after deploy.", "add"))
	e.must("Moved the docs folder (rerank-me).", "add")

	e.ai.rerankDelay.Store(10)
	start := time.Now()
	out, errOut, err := e.run("", "search", "login credentials")
	took := time.Since(start)
	if err != nil {
		t.Fatalf("search: %v\n%s", err, errOut)
	}
	if took > 8*time.Second {
		t.Errorf("search waited %s on a stalled reranker", took.Round(time.Second))
	}
	if !strings.HasPrefix(out, login) || !strings.Contains(errOut, "not reranked") {
		t.Errorf("want fused order with a warning; stderr %q, got:\n%s", errOut, out)
	}

	e.ai.rerankDelay.Store(0)
	e.ai.rerankFail.Store(true)
	start = time.Now()
	e.must("", "search", "login credentials")
	if took := time.Since(start); took > 3*time.Second {
		t.Errorf("search retried a failing reranker for %s", took.Round(time.Second))
	}
}

// With Ollama down, entries are still saved and search still answers by
// keyword; once Ollama is back, reindex makes them findable by meaning.
// There is no silent fallback to another embedder.
func TestOllamaDown(t *testing.T) {
	e := newEnv(t)
	e.ollama = "http://127.0.0.1:1" // nothing listens here
	out, errOut, err := e.run("Login form rejects valid credentials.", "add")
	if err != nil {
		t.Fatalf("add failed with Ollama down: %v\n%s", err, errOut)
	}
	id := firstID(out)
	if !strings.Contains(errOut, "ollama") || !strings.Contains(errOut, "reindex") {
		t.Errorf("add should say Ollama is down and to reindex: %q", errOut)
	}
	out, errOut, _ = e.run("", "search", "Login")
	if !strings.HasPrefix(out, id) || !strings.Contains(errOut, "keyword search only") {
		t.Errorf("search should fall back to keywords; stderr %q:\n%s", errOut, out)
	}

	e.ollama = e.url
	if out := e.must("", "reindex"); !strings.Contains(out, "embedded 1 ") {
		t.Fatalf("reindex after Ollama came back: %q", out)
	}
	if out := e.must("", "search", "authentication problem"); !strings.HasPrefix(out, id) {
		t.Fatalf("not findable by meaning after reindex:\n%s", out)
	}
}

// Vectors made by another embedder must never be compared with the query.
func TestVectorsFromAnotherEmbedderAreIgnored(t *testing.T) {
	e := newEnv(t)
	id := firstID(e.must("Login form rejects valid credentials.", "add"))
	if out := e.must("", "search", "authentication problem"); !strings.HasPrefix(out, id) {
		t.Fatalf("setup: not found by meaning:\n%s", out)
	}
	db, err := sql.Open("sqlite", filepath.Join(e.home, "test", "journal.db"))
	if err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(`UPDATE chunks SET embedder = 'some-other/model/768'`); err != nil {
		t.Fatal(err)
	}
	db.Close()
	if out := e.must("", "search", "authentication problem"); strings.Contains(out, id) {
		t.Fatalf("matched a vector from another embedder:\n%s", out)
	}
}

func ids(out string) []string {
	var got []string
	for _, l := range strings.Split(out, "\n") {
		if l != "" && !strings.HasPrefix(l, " ") {
			got = append(got, strings.Fields(l)[0])
		}
	}
	return got
}

func sameSet(t *testing.T, what string, got []string, want ...string) {
	t.Helper()
	g, w := strings.Join(slices.Sorted(slices.Values(got)), " "), strings.Join(slices.Sorted(slices.Values(want)), " ")
	if g != w {
		t.Errorf("%s:\n got  %s\n want %s", what, g, w)
	}
}

// "Yesterday", "last week" and friends are local calendar days. The clock is
// Thursday 2026-11-19 in Berlin; 00:30 local on the 18th is 23:30 UTC on the
// 17th, which a UTC day boundary gets wrong.
func TestTimeFilters(t *testing.T) {
	e := newEnv(t)
	at := func(when, body string) string { return firstID(e.must(body, "add", "--at", when)) }
	sun8 := at("2026-11-08T12:00:00+01:00", "sunday before last")
	mon9 := at("2026-11-09T10:00:00+01:00", "monday last week")
	sun15 := at("2026-11-15T22:00:00+01:00", "sunday night last week")
	mon16 := at("2026-11-16T08:00:00+01:00", "monday this week")
	tue17 := at("2026-11-17T23:30:00+01:00", "tuesday late")
	wedEarly := at("2026-11-18T00:30:00+01:00", "wednesday just after midnight")
	wedLate := at("2026-11-18T23:30:00+01:00", "wednesday late")
	thu := at("2026-11-19T00:10:00+01:00", "thursday just after midnight")
	oct := at("2026-10-31T23:59:00+01:00", "last day of october")

	list := func(args ...string) []string {
		return ids(e.must("", append([]string{"list", "--limit", "100"}, args...)...))
	}
	sameSet(t, "--on yesterday", list("--on", "yesterday"), wedEarly, wedLate)
	sameSet(t, "--since yesterday --until yesterday", list("--since", "yesterday", "--until", "yesterday"), wedEarly, wedLate)
	sameSet(t, "--on today", list("--on", "today"), thu)
	sameSet(t, "--on '2 days ago'", list("--on", "2 days ago"), tue17)
	sameSet(t, "--since monday", list("--since", "monday"), mon16, tue17, wedEarly, wedLate, thu)
	sameSet(t, "--on 'last week'", list("--on", "last week"), mon9, sun15)
	sameSet(t, "--until 'last week'", list("--until", "last week"), sun8, mon9, sun15, oct)
	sameSet(t, "--on october", list("--on", "october"), oct)

	got := list("--oldest", "--limit", "2")
	if len(got) != 2 || got[0] != oct || got[1] != sun8 {
		t.Errorf("--oldest: got %v, want [%s %s]", got, oct, sun8)
	}
	if _, errOut, err := e.run("", "list", "--on", "36h"); err == nil || !strings.Contains(errOut, "--on needs a day") {
		t.Errorf("--on with a point in time should be refused, stderr %q", errOut)
	}
}

// "What's the latest on X" and "when did X first happen" must find the newest
// and oldest matches, even when older entries match the words better.
func TestSearchNewestAndOldest(t *testing.T) {
	e := newEnv(t)
	first := firstID(e.must("Redis cluster went read-only after a long afternoon of unrelated chores on the reporting dashboards, the billing exports and the onboarding emails.", "add", "--at", "2026-08-01"))
	for i := 0; i < 35; i++ {
		e.must("Redis cluster: Redis cluster.", "add", "--at", fmt.Sprintf("2026-09-%02d", i%28+1))
	}
	latest := firstID(e.must("Redis cluster upgraded after a long afternoon of unrelated chores on the reporting dashboards, the billing exports and the onboarding emails.", "add", "--at", "2026-11-19T09:00:00+01:00"))

	if got := ids(e.must("", "search", "--newest", "--limit", "5", "Redis cluster")); len(got) == 0 || got[0] != latest {
		t.Errorf("--newest: got %v, want %s first", got, latest)
	}
	if got := ids(e.must("", "search", "--oldest", "--limit", "5", "Redis cluster")); len(got) == 0 || got[0] != first {
		t.Errorf("--oldest: got %v, want %s first", got, first)
	}
}

// "What did the last session do" means the session before the current one
// (the test runs as sess-1).
func TestLastSession(t *testing.T) {
	e := newEnv(t)
	a1 := firstID(e.must("first session, step one", "add", "--session", "sess-0", "--at", "2026-11-18T10:00:00+01:00"))
	a2 := firstID(e.must("first session, step two", "add", "--session", "sess-0", "--at", "2026-11-18T11:00:00+01:00"))
	e.must("an older session", "add", "--session", "sess-old", "--at", "2026-11-10")
	e.must("this session", "add") // sess-1, now
	sameSet(t, "--session last", ids(e.must("", "list", "--session", "last")), a1, a2)
	out := e.must("", "sessions")
	if !strings.HasPrefix(out, "sess-1") || !strings.Contains(strings.SplitN(out, "\n", 2)[0], "(this session)") ||
		!strings.Contains(out, "sess-0") || !strings.Contains(out, "  2 entries") {
		t.Errorf("sessions output:\n%s", out)
	}
}

func TestFileLinksAndStats(t *testing.T) {
	e := newEnv(t)
	cause := firstID(e.must("Writers hit SQLITE_BUSY.", "add", "--kind", "bug", "--tag", "db", "--file", "internal/store/store.go"))
	fix := firstID(e.must("Use BEGIN IMMEDIATE.", "add", "--kind", "decision", "--tag", "db", "--link", cause[:10]))
	e.must("Tweaked the README.", "add", "--kind", "log", "--file", "README.md")

	sameSet(t, "--file store.go", ids(e.must("", "list", "--file", "store.go")), cause)
	sameSet(t, "--file internal/store/store.go", ids(e.must("", "list", "--file", "internal/store/store.go")), cause)
	if got := ids(e.must("", "list", "--file", "other.go")); len(got) != 0 {
		t.Errorf("--file other.go matched %v", got)
	}
	if show := e.must("", "show", cause); !strings.Contains(show, "linked from: "+fix) {
		t.Errorf("show should list the entry linking here:\n%s", show)
	}
	if show := e.must("", "show", fix); !strings.Contains(show, "links: "+cause) {
		t.Errorf("a link given as a prefix should be stored as the full ID:\n%s", show)
	}
	if _, errOut, err := e.run("x", "add", "--link", "ZZZZ"); err == nil || !strings.Contains(errOut, "no entry") {
		t.Errorf("a link to nothing should be refused, stderr %q", errOut)
	}
	out := e.must("", "stats", "--tag", "db")
	for _, want := range []string{"2 entries", "by kind: ", "bug 1", "decision 1", "by tag: db 2"} {
		if !strings.Contains(out, want) {
			t.Errorf("stats --tag db missing %q:\n%s", want, out)
		}
	}
}

// "What's still open from last week?" — opt-in open state, closed with a
// note that stays searchable.
func TestOpenAndClose(t *testing.T) {
	e := newEnv(t)
	bug := firstID(e.must("Export times out for big tenants.", "add", "--kind", "bug", "--open", "--at", "2026-11-10"))
	todo := firstID(e.must("Write the runbook for restores.", "add", "--kind", "todo", "--at", "2026-11-11"))
	e.must("Fixed a typo.", "add", "--kind", "bug", "--at", "2026-11-12") // not tracked
	e.must("Old open idea.", "add", "--kind", "idea", "--open", "--at", "2026-10-01")

	sameSet(t, "open last week", ids(e.must("", "list", "--status", "open", "--on", "last week")), bug, todo)

	out := e.must("", "close", bug[:12], "--note", "Streamed the CSV instead of building it in memory.")
	if !strings.Contains(out, bug+"  closed") {
		t.Fatalf("close output: %q", out)
	}
	sameSet(t, "open last week after close", ids(e.must("", "list", "--status", "open", "--on", "last week")), todo)
	sameSet(t, "closed", ids(e.must("", "list", "--status", "closed")), bug)

	show := e.must("", "show", bug)
	note := ""
	for _, l := range strings.Split(show, "\n") {
		if strings.HasPrefix(l, "- status: closed") {
			f := strings.Fields(l)
			note = f[len(f)-1]
		}
	}
	if note == "" || !strings.Contains(show, "linked from: "+note) {
		t.Fatalf("show should say closed by the note entry, which links back:\n%s", show)
	}
	if out := e.must("", "search", "--limit", "3", "Streamed the CSV"); !strings.HasPrefix(out, note) {
		t.Errorf("the closing note should be searchable:\n%s", out)
	}

	e.must("", "reopen", bug)
	sameSet(t, "open after reopen", ids(e.must("", "list", "--status", "open", "--on", "last week")), bug, todo)
	if _, errOut, err := e.run("", "list", "--status", "done"); err == nil || !strings.Contains(errOut, "open or closed") {
		t.Errorf("unknown status should be refused, stderr %q", errOut)
	}
}

// A journal written by the first release (schema 1, no status or embedder
// columns) must keep working, with its entries intact.
func TestUpgradeFromSchema1(t *testing.T) {
	e := newEnv(t)
	id := firstID(e.must("Entry from the first release.", "add"))
	db, err := sql.Open("sqlite", filepath.Join(e.home, "test", "journal.db"))
	if err != nil {
		t.Fatal(err)
	}
	for _, q := range []string{
		`ALTER TABLE entries DROP COLUMN status`, `ALTER TABLE entries DROP COLUMN closed_at`,
		`ALTER TABLE entries DROP COLUMN closed_by`, `ALTER TABLE chunks DROP COLUMN embedder`, `PRAGMA user_version = 1`,
	} {
		if _, err := db.Exec(q); err != nil {
			t.Fatalf("%s: %v", q, err)
		}
	}
	db.Close()

	if out := e.must("", "list"); !strings.HasPrefix(out, id) {
		t.Fatalf("old entry missing after upgrade:\n%s", out)
	}
	e.must("", "close", id)
	sameSet(t, "closed after upgrade", ids(e.must("", "list", "--status", "closed")), id)
}

// A summary covers exactly the period asked for, cites every entry so the
// reader can open it, is saved and reused, and is redone when an entry is
// added to the period later.
func TestSummarize(t *testing.T) {
	e := newEnv(t)
	var yesterday []string
	for i, at := range []string{"2026-11-18T09:00:00+01:00", "2026-11-18T13:00:00+01:00", "2026-11-18T23:50:00+01:00"} {
		yesterday = append(yesterday, firstID(e.must(fmt.Sprintf("Yesterday's work, part %d.", i), "add", "--at", at)))
	}
	today := firstID(e.must("Today's work.", "add"))

	out := e.must("", "summarize", "--on", "yesterday")
	for _, id := range yesterday {
		if !strings.Contains(out, id[:14]) {
			t.Errorf("summary does not cite %s:\n%s", id[:14], out)
		}
	}
	if strings.Contains(out, today[:14]) || !strings.Contains(out, "Covers 3 entries") || !strings.Contains(out, "Summary: 2026-11-18 → 2026-11-18") {
		t.Errorf("summary should cover exactly yesterday's 3 entries:\n%s", out)
	}
	if strings.Contains(out, "99999") {
		t.Errorf("a citation that matches no entry must be removed:\n%s", out)
	}
	if got := ids(e.must("", "list", "--kind", "summary")); len(got) != 1 {
		t.Fatalf("want 1 saved summary, got %v", got)
	}

	calls := e.ai.summaryCalls.Load()
	e.must("", "summarize", "--on", "yesterday")
	if n := e.ai.summaryCalls.Load() - calls; n != 0 {
		t.Errorf("an unchanged period should reuse the saved summary, made %d calls", n)
	}

	late := firstID(e.must("Logged later, about yesterday.", "add", "--at", "2026-11-18T18:00:00+01:00"))
	out = e.must("", "summarize", "--on", "yesterday")
	if !strings.Contains(out, late[:14]) || !strings.Contains(out, "Covers 4 entries") {
		t.Errorf("summary should be redone with the late entry:\n%s", out)
	}
	if got := ids(e.must("", "list", "--kind", "summary")); len(got) != 1 {
		t.Errorf("the stale summary should be replaced, got %v", got)
	}
	if _, errOut, err := e.run("", "summarize"); err == nil || !strings.Contains(errOut, "which period") {
		t.Errorf("summarize without a period should ask for one, stderr %q", errOut)
	}
}

// A long period is built from day summaries: no entry may be lost, the day
// summaries are saved, and an entry added later redoes only its own day and
// the final combine.
func TestSummarizeLongPeriodByDay(t *testing.T) {
	e := newEnv(t)
	body := strings.Repeat("Investigated the export worker and tuned its batch size. ", 9)
	var all []string
	for i := 0; i < 120; i++ {
		all = append(all, firstID(e.must(body, "add", "--at", fmt.Sprintf("2026-11-%02dT10:%02d:00+01:00", 9+i%7, i%60))))
	}
	out := e.must("", "summarize", "--on", "last week")
	missing := 0
	for _, id := range all {
		if !strings.Contains(out, id[:14]) {
			missing++
		}
	}
	if missing > 0 || !strings.Contains(out, "Covers 120 entries") {
		t.Errorf("%d of 120 entries missing from the summary", missing)
	}
	if n := e.ai.summaryCalls.Load(); n != 8 {
		t.Errorf("want 7 day summaries + 1 combine = 8 calls, made %d", n)
	}
	if got := ids(e.must("", "list", "--kind", "summary", "--limit", "100")); len(got) != 8 {
		t.Errorf("want 7 day summaries and 1 week summary saved, got %d", len(got))
	}

	calls := e.ai.summaryCalls.Load()
	late := firstID(e.must("Found the export leak.", "add", "--at", "2026-11-12T18:00:00+01:00"))
	out = e.must("", "summarize", "--on", "last week")
	if n := e.ai.summaryCalls.Load() - calls; n != 2 {
		t.Errorf("an entry added to one day should redo that day and the combine (2 calls), made %d", n)
	}
	if !strings.Contains(out, late[:14]) || !strings.Contains(out, "Covers 121 entries") {
		t.Errorf("the late entry is missing from the redone summary")
	}
}
