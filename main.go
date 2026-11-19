// jr is a per-project journal for coding agents. Entries live in SQLite;
// markdown is only the input and output format.
package main

import (
	"flag"
	"fmt"
	"io"
	"os"
	"os/exec"
	"path/filepath"
	"slices"
	"sort"
	"strings"
	"time"
)

const usage = `jr — per-project journal

  jr add [flags] [body…]       add an entry (body from args or stdin); prints its ID
      --title T                default: first line of body
      --kind K                 decision | bug | idea | todo | log | milestone | note (default note)
      --tag T                  repeatable
      --link ID                related entry (ID prefix is enough), repeatable
      --file PATH              file the entry is about, repeatable
      --attach PATH[::CAPTION] image, video or audio, repeatable
      --at WHEN                when it happened (default now), e.g. yesterday
      --open                   track it as open until closed (todo entries start open)
  jr close ID… [--note TEXT]   mark entries closed; the note is saved as an entry linked to them
  jr reopen ID…                mark entries open again
  jr show ID…                  print entries as markdown (ID prefix is enough)
  jr list [filters]            newest first; --oldest for oldest first
  jr search [filters] QUERY    keyword + semantic search over entries and attachment descriptions
      --newest | --oldest      the latest / the first matches, instead of the best
  jr search --like FILE        find attachments that look or sound like FILE
  jr sessions [filters]        recent sessions: when, how many entries, last title
  jr stats [filters]           counts by kind, tag and author
  jr summarize [filters]       summary of a period, citing entry IDs; saved and reused
      --refresh                write it again even if nothing changed
  jr reindex                   describe and embed whatever failed earlier
  jr backup                    consistent copy of the database
  jr where                     print this project's journal directory

filters: --since WHEN  --until WHEN  --on WHEN
         --kind K  --tag T (repeatable, all must match)  --file NAME
         --status open|closed  --session ID|last  --author A  --limit N
WHEN, in local time: today, yesterday, 3 days ago, monday, last monday,
         this week, last week, this month, last month, october, 2026-11-18,
         2026-11, 36h, 7d, 2w. --since takes its start, --until its end,
         --on the whole day, week or month.
--session last: the most recent session before this one.
every command: --project P (default: from git remote, else directory name)

Text is embedded by Ollama on this machine (ollama pull embeddinggemma-2:270m).
Captions, reranking and --like use Workers AI: run jr through op-agent with
CLOUDFLARE_API_TOKEN and CLOUDFLARE_ACCOUNT_ID set.
`

func main() {
	if len(os.Args) < 2 {
		fmt.Fprint(os.Stderr, usage)
		os.Exit(2)
	}
	cmds := map[string]func([]string) error{
		"add": cmdAdd, "show": cmdShow, "list": cmdList, "ls": cmdList, "search": cmdSearch,
		"reindex": cmdReindex, "backup": cmdBackup, "where": cmdWhere,
		"sessions": cmdSessions, "stats": cmdStats, "close": cmdClose, "reopen": cmdReopen,
		"summarize": cmdSummarize, "summarise": cmdSummarize,
	}
	name := os.Args[1]
	if name == "help" || name == "-h" || name == "--help" {
		fmt.Print(usage)
		return
	}
	cmd, ok := cmds[name]
	if !ok {
		fmt.Fprintf(os.Stderr, "jr: unknown command %q\n\n%s", name, usage)
		os.Exit(2)
	}
	if err := cmd(os.Args[2:]); err != nil {
		fmt.Fprintln(os.Stderr, "jr:", err)
		os.Exit(1)
	}
}

// Raw media vectors serve only search --like. Embedding a 941 MB video
// peaked at 7.6 GB of memory before Workers AI rejected it with 413, while
// 52 MB went through.
const maxRawBytes = 50 << 20

// Search deadlines. Past them, search falls back to keyword-only results
// (embedding) or keyword+vector order (rerank) instead of waiting.
const (
	embedDeadline  = 3 * time.Second
	rerankDeadline = 4 * time.Second
)

type multi []string

func (m *multi) String() string     { return strings.Join(*m, ",") }
func (m *multi) Set(v string) error { *m = append(*m, v); return nil }

// parse lets flags and positional arguments mix in any order.
func parse(fs *flag.FlagSet, args []string) ([]string, error) {
	var pos []string
	for {
		if err := fs.Parse(args); err != nil {
			return nil, err
		}
		if fs.NArg() == 0 {
			return pos, nil
		}
		pos = append(pos, fs.Arg(0))
		args = fs.Args()[1:]
	}
}

func newFlags(name string) (*flag.FlagSet, *string) {
	fs := flag.NewFlagSet(name, flag.ContinueOnError)
	fs.Usage = func() { fmt.Fprint(os.Stderr, usage) }
	return fs, fs.String("project", "", "project name")
}

func filterFlags(fs *flag.FlagSet) (*Filter, *int) {
	f := &Filter{}
	fs.StringVar(&f.Kind, "kind", "", "")
	fs.StringVar(&f.Session, "session", "", "")
	fs.StringVar(&f.Author, "author", "", "")
	fs.StringVar(&f.Since, "since", "", "")
	fs.StringVar(&f.Until, "until", "", "")
	fs.StringVar(&f.On, "on", "", "")
	fs.StringVar(&f.File, "file", "", "")
	fs.StringVar(&f.Status, "status", "", "")
	fs.Var((*multi)(&f.Tags), "tag", "")
	return f, fs.Int("limit", 20, "")
}

// resolveFilter turns --session last into the previous session's ID.
func resolveFilter(s *Store, f *Filter) error {
	if f.Session == "last" {
		id, err := s.LastSession(os.Getenv("CLAUDE_CODE_SESSION_ID"))
		if err != nil {
			return err
		}
		f.Session = id
	}
	return nil
}

func open(project string) (*Store, gitInfo, error) {
	g := detectGit()
	if project == "" {
		project = g.project
	}
	s, err := Open(project)
	g.project = project
	return s, g, err
}

func cmdAdd(args []string) error {
	fs, project := newFlags("add")
	title := fs.String("title", "", "")
	kind := fs.String("kind", "note", "")
	author := fs.String("author", defaultAuthor(), "")
	session := fs.String("session", os.Getenv("CLAUDE_CODE_SESSION_ID"), "")
	var tags, links, files, attach multi
	fs.Var(&tags, "tag", "")
	fs.Var(&links, "link", "")
	fs.Var(&files, "file", "")
	fs.Var(&attach, "attach", "")
	at := fs.String("at", "", "")
	open_ := fs.Bool("open", false, "")
	pos, err := parse(fs, args)
	if err != nil {
		return err
	}

	body := strings.Join(pos, " ")
	if body == "" && !isTerminal(os.Stdin) {
		b, err := io.ReadAll(os.Stdin)
		if err != nil {
			return err
		}
		body = string(b)
	}
	body = strings.TrimSpace(body)
	if body == "" && *title == "" && len(attach) == 0 {
		return fmt.Errorf("empty entry: give a body as arguments or on stdin")
	}
	if *title == "" {
		*title = firstLine(body)
	}

	s, g, err := open(*project)
	if err != nil {
		return err
	}
	defer s.Close()

	ts := now()
	if *at != "" {
		sp, err := parseWhen(*at, ts)
		if err != nil {
			return fmt.Errorf("--at: %w", err)
		}
		ts = sp.start
	}
	for i, l := range links {
		if links[i], err = s.Resolve(l); err != nil {
			return fmt.Errorf("--link: %w", err)
		}
	}
	e := Entry{
		ID: newID(), TS: utc(ts), Project: g.project,
		Session: *session, Author: *author, Kind: *kind, Branch: g.branch, Commit: g.commit,
		Title: *title, Body: body, Tags: tags, Links: links, Files: files,
	}
	if *open_ || *kind == "todo" {
		e.Status = "open"
	}
	for _, a := range attach {
		path, caption, _ := strings.Cut(a, "::")
		asset, err := s.Ingest(path, caption)
		if err != nil {
			return fmt.Errorf("attach %s: %w", path, err)
		}
		e.Assets = append(e.Assets, asset)
	}

	// The entry is saved first; describing and embedding happen after, outside
	// the write lock, and anything that fails stays queued for `jr reindex`.
	chunks := []Chunk{{Modality: "text"}}
	for _, a := range e.Assets {
		if m := a.Modality(); m != "" && a.Bytes > maxRawBytes {
			// Its description is still made and searched; only --like misses it.
			fmt.Fprintf(os.Stderr, "jr: %s is over %d MB, so search --like won't match it\n", a.Name, maxRawBytes>>20)
		} else if m != "" {
			chunks = append(chunks, Chunk{Hash: a.Hash, Modality: m})
		} else {
			fmt.Fprintf(os.Stderr, "jr: %s stored but not searchable by content (%s is not image, video or audio)\n", a.Name, a.Mime)
		}
	}
	if err := s.Insert(e, chunks); err != nil {
		return err
	}
	// Print the ID as soon as the entry is saved, so a later failure can't
	// lead the caller to retry and create a duplicate.
	fmt.Println(e.ID)
	if ai, err := NewAI(); err != nil {
		fmt.Fprintf(os.Stderr, "jr: saved without embedding (run `jr reindex` later): %v\n", err)
	} else if _, err := process(s, ai, e.ID); err != nil {
		return err
	}
	return nil
}

func cmdShow(args []string) error {
	fs, project := newFlags("show")
	ids, err := parse(fs, args)
	if err != nil {
		return err
	}
	if len(ids) == 0 {
		return fmt.Errorf("show: give at least one entry ID")
	}
	s, _, err := open(*project)
	if err != nil {
		return err
	}
	defer s.Close()
	for i, p := range ids {
		id, err := s.Resolve(p)
		if err != nil {
			return err
		}
		e, err := s.Get(id)
		if err != nil {
			return err
		}
		if i > 0 {
			fmt.Print("\n---\n\n")
		}
		printEntry(e)
	}
	return nil
}

func printEntry(e Entry) {
	fmt.Printf("# %s\n\n", e.Title)
	fmt.Printf("- id: %s\n- time: %s\n- kind: %s · author: %s\n", e.ID, localTime(e.TS), e.Kind, e.Author)
	if e.Branch != "" || e.Commit != "" {
		fmt.Printf("- git: %s @ %s\n", e.Branch, e.Commit)
	}
	if e.Session != "" {
		fmt.Printf("- session: %s\n", e.Session)
	}
	if len(e.Tags) > 0 {
		fmt.Printf("- tags: %s\n", strings.Join(e.Tags, ", "))
	}
	if len(e.Links) > 0 {
		fmt.Printf("- links: %s\n", strings.Join(e.Links, ", "))
	}
	switch e.Status {
	case "open":
		fmt.Println("- status: open")
	case "closed":
		line := "- status: closed " + localTime(e.ClosedAt)
		if e.ClosedBy != "" {
			line += " by " + e.ClosedBy
		}
		fmt.Println(line)
	}
	if len(e.LinkedFrom) > 0 {
		fmt.Printf("- linked from: %s\n", strings.Join(e.LinkedFrom, ", "))
	}
	if len(e.Files) > 0 {
		fmt.Printf("- files: %s\n", strings.Join(e.Files, ", "))
	}
	for _, a := range e.Assets {
		line := fmt.Sprintf("- attachment: %s (%s, %s)", a.Path, a.Name, a.Mime)
		if a.Caption != "" {
			line += " — " + a.Caption
		}
		fmt.Println(line)
		if a.Description != "" {
			fmt.Printf("  - description: %s\n", oneLine(a.Description))
		}
	}
	if e.Body != "" {
		fmt.Printf("\n%s\n", e.Body)
	}
}

func summaryLine(e Entry) string {
	line := fmt.Sprintf("%s  %s  %-9s %s", e.ID, localTime(e.TS), e.Kind, e.Title)
	if e.Status != "" {
		line += "  [" + e.Status + "]"
	}
	for _, t := range e.Tags {
		line += " #" + t
	}
	if n := len(e.Assets); n > 0 {
		line += fmt.Sprintf("  [%d attachment(s)]", n)
	}
	return line
}

func cmdList(args []string) error {
	fs, project := newFlags("list")
	f, limit := filterFlags(fs)
	oldest := fs.Bool("oldest", false, "")
	if _, err := parse(fs, args); err != nil {
		return err
	}
	s, _, err := open(*project)
	if err != nil {
		return err
	}
	defer s.Close()
	if err := resolveFilter(s, f); err != nil {
		return err
	}
	es, err := s.List(*f, *limit, *oldest)
	if err != nil {
		return err
	}
	for _, e := range es {
		fmt.Println(summaryLine(e))
	}
	return nil
}

func cmdSessions(args []string) error {
	fs, project := newFlags("sessions")
	f, limit := filterFlags(fs)
	if _, err := parse(fs, args); err != nil {
		return err
	}
	s, _, err := open(*project)
	if err != nil {
		return err
	}
	defer s.Close()
	ss, err := s.Sessions(*f, *limit)
	if err != nil {
		return err
	}
	me := os.Getenv("CLAUDE_CODE_SESSION_ID")
	for _, si := range ss {
		mark := ""
		if si.ID == me {
			mark = "  (this session)"
		}
		fmt.Printf("%s  %s → %s  %3d entries  last: %s%s\n", si.ID, localTime(si.First), localTime(si.Last)[11:], si.Entries, si.LastTitle, mark)
	}
	return nil
}

func cmdStats(args []string) error {
	fs, project := newFlags("stats")
	f, _ := filterFlags(fs)
	if _, err := parse(fs, args); err != nil {
		return err
	}
	s, _, err := open(*project)
	if err != nil {
		return err
	}
	defer s.Close()
	if err := resolveFilter(s, f); err != nil {
		return err
	}
	total, first, last, by, err := s.Stats(*f)
	if err != nil {
		return err
	}
	fmt.Printf("%d entries", total)
	if total > 0 {
		fmt.Printf(", %s → %s", localTime(first), localTime(last))
	}
	fmt.Println()
	for _, name := range []string{"kind", "tag", "author"} {
		var parts []string
		for _, c := range by[name] {
			parts = append(parts, fmt.Sprintf("%s %d", c.Key, c.N))
		}
		if len(parts) > 0 {
			fmt.Printf("by %s: %s\n", name, strings.Join(parts, ", "))
		}
	}
	return nil
}

func cmdSearch(args []string) error {
	fs, project := newFlags("search")
	f, limit := filterFlags(fs)
	like := fs.String("like", "", "")
	newest := fs.Bool("newest", false, "")
	oldest := fs.Bool("oldest", false, "")
	pos, err := parse(fs, args)
	if err != nil {
		return err
	}
	q := strings.TrimSpace(strings.Join(pos, " "))
	if q == "" && *like == "" {
		return fmt.Errorf("search: give a query, or --like FILE")
	}
	order := ""
	switch {
	case *newest && *oldest:
		return fmt.Errorf("search: pick one of --newest and --oldest")
	case *newest:
		order = "newest"
	case *oldest:
		order = "oldest"
	}
	s, _, err := open(*project)
	if err != nil {
		return err
	}
	defer s.Close()
	if err := resolveFilter(s, f); err != nil {
		return err
	}
	var qvec []float32
	// Text queries rank entry text and attachment descriptions. --like ranks
	// raw media of the same kind: vectors compare well within one modality,
	// not across them.
	in, modalities, embedder := Input{Text: queryText(q)}, []string{"text", "caption"}, textEmbedder
	if *like != "" {
		a := Asset{Mime: mediaTypes[strings.ToLower(filepath.Ext(*like))]}
		if a.Modality() == "" {
			return fmt.Errorf("--like needs an image, video or audio file")
		}
		in, modalities, embedder, q = Input{File: *like, Mime: a.Mime}, []string{a.Modality()}, mediaEmbedder, ""
	}
	ai, aiErr := NewAI()
	if aiErr == nil {
		var vecs [][]float32
		// A text query embeds locally in ~40 ms, so it gets a deadline; a
		// --like file goes to Workers AI and a video can take ~10 s.
		client := ai.Interactive(embedDeadline)
		if *like != "" {
			client = ai
		}
		if vecs, aiErr = client.Embed([]Input{in}); aiErr == nil {
			qvec = vecs[0]
		}
	}
	if aiErr != nil {
		if *like != "" {
			return aiErr
		}
		fmt.Fprintf(os.Stderr, "jr: keyword search only: %v\n", aiErr)
	}
	hits, err := s.Search(q, qvec, embedder, modalities, *f, order)
	if err != nil {
		return err
	}
	if qvec != nil && q != "" && len(hits) > 1 {
		k := min(*limit, 20)
		if order != "" {
			k = 0 // every relevant entry; sorted by date below
		}
		hits = rerank(ai.Interactive(rerankDeadline), q, hits, k)
	}
	if order != "" {
		// The reranker's picks are the relevant ones; show them in date
		// order. Without a reranker, all candidates in date order.
		var picked []Hit
		for _, h := range hits {
			if slices.Contains(h.Via, "reranked") {
				picked = append(picked, h)
			}
		}
		if len(picked) > 0 {
			hits = picked
		}
		sort.SliceStable(hits, func(i, j int) bool {
			if order == "oldest" {
				return hits[i].Entry.TS < hits[j].Entry.TS
			}
			return hits[i].Entry.TS > hits[j].Entry.TS
		})
	}
	if len(hits) > *limit {
		hits = hits[:*limit]
	}
	for _, h := range hits {
		fmt.Printf("%s  (%s)\n", summaryLine(h.Entry), strings.Join(h.Via, "+"))
		if h.Snippet != "" {
			fmt.Printf("    %s\n", oneLine(h.Snippet))
		} else if h.Entry.Body != "" {
			fmt.Printf("    %s\n", truncate(oneLine(h.Entry.Body), 160))
		}
		if h.Match != "" {
			fmt.Printf("    matched %s\n", h.Match)
		}
	}
	return nil
}

func cmdReindex(args []string) error {
	fs, project := newFlags("reindex")
	if _, err := parse(fs, args); err != nil {
		return err
	}
	s, _, err := open(*project)
	if err != nil {
		return err
	}
	defer s.Close()
	ai, err := NewAI()
	if err != nil {
		return err
	}
	r, err := process(s, ai, "")
	if err != nil {
		return err
	}
	fmt.Printf("described %d file(s), embedded %d pending chunk(s)", r.described, r.embedded)
	if r.failed > 0 {
		fmt.Printf(", %d can never be processed", r.failed)
	}
	if r.waiting > 0 {
		fmt.Printf(", %d still pending (%v)", r.waiting, r.lastErr)
	}
	fmt.Println()
	return nil
}

func cmdBackup(args []string) error {
	fs, project := newFlags("backup")
	if _, err := parse(fs, args); err != nil {
		return err
	}
	s, _, err := open(*project)
	if err != nil {
		return err
	}
	defer s.Close()
	dst, err := s.Backup()
	if err != nil {
		return err
	}
	fmt.Println(dst)
	return nil
}

func cmdWhere(args []string) error {
	fs, project := newFlags("where")
	if _, err := parse(fs, args); err != nil {
		return err
	}
	p := *project
	if p == "" {
		p = detectGit().project
	}
	fmt.Println(filepath.Join(journalHome(), p))
	return nil
}

type gitInfo struct{ project, branch, commit string }

func git(args ...string) string {
	out, err := exec.Command("git", args...).Output()
	if err != nil {
		return ""
	}
	return strings.TrimSpace(string(out))
}

// detectGit names the project after its origin remote, so every worktree and
// clone of one repository shares a journal.
func detectGit() gitInfo {
	g := gitInfo{branch: git("branch", "--show-current"), commit: git("rev-parse", "--short", "HEAD")}
	if r := git("remote", "get-url", "origin"); r != "" {
		g.project = slugFromRemote(r)
	} else if top := git("rev-parse", "--show-toplevel"); top != "" {
		g.project = slug(filepath.Base(top))
	} else {
		wd, _ := os.Getwd()
		g.project = slug(filepath.Base(wd))
	}
	return g
}

func slugFromRemote(r string) string {
	r = strings.TrimSuffix(strings.TrimSuffix(r, "/"), ".git")
	parts := strings.FieldsFunc(r, func(c rune) bool { return c == '/' || c == ':' })
	if len(parts) >= 2 {
		return slug(parts[len(parts)-2] + "-" + parts[len(parts)-1])
	}
	return slug(r)
}

func slug(s string) string {
	var b strings.Builder
	dash := false
	for _, c := range strings.ToLower(s) {
		if (c >= 'a' && c <= 'z') || (c >= '0' && c <= '9') {
			b.WriteRune(c)
			dash = false
		} else if !dash && b.Len() > 0 {
			b.WriteByte('-')
			dash = true
		}
	}
	return strings.TrimSuffix(b.String(), "-")
}

func defaultAuthor() string {
	if os.Getenv("CLAUDECODE") != "" {
		return "claude"
	}
	return "me"
}

func firstLine(body string) string {
	line, _, _ := strings.Cut(body, "\n")
	line = strings.TrimSpace(strings.TrimLeft(line, "# "))
	if line == "" {
		return "untitled"
	}
	return truncate(line, 80)
}

func oneLine(s string) string { return strings.Join(strings.Fields(s), " ") }

func localTime(ts string) string {
	t, err := time.Parse(time.RFC3339, ts)
	if err != nil {
		return ts
	}
	return t.Local().Format("2006-01-02 15:04")
}

func isTerminal(f *os.File) bool {
	st, err := f.Stat()
	return err == nil && st.Mode()&os.ModeCharDevice != 0
}

// rerank reorders hits with Gemma; the ones it picks come first, marked
// "reranked", then the rest in fused order. If the call fails, the fused
// order stands.
// k is how many best entries to ask for; 0 asks for every relevant one.
func rerank(ai *AI, q string, hits []Hit, k int) []Hit {
	// Entries with identical text are shown to the reranker once.
	var items []string
	var itemHit []int
	seen := map[string]bool{}
	for i, h := range hits {
		text := truncate(oneLine(h.Entry.Body), 400)
		for _, a := range h.Entry.Assets {
			if a.Description != "" {
				text += " | attachment: " + truncate(oneLine(a.Description), 200)
			}
		}
		if !seen[text] {
			seen[text] = true
			items = append(items, text)
			itemHit = append(itemHit, i)
		}
	}
	order, err := ai.Rerank(q, items, k)
	if err != nil {
		fmt.Fprintf(os.Stderr, "jr: not reranked: %v\n", err)
		return hits
	}
	if len(order) == 0 {
		fmt.Fprintln(os.Stderr, "jr: no entry clearly answers this; showing the closest matches")
		return hits
	}
	out := make([]Hit, 0, len(hits))
	picked := map[int]bool{}
	for _, it := range order {
		i := itemHit[it]
		h := hits[i]
		h.Via = append(h.Via, "reranked")
		out = append(out, h)
		picked[i] = true
	}
	for i, h := range hits {
		if !picked[i] {
			out = append(out, h)
		}
	}
	return out
}

func embed(ins []Input) ([][]float32, error) {
	ai, err := NewAI()
	if err != nil {
		return nil, err
	}
	return ai.Embed(ins)
}

func cmdClose(args []string) error {
	fs, project := newFlags("close")
	note := fs.String("note", "", "")
	pos, err := parse(fs, args)
	if err != nil {
		return err
	}
	return setStatus(*project, pos, "closed", *note)
}

func cmdReopen(args []string) error {
	fs, project := newFlags("reopen")
	pos, err := parse(fs, args)
	if err != nil {
		return err
	}
	return setStatus(*project, pos, "open", "")
}

// setStatus closes or reopens entries. A closing note becomes its own entry
// linked to them, so how something was resolved stays searchable.
func setStatus(project string, prefixes []string, status, note string) error {
	if len(prefixes) == 0 {
		return fmt.Errorf("give at least one entry ID")
	}
	s, g, err := open(project)
	if err != nil {
		return err
	}
	defer s.Close()
	var ids, titles []string
	for _, p := range prefixes {
		id, err := s.Resolve(p)
		if err != nil {
			return err
		}
		e, err := s.Get(id)
		if err != nil {
			return err
		}
		ids, titles = append(ids, id), append(titles, e.Title)
	}
	closedBy := ""
	if note != "" {
		ne := Entry{
			ID: newID(), TS: utc(now()), Project: g.project, Session: os.Getenv("CLAUDE_CODE_SESSION_ID"),
			Author: defaultAuthor(), Kind: "log", Branch: g.branch, Commit: g.commit,
			Title: "Closed: " + truncate(strings.Join(titles, "; "), 70), Body: note, Links: ids,
		}
		if err := s.Insert(ne, []Chunk{{Modality: "text"}}); err != nil {
			return err
		}
		if ai, err := NewAI(); err == nil {
			if _, err := process(s, ai, ne.ID); err != nil {
				return err
			}
		}
		closedBy = ne.ID
	}
	if err := s.SetStatus(ids, status, closedBy); err != nil {
		return err
	}
	for i, id := range ids {
		fmt.Printf("%s  %s  %s\n", id, status, titles[i])
	}
	return nil
}
