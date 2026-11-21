package main

import (
	"errors"
	"fmt"
	"os"
	"os/exec"
	"sync"
)

type report struct {
	described, embedded, failed, waiting int
	lastErr                              error
}

// awaitingSetup reports whether err only means an optional part of jrnl has
// never been set up on this machine (for Ollama: not installed, and never
// used by this journal). That work stays pending for `jrnl reindex`
// without a warning, so a fresh install runs quietly. A part that was set up
// and then breaks still warns.
func awaitingSetup(s *Store, err error) bool {
	var ns notSetUp
	var down embedDown
	switch {
	case errors.As(err, &ns):
		return true
	case errors.As(err, &down):
		// Chosen with JRNL_EMBED_URL, installed but not running, or used here
		// before: worth a warning.
		_, installed := exec.LookPath("ollama")
		return !down.set && installed != nil && !s.HasTextVectors()
	}
	return false
}

// process describes the entry's attachments, then embeds everything still
// pending. entryID "" means every entry (reindex). It runs after the entry is
// saved, so a failure here never loses an entry, and all network calls happen
// outside write transactions.
// quiet holds back the warnings for parts that were never set up (add,
// close); reindex passes false, because it is how the user asks for them.
func process(s *Store, ai *AI, entryID string, quiet bool) (report, error) {
	var r report
	hint := false
	toDescribe, err := s.ToDescribe(entryID)
	if err != nil {
		return r, err
	}
	type result struct {
		a    Asset
		desc string
		err  error
	}
	results := make(chan result)
	sem := make(chan struct{}, 4)
	var wg sync.WaitGroup
	for _, a := range toDescribe {
		wg.Add(1)
		go func(a Asset) {
			defer wg.Done()
			sem <- struct{}{}
			defer func() { <-sem }()
			d, err := ai.Describe(a)
			results <- result{a, d, err}
		}(a)
	}
	go func() { wg.Wait(); close(results) }()
	for res := range results {
		switch {
		case res.err == nil:
			err = s.SetDescription(res.a.Hash, res.desc)
			r.described++
		case isPermanent(res.err):
			err = s.SetDescribeError(res.a.Hash, res.err.Error())
			r.failed++
			fmt.Fprintf(os.Stderr, "jrnl: %s can't be described: %v\n", res.a.Name, res.err)
		case quiet && awaitingSetup(s, res.err):
			r.waiting++
			r.lastErr = res.err
			hint = true
		default:
			r.waiting++
			r.lastErr = res.err
			fmt.Fprintf(os.Stderr, "jrnl: %s not described yet (run `jrnl reindex` later): %v\n", res.a.Name, res.err)
		}
		if err != nil {
			return r, err
		}
	}

	if hint {
		fmt.Fprintln(os.Stderr, "jrnl: attachment saved; captions and transcripts need setup (see `jrnl doctor`)")
	}

	ps, err := s.Pending(entryID, ai.embedID, ai.mediaID)
	if err != nil {
		return r, err
	}
	for start := 0; start < len(ps); start += 16 {
		batch := ps[start:min(start+16, len(ps))]
		ins := make([]Input, len(batch))
		for i, p := range batch {
			ins[i] = p.in
		}
		vecs, errs := ai.EmbedEach(ins)
		for i, p := range batch {
			switch {
			case errs[i] == nil:
				emb := ai.embedID
				if p.in.File != "" {
					emb = ai.mediaID
				}
				err = s.SetVec(p.chunkID, vecs[i], emb)
				r.embedded++
			case isPermanent(errs[i]):
				// Recorded so the next reindex skips it instead of retrying forever.
				err = s.SetError(p.chunkID, errs[i].Error())
				r.failed++
				fmt.Fprintf(os.Stderr, "jrnl: %s can't be embedded: %v\n", p.label, errs[i])
			case quiet && awaitingSetup(s, errs[i]):
				r.waiting++
				r.lastErr = errs[i]
			default:
				r.waiting++
				r.lastErr = errs[i]
				fmt.Fprintf(os.Stderr, "jrnl: %s saved without embedding (run `jrnl reindex` later): %v\n", p.label, errs[i])
			}
			if err != nil {
				return r, err
			}
		}
	}
	return r, nil
}
