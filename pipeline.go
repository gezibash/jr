package main

import (
	"fmt"
	"os"
	"sync"
)

type report struct {
	described, embedded, failed, waiting int
	lastErr                              error
}

// process describes the entry's attachments, then embeds everything still
// pending. entryID "" means every entry (reindex). It runs after the entry is
// saved, so a failure here never loses an entry, and all network calls happen
// outside write transactions.
func process(s *Store, ai *AI, entryID string) (report, error) {
	var r report
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
			fmt.Fprintf(os.Stderr, "jr: %s can't be described: %v\n", res.a.Name, res.err)
		default:
			r.waiting++
			r.lastErr = res.err
			fmt.Fprintf(os.Stderr, "jr: %s not described yet (run `jr reindex` later): %v\n", res.a.Name, res.err)
		}
		if err != nil {
			return r, err
		}
	}

	ps, err := s.Pending(entryID)
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
				emb := textEmbedder
				if p.in.File != "" {
					emb = mediaEmbedder
				}
				err = s.SetVec(p.chunkID, vecs[i], emb)
				r.embedded++
			case isPermanent(errs[i]):
				// Recorded so the next reindex skips it instead of retrying forever.
				err = s.SetError(p.chunkID, errs[i].Error())
				r.failed++
				fmt.Fprintf(os.Stderr, "jr: %s can't be embedded: %v\n", p.label, errs[i])
			default:
				r.waiting++
				r.lastErr = errs[i]
				fmt.Fprintf(os.Stderr, "jr: %s saved without embedding (run `jr reindex` later): %v\n", p.label, errs[i])
			}
			if err != nil {
				return r, err
			}
		}
	}
	return r, nil
}
