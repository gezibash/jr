package main

import (
	"strings"
	"testing"
	"time"
)

// Expected dates are worked out by hand from the calendar: 2026-11-19 is a
// Thursday, weeks start on Monday, and Berlin is UTC+1 in November.
func TestParseWhen(t *testing.T) {
	berlin, err := time.LoadLocation("Europe/Berlin")
	if err != nil {
		t.Fatal(err)
	}
	ref := time.Date(2026, 11, 19, 15, 30, 0, 0, berlin) // Thursday afternoon
	d := func(y int, m time.Month, day int) time.Time { return time.Date(y, m, day, 0, 0, 0, 0, berlin) }
	cases := []struct {
		in         string
		start, end time.Time
	}{
		{"today", d(2026, 11, 19), d(2026, 11, 20)},
		{"Yesterday", d(2026, 11, 18), d(2026, 11, 19)},
		{"3 days ago", d(2026, 11, 16), d(2026, 11, 17)},
		{"a couple of days ago", d(2026, 11, 17), d(2026, 11, 18)},
		{"monday", d(2026, 11, 16), d(2026, 11, 17)},
		{"thursday", d(2026, 11, 19), d(2026, 11, 20)}, // today counts
		{"friday", d(2026, 11, 13), d(2026, 11, 14)},
		{"last monday", d(2026, 11, 9), d(2026, 11, 10)},
		{"this week", d(2026, 11, 16), d(2026, 11, 23)},
		{"last week", d(2026, 11, 9), d(2026, 11, 16)},
		{"2 weeks ago", d(2026, 11, 2), d(2026, 11, 9)},
		{"this month", d(2026, 11, 1), d(2026, 12, 1)},
		{"last month", d(2026, 10, 1), d(2026, 11, 1)},
		{"october", d(2026, 10, 1), d(2026, 11, 1)},
		{"dec", d(2025, 12, 1), d(2026, 1, 1)}, // not in the future
		{"march 2025", d(2025, 3, 1), d(2025, 4, 1)},
		{"2026-11-18", d(2026, 11, 18), d(2026, 11, 19)},
		{"2026-10", d(2026, 10, 1), d(2026, 11, 1)},
	}
	for _, c := range cases {
		sp, err := parseWhen(c.in, ref)
		if err != nil {
			t.Errorf("%q: %v", c.in, err)
			continue
		}
		if !sp.start.Equal(c.start) || !sp.end.Equal(c.end) {
			t.Errorf("%q = %s → %s, want %s → %s", c.in, sp.start, sp.end, c.start, c.end)
		}
	}
	// Durations are points in time.
	for in, want := range map[string]time.Time{"36h": ref.Add(-36 * time.Hour), "7d": ref.AddDate(0, 0, -7), "2 hours ago": ref.Add(-2 * time.Hour)} {
		sp, err := parseWhen(in, ref)
		if err != nil || !sp.start.Equal(want) || !sp.end.Equal(want) {
			t.Errorf("%q = %v %v, want point %s", in, sp, err, want)
		}
	}
	if _, err := parseWhen("sometime soon", ref); err == nil || !strings.Contains(err.Error(), "yesterday") {
		t.Errorf("unknown phrase should list the accepted forms, got %v", err)
	}
}
