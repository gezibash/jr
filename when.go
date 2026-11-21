package main

import (
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
	"time"
)

// now is the reference time for relative dates. JRNL_NOW (RFC 3339) pins it,
// so tests and replays don't depend on the wall clock.
func now() time.Time {
	if s := os.Getenv("JRNL_NOW"); s != "" {
		if t, err := time.Parse(time.RFC3339, s); err == nil {
			return t.In(time.Local)
		}
	}
	return time.Now()
}

// span is a stretch of local time, [start, end). A point in time has
// start == end.
type span struct{ start, end time.Time }

const whenHelp = "use today, yesterday, 3 days ago, monday, last monday, this week, last week, " +
	"this month, last month, october, october 2025, 2026-11-18, 2026-11, 36h, 7d, 2w or an RFC 3339 time"

var (
	reDur    = regexp.MustCompile(`^(\d+)\s*(min|m|h|d|w)$`)
	reAgo    = regexp.MustCompile(`^(\d+|a|an|one|two|three|four|five|six|seven|a couple of|couple of|a few|few) (hour|hours|day|days|week|weeks|month|months) ago$`)
	words    = map[string]int{"a": 1, "an": 1, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "a couple of": 2, "couple of": 2, "a few": 3, "few": 3}
	weekdays = map[string]time.Weekday{"monday": time.Monday, "tuesday": time.Tuesday, "wednesday": time.Wednesday,
		"thursday": time.Thursday, "friday": time.Friday, "saturday": time.Saturday, "sunday": time.Sunday}
	months = map[string]time.Month{}
)

func init() {
	for m := time.January; m <= time.December; m++ {
		name := strings.ToLower(m.String())
		months[name], months[name[:3]] = m, m
	}
}

func dayOf(t time.Time) time.Time {
	y, m, d := t.Date()
	return time.Date(y, m, d, 0, 0, 0, 0, t.Location())
}

func days(start time.Time, n int) span { return span{start, start.AddDate(0, 0, n)} }

// weekStart is the Monday of t's week.
func weekStart(t time.Time) time.Time {
	d := dayOf(t)
	return d.AddDate(0, 0, -((int(d.Weekday()) + 6) % 7))
}

// parseWhen reads a date the way people and agents say it, in local time.
// Days, weeks and months are spans; durations and exact times are points.
func parseWhen(s string, ref time.Time) (span, error) {
	s = strings.ToLower(strings.TrimSpace(s))
	today := dayOf(ref)
	switch s {
	case "now":
		return span{ref, ref}, nil
	case "today":
		return days(today, 1), nil
	case "yesterday":
		return days(today.AddDate(0, 0, -1), 1), nil
	case "this week":
		return days(weekStart(ref), 7), nil
	case "last week":
		return days(weekStart(ref).AddDate(0, 0, -7), 7), nil
	case "this month":
		m := time.Date(ref.Year(), ref.Month(), 1, 0, 0, 0, 0, ref.Location())
		return span{m, m.AddDate(0, 1, 0)}, nil
	case "last month":
		m := time.Date(ref.Year(), ref.Month(), 1, 0, 0, 0, 0, ref.Location()).AddDate(0, -1, 0)
		return span{m, m.AddDate(0, 1, 0)}, nil
	}
	if m := reDur.FindStringSubmatch(s); m != nil {
		n, _ := strconv.Atoi(m[1])
		unit := map[string]time.Duration{"min": time.Minute, "m": time.Minute, "h": time.Hour, "d": 24 * time.Hour, "w": 7 * 24 * time.Hour}[m[2]]
		t := ref.Add(-time.Duration(n) * unit)
		return span{t, t}, nil
	}
	if m := reAgo.FindStringSubmatch(s); m != nil {
		n, ok := words[m[1]]
		if !ok {
			n, _ = strconv.Atoi(m[1])
		}
		switch strings.TrimSuffix(m[2], "s") {
		case "hour":
			t := ref.Add(-time.Duration(n) * time.Hour)
			return span{t, t}, nil
		case "day": // that calendar day
			return days(today.AddDate(0, 0, -n), 1), nil
		case "week": // that calendar week
			return days(weekStart(ref).AddDate(0, 0, -7*n), 7), nil
		case "month":
			mo := time.Date(ref.Year(), ref.Month(), 1, 0, 0, 0, 0, ref.Location()).AddDate(0, -n, 0)
			return span{mo, mo.AddDate(0, 1, 0)}, nil
		}
	}
	// monday = the most recent Monday, today included; last monday = the
	// Monday of last week.
	name, last := strings.CutPrefix(s, "last ")
	name = strings.TrimPrefix(name, "this ")
	if wd, ok := weekdays[name]; ok {
		if last {
			return days(weekStart(ref).AddDate(0, 0, -7+(int(wd)+6)%7), 1), nil
		}
		back := (int(today.Weekday()) - int(wd) + 7) % 7
		return days(today.AddDate(0, 0, -back), 1), nil
	}
	// october, oct, october 2025: the most recent such month not in the future.
	f := strings.Fields(s)
	if len(f) >= 1 && len(f) <= 2 {
		if mo, ok := months[f[0]]; ok {
			y := ref.Year()
			if len(f) == 2 {
				var err error
				if y, err = strconv.Atoi(f[1]); err != nil {
					return span{}, fmt.Errorf("can't read %q: %s", s, whenHelp)
				}
			} else if mo > ref.Month() {
				y--
			}
			start := time.Date(y, mo, 1, 0, 0, 0, 0, ref.Location())
			return span{start, start.AddDate(0, 1, 0)}, nil
		}
	}
	if t, err := time.ParseInLocation("2006-01-02", s, ref.Location()); err == nil {
		return days(t, 1), nil
	}
	if t, err := time.ParseInLocation("2006-01", s, ref.Location()); err == nil {
		return span{t, t.AddDate(0, 1, 0)}, nil
	}
	if t, err := time.Parse(time.RFC3339, strings.ToUpper(s)); err == nil {
		return span{t, t}, nil
	}
	return span{}, fmt.Errorf("can't read %q: %s", s, whenHelp)
}

// ago says how long before now ts was, in the largest whole unit: "just
// now", "5m ago", "3h ago", "4d ago", "3w ago", "5mo ago", "2y ago". Agents
// read it without doing date arithmetic against today.
func ago(ts string) string {
	t, err := time.Parse(time.RFC3339, ts)
	if err != nil {
		return ts
	}
	d := now().Sub(t)
	suffix := " ago"
	prefix := ""
	if d < 0 {
		d, prefix, suffix = -d, "in ", ""
	}
	days := int(d.Hours() / 24)
	var n string
	switch {
	case d < time.Minute:
		return "just now"
	case d < time.Hour:
		n = fmt.Sprintf("%dm", int(d.Minutes()))
	case d < 48*time.Hour:
		n = fmt.Sprintf("%dh", int(d.Hours()))
	case days < 14:
		n = fmt.Sprintf("%dd", days)
	case days < 60:
		n = fmt.Sprintf("%dw", days/7)
	case days < 730:
		n = fmt.Sprintf("%dmo", days/30)
	default:
		n = fmt.Sprintf("%dy", days/365)
	}
	return prefix + n + suffix
}
