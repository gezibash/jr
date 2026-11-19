package main

import "testing"

// Worked by hand: 6 entries; [3, 1, 3, 9, 2, 4, 5, 6] keeps the first five
// distinct valid numbers (3, 1, 2, 4, 5), drops 9 (no such entry), and
// leaves non-citation brackets such as [x] and [see docs] alone.
func TestCiteIDs(t *testing.T) {
	var es []Entry
	for _, id := range []string{"AAAAAAAAAAAAAA01", "BBBBBBBBBBBBBB02", "CCCCCCCCCCCCCC03", "DDDDDDDDDDDDDD04", "EEEEEEEEEEEEEE05", "FFFFFFFFFFFFFF06"} {
		es = append(es, Entry{ID: id})
	}
	got, dropped := citeIDs("- a [3, 1, 3, 9, 2, 4, 5, 6]\n- b [7]\n- [x] done [see docs]", es)
	want := "- a [CCCCCCCCCCCCCC, AAAAAAAAAAAAAA, BBBBBBBBBBBBBB, DDDDDDDDDDDDDD, EEEEEEEEEEEEEE]\n- b \n- [x] done [see docs]"
	if got != want || dropped != 2 {
		t.Errorf("got %q (dropped %d)\nwant %q (dropped 2)", got, dropped, want)
	}
}
