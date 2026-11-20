package main

import (
	"errors"
	"fmt"
	"os"
	"path/filepath"
	"slices"
	"strings"

	"github.com/BurntSushi/toml"
)

// config.toml names providers (where a service is and how to talk to it) and
// gives each role (embed, media_embed, rerank, chat) a provider and a model:
//
//	[providers.vercel]
//	type = "openai"
//	url  = "https://ai-gateway.vercel.sh/v1"
//	key  = "env:AI_GATEWAY_API_KEY"
//
//	[chat]
//	provider = "vercel"
//	model    = "google/gemma-4-26b-a4b-it"
//
// The file holds no secrets: a key names the variable that has it. JR_*
// variables win over the file (see resolve).
type config struct {
	Providers  map[string]providerConf `toml:"providers"`
	Embed      roleConf                `toml:"embed"`
	MediaEmbed roleConf                `toml:"media_embed"`
	Rerank     roleConf                `toml:"rerank"`
	Chat       roleConf                `toml:"chat"`
}

type providerConf struct {
	Type    string `toml:"type"`    // openai (default) or cloudflare
	URL     string `toml:"url"`     // openai: the API base, …/v1; cloudflare: replaces …/accounts/<id>/ai
	Key     string `toml:"key"`     // env:NAME
	Account string `toml:"account"` // cloudflare: the account ID, or env:NAME
}

type roleConf struct {
	Provider string         `toml:"provider"`
	Model    string         `toml:"model"`
	Extra    map[string]any `toml:"extra"` // chat: JSON fields added to each request
}

// What each provider type can do. Cloudflare's OpenAI-style embeddings
// endpoint rejected text (400), so it serves chat and media vectors only.
var roleTypes = map[string][]string{
	"embed":       {"openai"},
	"media_embed": {"cloudflare"},
	"rerank":      {"openai"},
	"chat":        {"openai", "cloudflare"},
}

// configFile is JR_CONFIG, or ~/.config/jr/config.toml.
func configFile() string {
	if p := os.Getenv("JR_CONFIG"); p != "" {
		return p
	}
	home, _ := os.UserHomeDir()
	return filepath.Join(home, ".config", "jr", "config.toml")
}

// loadConfig reads the file; a missing file is an empty config. Anything
// it does not know is an error, so a typo can't quietly turn a part off.
func loadConfig(path string) (config, error) {
	var c config
	md, err := toml.DecodeFile(path, &c)
	switch {
	case errors.Is(err, os.ErrNotExist):
		return config{}, nil
	case err != nil:
		return c, fmt.Errorf("%s: %w", path, err)
	}
	for _, k := range md.Undecoded() {
		// extra is free-form JSON for the API; its nested keys are its own.
		if len(k) > 2 && k[1] == "extra" {
			continue
		}
		return c, fmt.Errorf("%s: unknown setting %s", path, k)
	}
	return c, nil
}

// endpoint is a role, resolved: how to reach its model.
type endpoint struct {
	kind  string // provider type: openai or cloudflare
	url   string // OpenAI-compatible base, …/v1
	run   string // cloudflare: Workers AI's own API, …/ai/run
	key   string
	model string
	extra map[string]any
	from  string // where it was set, for jr doctor
}

// resolve gives role r its endpoint. JR_<env>_URL, when set, defines the
// whole role from the environment (an OpenAI-compatible API), so a key from
// the file is never sent to another URL. JR_<env>_MODEL and JR_<env>_KEY
// alone replace just that field. env "" means the role has no variables.
func (c config) resolve(role string, r roleConf, env string) (endpoint, error) {
	get := func(field string) string {
		if env == "" {
			return ""
		}
		return os.Getenv("JR_" + env + "_" + field)
	}
	var e endpoint
	if u := get("URL"); u != "" {
		e = endpoint{kind: "openai", url: strings.TrimSuffix(u, "/"), key: get("KEY"), model: get("MODEL"),
			from: "JR_" + env + "_URL"}
		if e.model == "" {
			return e, fmt.Errorf("JR_%s_URL is set, so set JR_%s_MODEL too", env, env)
		}
		return e, nil
	}
	if r.Provider == "" {
		if r.Model != "" || r.Extra != nil {
			return e, fmt.Errorf("config [%s] needs a provider", role)
		}
		e.model, e.key = get("MODEL"), get("KEY")
		return e, nil
	}
	p, ok := c.Providers[r.Provider]
	if !ok {
		return e, fmt.Errorf("config [%s]: no provider %q under [providers]", role, r.Provider)
	}
	kind := p.Type
	if kind == "" {
		kind = "openai"
	}
	if !slices.Contains([]string{"openai", "cloudflare"}, kind) {
		return e, fmt.Errorf("config [providers.%s]: unknown type %q (openai or cloudflare)", r.Provider, kind)
	}
	if !slices.Contains(roleTypes[role], kind) {
		return e, fmt.Errorf("config [%s]: a %s provider can't do %s", role, kind, role)
	}
	key, err := keyRef(p.Key)
	if err != nil {
		return e, fmt.Errorf("config [providers.%s]: %w", r.Provider, err)
	}
	e = endpoint{kind: kind, key: key, model: r.Model, extra: r.Extra, from: "config.toml"}
	switch kind {
	case "openai":
		if p.URL == "" {
			return e, fmt.Errorf("config [providers.%s]: set url", r.Provider)
		}
		e.url = strings.TrimSuffix(p.URL, "/")
	case "cloudflare":
		root := strings.TrimSuffix(p.URL, "/")
		if root == "" {
			acct := p.Account
			if name, ok := strings.CutPrefix(acct, "env:"); ok {
				acct = os.Getenv(name)
			}
			if acct == "" {
				return e, fmt.Errorf("config [providers.%s]: set account (the ID, or env:NAME)", r.Provider)
			}
			root = "https://api.cloudflare.com/client/v4/accounts/" + acct + "/ai"
		}
		e.url, e.run = root+"/v1", root+"/run"
	}
	if e.model == "" {
		return e, fmt.Errorf("config [%s]: set model", role)
	}
	if m := get("MODEL"); m != "" {
		e.model = m
	}
	if k := get("KEY"); k != "" {
		e.key = k
	}
	return e, nil
}

// keyRef reads a key from the variable it names. Keys never sit in the file.
func keyRef(ref string) (string, error) {
	if ref == "" {
		return "", nil
	}
	name, ok := strings.CutPrefix(ref, "env:")
	if !ok || name == "" {
		return "", errors.New(`key must name a variable, like key = "env:MY_API_KEY"; config.toml holds no keys`)
	}
	return os.Getenv(name), nil
}
