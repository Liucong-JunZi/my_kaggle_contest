package main

import (
	"bufio"
	"encoding/csv"
	"encoding/json"
	"flag"
	"fmt"
	"io"
	"log"
	"math"
	"net/http"
	"os"
	"os/signal"
	"path/filepath"
	"regexp"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

// === Config ===
var (
	apiEndpoint string
	apiModel    string
	apiKeys     []string
	apiKeyIdx   atomic.Uint64
	apiFormat   string // "anthropic" | "openai-chat" | "openai-responses"
	baseDir     string
)

// nextKey returns the next API key in round-robin fashion
func nextKey() string {
	idx := apiKeyIdx.Add(1) - 1
	return apiKeys[idx%uint64(len(apiKeys))]
}

const (
	maxSolveRetries = 10  // max solve attempts per problem before moving on
	maxAPIRetries   = 999 // unlimited API retries for transient errors
	retryDelay      = 2 * time.Second
	apiTimeout      = 600 * time.Second // 10 min for very long reasoning
)

var (
	outputDir       string
	checkpointFile  string
	rawCotDir       string
	distilledCotDir string
	techniquesDir   string
	apiConfigCache  string
)

// === Types ===
type Problem struct {
	ID     string `json:"id"`
	Prompt string `json:"prompt"`
	Answer string `json:"answer"`
}

type RawCOT struct {
	ID            string `json:"id"`
	Prompt        string `json:"prompt"`
	Answer        string `json:"answer"`
	FullReasoning string `json:"full_reasoning"`
}

type DistilledCOT struct {
	ID           string `json:"id"`
	Prompt       string `json:"prompt"`
	Answer       string `json:"answer"`
	DistilledCOT string `json:"distilled_cot"`
}

type TechniqueData struct {
	ID        string `json:"id"`
	Prompt    string `json:"prompt"`
	Answer    string `json:"answer"`
	Technique string `json:"technique"`
}

type Checkpoint struct {
	Solved              map[string]*SolveInfo `json:"solved"`
	Distilled           map[string]string     `json:"distilled"`
	TechniquesExtracted map[string]string     `json:"techniques_extracted"`
}

type SolveInfo struct {
	RawPath  string `json:"raw_path"`
	Attempts int    `json:"attempts"`
	Failed   bool   `json:"failed,omitempty"`
}

// API message type (shared between formats)
type ChatMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"` // string or []ContentBlock
}

// Anthropic format types
type AnthropicRequest struct {
	Model       string        `json:"model"`
	MaxTokens   int           `json:"max_tokens"`
	System      string        `json:"system,omitempty"`
	Messages    []ChatMessage `json:"messages"`
	Temperature float64       `json:"temperature"`
}

type AnthropicResponseContent struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	Thinking string `json:"thinking,omitempty"`
}

type AnthropicResponse struct {
	Content []AnthropicResponseContent `json:"content"`
}

// OpenAI format types
type OpenAIRequest struct {
	Model       string        `json:"model"`
	MaxTokens   int           `json:"max_tokens"`
	Messages    []ChatMessage `json:"messages"`
	Temperature float64       `json:"temperature"`
	Stream      bool          `json:"stream"`
}

// OpenAI Responses API types
type OpenAIResponsesInputText struct {
	Type string `json:"type"`
	Text string `json:"text"`
}

type OpenAIResponsesInputItem struct {
	Role    string                     `json:"role"`
	Content []OpenAIResponsesInputText `json:"content"`
}

type OpenAIResponsesRequest struct {
	Model           string                     `json:"model"`
	Input           []OpenAIResponsesInputItem `json:"input"`
	Temperature     float64                    `json:"temperature,omitempty"`
	MaxOutputTokens int                        `json:"max_output_tokens,omitempty"`
	MaxTokens       int                        `json:"max_tokens,omitempty"`
	Stream          bool                       `json:"stream"`
}

type OpenAIChoiceMessage struct {
	Role             string `json:"role"`
	Content          string `json:"content"`
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

type OpenAIChoice struct {
	Message OpenAIChoiceMessage `json:"message"`
}

type OpenAIResponse struct {
	Choices []OpenAIChoice `json:"choices"`
}

type OpenAIStreamDelta struct {
	Role             string `json:"role,omitempty"`
	Content          string `json:"content,omitempty"`
	ReasoningContent string `json:"reasoning_content,omitempty"`
}

type OpenAIStreamChoice struct {
	Delta OpenAIStreamDelta `json:"delta"`
}

type OpenAIStreamChunk struct {
	Choices []OpenAIStreamChoice `json:"choices"`
}

type APIConfigEntry struct {
	Format    string `json:"format"`
	Endpoint  string `json:"endpoint"`
	UpdatedAt string `json:"updated_at"`
}

type APICandidate struct {
	Format   string
	Endpoint string
}

// === Globals ===
var (
	shutdown   atomic.Bool
	ckptMu     sync.Mutex
	apiCfgMu   sync.RWMutex
	noChatMode atomic.Bool
	httpClient *http.Client
)

// === Load .env ===
func trimEnvValue(v string) string {
	v = strings.TrimSpace(v)
	if len(v) >= 2 {
		if (v[0] == '"' && v[len(v)-1] == '"') || (v[0] == '\'' && v[len(v)-1] == '\'') {
			v = v[1 : len(v)-1]
		}
	}
	return strings.TrimSpace(v)
}

func loadEnv(path string) (map[string]string, []string) {
	env := map[string]string{}
	var extraKeys []string
	f, err := os.Open(path)
	if err != nil {
		return env, extraKeys
	}
	defer f.Close()
	buf, _ := io.ReadAll(f)
	for _, line := range strings.Split(string(buf), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		if strings.HasPrefix(line, "export ") {
			line = strings.TrimSpace(strings.TrimPrefix(line, "export "))
		}
		// Lines that look like raw API keys (sk-... without = sign)
		if strings.HasPrefix(line, "sk-") && !strings.Contains(line, "=") {
			extraKeys = append(extraKeys, line)
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) == 2 {
			key := strings.TrimSpace(parts[0])
			val := trimEnvValue(parts[1])
			env[key] = val
		}
	}
	return env, extraKeys
}

func loadAPIConfigCache(path string) map[string]APIConfigEntry {
	cache := map[string]APIConfigEntry{}
	data, err := os.ReadFile(path)
	if err != nil {
		return cache
	}
	_ = json.Unmarshal(data, &cache)
	return cache
}

func saveAPIConfigCache(path string, cache map[string]APIConfigEntry) {
	_ = os.MkdirAll(filepath.Dir(path), 0755)
	data, _ := json.MarshalIndent(cache, "", "  ")
	_ = os.WriteFile(path, data, 0644)
}

func inferAPIFormatFromEndpoint(endpoint string) string {
	ep := strings.ToLower(endpoint)
	switch {
	case strings.Contains(ep, "/v1/messages") || strings.HasSuffix(ep, "/messages"):
		return "anthropic"
	case strings.Contains(ep, "/chat/completions"):
		return "openai-chat"
	case strings.Contains(ep, "/responses"):
		return "openai-responses"
	default:
		return ""
	}
}

func normalizeFormat(s string) string {
	s = strings.TrimSpace(strings.ToLower(s))
	switch s {
	case "anthropic":
		return "anthropic"
	case "openai", "openai-chat", "chat", "chat-completions":
		return "openai-chat"
	case "openai-responses", "responses", "response":
		return "openai-responses"
	default:
		return ""
	}
}

func canonicalBaseURL(apiBase string) string {
	return strings.TrimRight(strings.TrimSpace(apiBase), "/")
}

func defaultEndpointForFormat(apiBase, format string) string {
	base := canonicalBaseURL(apiBase)
	if base == "" {
		return ""
	}
	switch format {
	case "anthropic":
		if strings.HasSuffix(base, "/v1/messages") {
			return base
		}
		if strings.HasSuffix(base, "/v1") {
			return base + "/messages"
		}
		if strings.HasSuffix(base, "/anthropic") || strings.Contains(base, "/anthropic/") {
			return base + "/v1/messages"
		}
		return base + "/v1/messages"
	case "openai-chat":
		if strings.HasSuffix(base, "/chat/completions") {
			return base
		}
		if strings.HasSuffix(base, "/v1") || strings.Contains(base, "/v1/") {
			return base + "/chat/completions"
		}
		return base + "/v1/chat/completions"
	case "openai-responses":
		if strings.HasSuffix(base, "/responses") {
			return base
		}
		if strings.HasSuffix(base, "/v1") || strings.Contains(base, "/v1/") {
			return base + "/responses"
		}
		return base + "/v1/responses"
	default:
		return ""
	}
}

func buildAPICandidates(apiBase, formatHint, endpointHint string) []APICandidate {
	base := canonicalBaseURL(apiBase)
	formatHint = normalizeFormat(formatHint)
	endpointHint = strings.TrimSpace(endpointHint)
	if endpointHint != "" {
		endpointHint = strings.TrimRight(endpointHint, "/")
	}

	seen := map[string]bool{}
	add := func(format, endpoint string, out *[]APICandidate) {
		format = normalizeFormat(format)
		endpoint = strings.TrimRight(strings.TrimSpace(endpoint), "/")
		if format == "" || endpoint == "" {
			return
		}
		key := format + "|" + endpoint
		if seen[key] {
			return
		}
		seen[key] = true
		*out = append(*out, APICandidate{Format: format, Endpoint: endpoint})
	}

	var candidates []APICandidate
	if endpointHint != "" {
		if formatHint != "" {
			add(formatHint, endpointHint, &candidates)
		} else {
			if inferred := inferAPIFormatFromEndpoint(endpointHint); inferred != "" {
				add(inferred, endpointHint, &candidates)
			}
		}
	}

	if formatHint != "" {
		add(formatHint, defaultEndpointForFormat(base, formatHint), &candidates)
	}

	if base != "" {
		if inferred := inferAPIFormatFromEndpoint(base); inferred != "" {
			add(inferred, base, &candidates)
		}
		add("openai-chat", defaultEndpointForFormat(base, "openai-chat"), &candidates)
		add("openai-responses", defaultEndpointForFormat(base, "openai-responses"), &candidates)
		add("anthropic", defaultEndpointForFormat(base, "anthropic"), &candidates)

		// Also try direct non-v1 paths for OpenAI-compatible gateways.
		if !strings.HasSuffix(base, "/v1") && !strings.Contains(base, "/v1/") {
			add("openai-chat", base+"/chat/completions", &candidates)
			add("openai-responses", base+"/responses", &candidates)
		}
	}

	return candidates
}

func probeAPIConfig(format, endpoint, key, model string) error {
	var reqBody []byte
	switch format {
	case "anthropic":
		req := AnthropicRequest{
			Model:       model,
			MaxTokens:   8,
			System:      "",
			Messages:    []ChatMessage{userMsg("Reply with OK")},
			Temperature: 0,
		}
		reqBody, _ = json.Marshal(req)
	case "openai-chat":
		req := OpenAIRequest{
			Model:       model,
			MaxTokens:   8,
			Messages:    []ChatMessage{userMsg("Reply with OK")},
			Temperature: 0,
			Stream:      false,
		}
		reqBody, _ = json.Marshal(req)
	case "openai-responses":
		req := OpenAIResponsesRequest{
			Model:           model,
			Input:           []OpenAIResponsesInputItem{{Role: "user", Content: []OpenAIResponsesInputText{{Type: "input_text", Text: "Reply with OK"}}}},
			Temperature:     0,
			MaxOutputTokens: 8,
			MaxTokens:       8,
			Stream:          false,
		}
		reqBody, _ = json.Marshal(req)
	default:
		return fmt.Errorf("unknown format: %s", format)
	}

	req, err := http.NewRequest("POST", endpoint, strings.NewReader(string(reqBody)))
	if err != nil {
		return err
	}
	if format == "anthropic" {
		req.Header.Set("x-api-key", key)
		req.Header.Set("anthropic-version", "2023-06-01")
	} else {
		req.Header.Set("Authorization", "Bearer "+key)
	}
	req.Header.Set("Content-Type", "application/json")

	client := &http.Client{Timeout: 30 * time.Second}
	resp, err := client.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != 200 {
		snippet := strings.TrimSpace(string(body))
		if len(snippet) > 200 {
			snippet = snippet[:200]
		}
		return fmt.Errorf("status=%d body=%s", resp.StatusCode, snippet)
	}

	trimmed := strings.TrimSpace(string(body))
	if trimmed == "" {
		return fmt.Errorf("empty response body")
	}

	var generic map[string]interface{}
	if err := json.Unmarshal(body, &generic); err != nil {
		snippet := trimmed
		if len(snippet) > 200 {
			snippet = snippet[:200]
		}
		return fmt.Errorf("non-json response body: %q", snippet)
	}
	if _, ok := generic["error"]; ok {
		return fmt.Errorf("response contains error field")
	}
	return nil
}

func resolveAPIConfig(apiBase, model, formatHint, endpointHint string) (string, string, error) {
	base := canonicalBaseURL(apiBase)
	cacheKey := base + "||" + strings.TrimSpace(model)
	cache := loadAPIConfigCache(apiConfigCache)

	if entry, ok := cache[cacheKey]; ok && entry.Format != "" && entry.Endpoint != "" {
		log.Printf("Trying cached API config for [%s]: %s @ %s", cacheKey, entry.Format, entry.Endpoint)
		if err := probeAPIConfig(entry.Format, entry.Endpoint, apiKeys[0], model); err == nil {
			return entry.Format, entry.Endpoint, nil
		}
		log.Printf("Cached config failed for [%s], will rediscover", cacheKey)
	}

	candidates := buildAPICandidates(base, formatHint, endpointHint)
	if len(candidates) == 0 {
		return "", "", fmt.Errorf("no API endpoint candidates generated")
	}

	var lastErr error
	for _, c := range candidates {
		log.Printf("Probing API config: %s @ %s", c.Format, c.Endpoint)
		if err := probeAPIConfig(c.Format, c.Endpoint, apiKeys[0], model); err != nil {
			lastErr = err
			log.Printf("Probe failed: %v", err)
			continue
		}
		cache[cacheKey] = APIConfigEntry{
			Format:    c.Format,
			Endpoint:  c.Endpoint,
			UpdatedAt: time.Now().Format(time.RFC3339),
		}
		saveAPIConfigCache(apiConfigCache, cache)
		log.Printf("Selected API config: %s @ %s", c.Format, c.Endpoint)
		return c.Format, c.Endpoint, nil
	}
	if lastErr == nil {
		lastErr = fmt.Errorf("all candidates failed")
	}
	return "", "", fmt.Errorf("failed to resolve API config for %s: %w", cacheKey, lastErr)
}

func getAPIConfig() (string, string) {
	apiCfgMu.RLock()
	defer apiCfgMu.RUnlock()
	return apiFormat, apiEndpoint
}

func setAPIConfig(format, endpoint string) {
	apiCfgMu.Lock()
	defer apiCfgMu.Unlock()
	apiFormat = format
	apiEndpoint = endpoint
}

func endpointBaseFromResolved(endpoint string) string {
	ep := strings.TrimRight(strings.TrimSpace(endpoint), "/")
	switch {
	case strings.HasSuffix(ep, "/v1/chat/completions"):
		return strings.TrimSuffix(ep, "/v1/chat/completions")
	case strings.HasSuffix(ep, "/chat/completions"):
		return strings.TrimSuffix(ep, "/chat/completions")
	case strings.HasSuffix(ep, "/v1/responses"):
		return strings.TrimSuffix(ep, "/v1/responses")
	case strings.HasSuffix(ep, "/responses"):
		return strings.TrimSuffix(ep, "/responses")
	case strings.HasSuffix(ep, "/v1/messages"):
		return strings.TrimSuffix(ep, "/v1/messages")
	case strings.HasSuffix(ep, "/messages"):
		return strings.TrimSuffix(ep, "/messages")
	default:
		return ep
	}
}

func switchResponsesToChatIfHealthy(reason string) bool {
	if noChatMode.Load() {
		return false
	}
	format, endpoint := getAPIConfig()
	if format != "openai-responses" {
		return false
	}
	base := endpointBaseFromResolved(endpoint)
	chatEndpoint := defaultEndpointForFormat(base, "openai-chat")
	if chatEndpoint == "" || chatEndpoint == endpoint {
		return false
	}
	if err := probeAPIConfig("openai-chat", chatEndpoint, apiKeys[0], apiModel); err != nil {
		log.Printf("Chat fallback probe failed: %v", err)
		errText := strings.ToLower(err.Error())
		if strings.Contains(errText, "unsupported legacy protocol") || strings.Contains(errText, "chat/completions is not supported") {
			noChatMode.Store(true)
			log.Printf("Detected responses-only provider, disabling chat fallback.")
		}
		return false
	}
	setAPIConfig("openai-chat", chatEndpoint)
	log.Printf("Switched API config to openai-chat @ %s due to: %s", chatEndpoint, reason)
	return true
}

func switchResponsesEndpointIfHealthy(reason string) bool {
	format, endpoint := getAPIConfig()
	if format != "openai-responses" {
		return false
	}

	base := endpointBaseFromResolved(endpoint)
	var candidates []string
	add := func(s string) {
		s = strings.TrimRight(strings.TrimSpace(s), "/")
		if s == "" || s == endpoint {
			return
		}
		for _, e := range candidates {
			if e == s {
				return
			}
		}
		candidates = append(candidates, s)
	}

	add(defaultEndpointForFormat(base, "openai-responses"))
	if strings.HasSuffix(endpoint, "/v1/responses") {
		add(base + "/responses")
	} else if strings.HasSuffix(endpoint, "/responses") {
		add(base + "/v1/responses")
	}

	for _, cand := range candidates {
		if err := probeAPIConfig("openai-responses", cand, apiKeys[0], apiModel); err != nil {
			log.Printf("Responses endpoint switch probe failed (%s): %v", cand, err)
			continue
		}
		setAPIConfig("openai-responses", cand)
		log.Printf("Switched API config to openai-responses @ %s due to: %s", cand, reason)
		return true
	}
	return false
}

func buildReqBodyForFormat(format, systemPrompt string, messages []ChatMessage, temperature float64, maxTokens int) ([]byte, error) {
	switch format {
	case "openai-chat":
		allMsgs := append([]ChatMessage{systemMsg(systemPrompt)}, messages...)
		req := OpenAIRequest{
			Model:       apiModel,
			MaxTokens:   maxTokens,
			Messages:    allMsgs,
			Temperature: temperature,
			Stream:      true,
		}
		return json.Marshal(req)
	case "openai-responses":
		req := OpenAIResponsesRequest{
			Model:           apiModel,
			Input:           toResponsesInput(systemPrompt, messages),
			Temperature:     temperature,
			MaxOutputTokens: maxTokens,
			MaxTokens:       maxTokens,
			Stream:          false,
		}
		return json.Marshal(req)
	case "anthropic":
		req := AnthropicRequest{
			Model:       apiModel,
			MaxTokens:   maxTokens,
			System:      systemPrompt,
			Messages:    messages,
			Temperature: temperature,
		}
		return json.Marshal(req)
	default:
		return nil, fmt.Errorf("unsupported api format: %s", format)
	}
}

func appendWithSep(dst *string, v string) {
	v = strings.TrimSpace(v)
	if v == "" {
		return
	}
	if *dst == "" {
		*dst = v
		return
	}
	*dst += "\n" + v
}

func flattenText(v interface{}) string {
	switch vv := v.(type) {
	case string:
		return vv
	case []interface{}:
		var out string
		for _, item := range vv {
			appendWithSep(&out, flattenText(item))
		}
		return out
	case map[string]interface{}:
		if text, ok := vv["text"].(string); ok && text != "" {
			return text
		}
		if delta, ok := vv["delta"].(string); ok && delta != "" {
			return delta
		}
		if content, ok := vv["content"]; ok {
			return flattenText(content)
		}
	}
	return ""
}

func parseContentBlocks(v interface{}) (reasoning string, content string) {
	switch vv := v.(type) {
	case string:
		appendWithSep(&content, vv)
	case []interface{}:
		for _, item := range vv {
			r, c := parseContentBlocks(item)
			appendWithSep(&reasoning, r)
			appendWithSep(&content, c)
		}
	case map[string]interface{}:
		t, _ := vv["type"].(string)
		t = strings.ToLower(strings.TrimSpace(t))
		text := flattenText(vv)
		switch {
		case strings.Contains(t, "reasoning"), strings.Contains(t, "summary"):
			appendWithSep(&reasoning, text)
		case t == "output_text", t == "text", t == "input_text", t == "":
			appendWithSep(&content, text)
		default:
			appendWithSep(&content, text)
		}
	}
	return strings.TrimSpace(reasoning), strings.TrimSpace(content)
}

func combineReasoningAndContent(reasoning, content string) string {
	reasoning = strings.TrimSpace(reasoning)
	content = strings.TrimSpace(content)
	if reasoning != "" && content != "" {
		return reasoning + "\n\n" + content
	}
	if content != "" {
		return content
	}
	return reasoning
}

func parseOpenAIChatJSON(respBody []byte) (string, error) {
	var generic map[string]interface{}
	if err := json.Unmarshal(respBody, &generic); err != nil {
		return "", err
	}
	if _, ok := generic["error"]; ok {
		return "", fmt.Errorf("openai chat response contains error field")
	}

	var reasoning, content string
	choices, _ := generic["choices"].([]interface{})
	if len(choices) == 0 {
		return "", fmt.Errorf("no choices found")
	}
	choice, _ := choices[0].(map[string]interface{})
	msg, _ := choice["message"].(map[string]interface{})
	if msg != nil {
		if rc, ok := msg["reasoning_content"].(string); ok {
			appendWithSep(&reasoning, rc)
		}
		if c, ok := msg["content"].(string); ok {
			appendWithSep(&content, c)
		} else if cAny, ok := msg["content"]; ok {
			r, c := parseContentBlocks(cAny)
			appendWithSep(&reasoning, r)
			appendWithSep(&content, c)
		}
	}
	result := combineReasoningAndContent(reasoning, content)
	if strings.TrimSpace(result) == "" {
		return "", fmt.Errorf("empty choices message")
	}
	return result, nil
}

func parseOpenAIResponsesJSON(respBody []byte) (string, error) {
	var generic map[string]interface{}
	if err := json.Unmarshal(respBody, &generic); err != nil {
		return "", err
	}
	if _, ok := generic["error"]; ok {
		return "", fmt.Errorf("responses payload contains error field")
	}

	var reasoning, content string
	if outText, ok := generic["output_text"]; ok {
		appendWithSep(&content, flattenText(outText))
	}

	if output, ok := generic["output"].([]interface{}); ok {
		for _, item := range output {
			m, ok := item.(map[string]interface{})
			if !ok {
				continue
			}
			t, _ := m["type"].(string)
			t = strings.ToLower(strings.TrimSpace(t))
			switch t {
			case "reasoning":
				if summary, ok := m["summary"]; ok {
					r, c := parseContentBlocks(summary)
					appendWithSep(&reasoning, r)
					appendWithSep(&reasoning, c)
				}
				if blocks, ok := m["content"]; ok {
					r, c := parseContentBlocks(blocks)
					appendWithSep(&reasoning, r)
					appendWithSep(&reasoning, c)
				}
			default:
				if blocks, ok := m["content"]; ok {
					r, c := parseContentBlocks(blocks)
					appendWithSep(&reasoning, r)
					appendWithSep(&content, c)
				}
			}
		}
	}

	result := combineReasoningAndContent(reasoning, content)
	if strings.TrimSpace(result) != "" {
		return result, nil
	}

	// Some gateways reply with chat-completions schema even on /responses.
	return parseOpenAIChatJSON(respBody)
}

func chatMessageText(content interface{}) string {
	switch v := content.(type) {
	case string:
		return v
	default:
		text := flattenText(v)
		if text != "" {
			return text
		}
		b, _ := json.Marshal(v)
		return string(b)
	}
}

func toResponsesInput(systemPrompt string, messages []ChatMessage) []OpenAIResponsesInputItem {
	var input []OpenAIResponsesInputItem
	if strings.TrimSpace(systemPrompt) != "" {
		input = append(input, OpenAIResponsesInputItem{
			Role:    "system",
			Content: []OpenAIResponsesInputText{{Type: "input_text", Text: systemPrompt}},
		})
	}
	for _, msg := range messages {
		text := chatMessageText(msg.Content)
		role := strings.TrimSpace(msg.Role)
		if role == "" {
			role = "user"
		}
		input = append(input, OpenAIResponsesInputItem{
			Role:    role,
			Content: []OpenAIResponsesInputText{{Type: "input_text", Text: text}},
		})
	}
	return input
}

// === API call (supports Anthropic / OpenAI Chat / OpenAI Responses) ===
func apiCall(systemPrompt string, messages []ChatMessage, temperature float64, maxTokens int) (string, error) {
	for attempt := 0; attempt < maxAPIRetries; attempt++ {
		if shutdown.Load() {
			return "", fmt.Errorf("shutdown requested")
		}
		format, endpoint := getAPIConfig()
		reqBody, err := buildReqBodyForFormat(format, systemPrompt, messages, temperature, maxTokens)
		if err != nil {
			return "", err
		}
		key := nextKey()
		req, err := http.NewRequest("POST", endpoint, strings.NewReader(string(reqBody)))
		if err != nil {
			return "", err
		}

		if format == "openai-chat" || format == "openai-responses" {
			req.Header.Set("Authorization", "Bearer "+key)
		} else {
			req.Header.Set("x-api-key", key)
			req.Header.Set("anthropic-version", "2023-06-01")
		}
		req.Header.Set("Content-Type", "application/json")

		resp, err := httpClient.Do(req)
		if err != nil {
			delay := retryDelay * time.Duration(math.Pow(2, float64(attempt)))
			if delay > 5*time.Minute {
				delay = 5 * time.Minute
			}
			log.Printf("  API error (attempt %d/%d): %v. Retrying in %v...", attempt+1, maxAPIRetries, err, delay)
			time.Sleep(delay)
			continue
		}

		if resp.StatusCode != 200 {
			respBody, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			snippet := string(respBody)
			if len(snippet) > 200 {
				snippet = snippet[:200]
			}
			delay := retryDelay * time.Duration(math.Pow(2, float64(attempt)))
			log.Printf("  API error %d (attempt %d/%d): %s. Retrying in %v...", resp.StatusCode, attempt+1, maxAPIRetries, snippet, delay)
			time.Sleep(delay)
			continue
		}

		var result string

		if format == "openai-chat" {
			var reasoning, content string
			scanner := bufio.NewScanner(resp.Body)
			scanner.Buffer(make([]byte, 0, 64*1024), 1024*1024)
			for scanner.Scan() {
				line := scanner.Text()
				if !strings.HasPrefix(line, "data: ") {
					continue
				}
				data := strings.TrimPrefix(line, "data: ")
				if data == "[DONE]" {
					break
				}
				var chunk OpenAIStreamChunk
				if err := json.Unmarshal([]byte(data), &chunk); err != nil {
					continue
				}
				if len(chunk.Choices) > 0 {
					delta := chunk.Choices[0].Delta
					reasoning += delta.ReasoningContent
					content += delta.Content
				}
			}
			if err := scanner.Err(); err != nil {
				resp.Body.Close()
				delay := retryDelay * time.Duration(math.Pow(2, float64(attempt)))
				log.Printf("  Stream parse error (attempt %d/%d): %v. Retrying in %v...", attempt+1, maxAPIRetries, err, delay)
				time.Sleep(delay)
				continue
			}
			resp.Body.Close()
			result = combineReasoningAndContent(reasoning, content)
		} else if format == "openai-responses" {
			respBody, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			parsed, err := parseOpenAIResponsesJSON(respBody)
			if err != nil {
				bodyTrim := strings.TrimSpace(string(respBody))
				preview := bodyTrim
				if len(preview) > 120 {
					preview = preview[:120]
				}
				if strings.HasPrefix(strings.ToLower(bodyTrim), "<!doctype html") || strings.HasPrefix(bodyTrim, "<") {
					if switchResponsesEndpointIfHealthy("responses endpoint returned HTML") {
						log.Printf("  Responses returned HTML, switched to alternate /responses endpoint and retrying immediately.")
						continue
					}
					if switchResponsesToChatIfHealthy("responses endpoint returned HTML") {
						log.Printf("  Responses returned HTML, switched to chat/completions and retrying immediately.")
						continue
					}
				}
				delay := retryDelay * time.Duration(math.Pow(2, float64(attempt)))
				log.Printf("  Responses parse error (attempt %d/%d): %v. Body preview: %q. Retrying in %v...", attempt+1, maxAPIRetries, err, preview, delay)
				time.Sleep(delay)
				continue
			}
			result = parsed
		} else {
			var apiResp AnthropicResponse
			respBody, _ := io.ReadAll(resp.Body)
			resp.Body.Close()
			if err := json.Unmarshal(respBody, &apiResp); err != nil {
				delay := retryDelay * time.Duration(math.Pow(2, float64(attempt)))
				log.Printf("  JSON parse error (attempt %d/%d): %v. Retrying in %v...", attempt+1, maxAPIRetries, err, delay)
				time.Sleep(delay)
				continue
			}
			var thinking, text string
			for _, block := range apiResp.Content {
				switch block.Type {
				case "thinking":
					thinking += block.Thinking
				case "text":
					text += block.Text
				}
			}
			result = combineReasoningAndContent(thinking, text)
		}

		if result == "" {
			delay := retryDelay * time.Duration(math.Pow(2, float64(attempt)))
			log.Printf("  Empty response (attempt %d/%d). Retrying in %v...", attempt+1, maxAPIRetries, delay)
			time.Sleep(delay)
			continue
		}
		return result, nil
	}
	return "", fmt.Errorf("API call failed after %d retries", maxAPIRetries)
}

// === Checkpoint ===
func loadCheckpoint() *Checkpoint {
	ckpt := &Checkpoint{
		Solved:              make(map[string]*SolveInfo),
		Distilled:           make(map[string]string),
		TechniquesExtracted: make(map[string]string),
	}
	data, err := os.ReadFile(checkpointFile)
	if err != nil {
		return ckpt
	}
	json.Unmarshal(data, ckpt)
	return ckpt
}

func saveCheckpoint(ckpt *Checkpoint) {
	ckptMu.Lock()
	defer ckptMu.Unlock()
	data, _ := json.MarshalIndent(ckpt, "", "  ")
	os.WriteFile(checkpointFile, data, 0644)
}

// === Load training data ===
func loadTrainData() ([]Problem, error) {
	f, err := os.Open(filepath.Join(baseDir, "train.csv"))
	if err != nil {
		return nil, err
	}
	defer f.Close()
	reader := csv.NewReader(f)
	reader.Read() // skip header
	var problems []Problem
	for {
		record, err := reader.Read()
		if err == io.EOF {
			break
		}
		if err != nil {
			continue
		}
		problems = append(problems, Problem{ID: record[0], Prompt: record[1], Answer: record[2]})
	}
	return problems, nil
}

// === Answer extraction via regex (fallback) ===
var boxedRe = regexp.MustCompile(`\\boxed\{([^}]+)\}`)

func extractBoxedAnswer(text string) string {
	matches := boxedRe.FindAllStringSubmatch(text, -1)
	if len(matches) > 0 {
		return strings.TrimSpace(matches[len(matches)-1][1])
	}
	return ""
}

// === Agent 3: Judge — use LLM to extract and compare answers ===
const judgeSystem = `You are an answer verification assistant. Your ONLY job is to extract the final answer from the model's response and compare it with the expected answer.

Rules:
1. Extract ONLY the final numerical or textual answer from the response
2. Ignore all reasoning, explanations, and intermediate steps
3. If the answer is inside \boxed{}, extract it from there
4. If not in \boxed{}, look for the final answer at the end or clearly stated as the answer
5. Compare the extracted answer with the expected answer — consider mathematically equivalent answers as correct (e.g., 1/2 = 0.5, 2.0 = 2)

Respond with EXACTLY one of these two words:
- CORRECT (if the answers match or are equivalent)
- WRONG (if they don't match)

Nothing else. Just one word.`

func judgeAnswer(response, correctAnswer string) bool {
	truncated := response
	if len(truncated) > 3000 {
		truncated = truncated[len(truncated)-3000:]
	}
	messages := []ChatMessage{
		userMsg(fmt.Sprintf("Expected answer: %s\n\nModel response:\n%s\n\nIs the final answer in the response equivalent to the expected answer? Reply CORRECT or WRONG.", correctAnswer, truncated)),
	}
	result, err := apiCall(judgeSystem, messages, 0.0, 16)
	if err != nil {
		log.Printf("  Judge API error: %v", err)
		return false
	}
	result = strings.TrimSpace(strings.ToUpper(result))
	return strings.HasPrefix(result, "CORRECT")
}

func normalizeAnswer(ans string) string {
	ans = strings.TrimSpace(ans)
	if f, err := strconv.ParseFloat(ans, 64); err == nil {
		if f == float64(int(f)) && !math.IsInf(f, 0) {
			return strconv.Itoa(int(f))
		}
		return fmt.Sprintf("%.4f", math.Round(f*10000)/10000)
	}
	return strings.ToLower(ans)
}

// helper: wrap string into ChatMessage
func userMsg(text string) ChatMessage {
	return ChatMessage{Role: "user", Content: text}
}

func assistantMsg(text string) ChatMessage {
	return ChatMessage{Role: "assistant", Content: text}
}

func systemMsg(text string) ChatMessage {
	return ChatMessage{Role: "system", Content: text}
}

// === Agent 1: Solver ===
const solverSystem = `You are a math and logic reasoning expert. Solve the given problem step by step.

Rules:
1. Think carefully and show your complete reasoning process
2. Put your final answer inside \boxed{...}
3. Show all intermediate steps and calculations
4. If you make a mistake, you will be told to try again without the correct answer`

func solveProblem(prompt, correctAnswer string) (*RawCOT, error) {
	messages := []ChatMessage{
		userMsg(prompt + "\n\nPut your final answer inside \\boxed{}."),
	}

	for attempt := 0; attempt < maxSolveRetries; attempt++ {
		if shutdown.Load() {
			return nil, fmt.Errorf("shutdown")
		}
		log.Printf("  Solve attempt %d/%d", attempt+1, maxSolveRetries)

		temperature := 1.0
		if attempt > 0 {
			temperature = math.Max(0.1, 1.0-0.1*float64(min(attempt, 5)))
		}

		response, err := apiCall(solverSystem, messages, temperature, 8192)
		if err != nil {
			return nil, err
		}

		// First try regex extraction
		extracted := extractBoxedAnswer(response)
		if extracted != "" && normalizeAnswer(extracted) == normalizeAnswer(correctAnswer) {
			log.Printf("  Correct answer on attempt %d (regex)", attempt+1)
			return &RawCOT{
				Prompt:        prompt,
				Answer:        correctAnswer,
				FullReasoning: response,
			}, nil
		}
		// Regex failed — use LLM judge
		if judgeAnswer(response, correctAnswer) {
			log.Printf("  Correct answer on attempt %d (judge)", attempt+1)
			return &RawCOT{
				Prompt:        prompt,
				Answer:        correctAnswer,
				FullReasoning: response,
			}, nil
		}

		log.Printf("  Wrong: got '%s', expected '%s'", extracted, correctAnswer)
		time.Sleep(2 * time.Second) // brief pause between attempts

		prev := response
		if len(prev) > 500 {
			prev = prev[len(prev)-500:]
		}
		messages = append(messages,
			assistantMsg(response),
			userMsg("Your previous answer is INCORRECT. The correct answer is NOT what you got.\n\nPlease reconsider the problem from scratch. Try a different approach if needed.\nShow your complete reasoning and put the final answer inside \\boxed{}.\n\nYour previous attempt:\n"+prev),
		)
	}
	return nil, fmt.Errorf("failed to solve after %d attempts", maxSolveRetries)
}

// === Agent 2: Distiller ===
const distillerSystem = `You are a reasoning compression expert. Your job is to take a detailed problem-solving process and create a concise, high-quality Chain-of-Thought (CoT).

Requirements:
1. Keep ONLY the essential reasoning steps that lead to the correct answer
2. Remove false starts, redundant calculations, and verbose explanations
3. Keep the logic clear and easy to follow
4. The CoT should be complete enough that someone could reproduce the answer from it
5. Put the final answer inside \boxed{...}

Output format: concise step-by-step reasoning followed by \boxed{answer}`

const techniqueSystem = `You are a math education expert. Based on the detailed solution process provided, extract:

1. Problem-solving technique(s) used
2. Key insights or shortcuts
3. Common pitfalls to avoid
4. A general approach that would work for similar problems

Be concise and actionable. Focus on transferable skills, not problem-specific details.`

func distillReasoning(prompt, fullReasoning string) (string, error) {
	messages := []ChatMessage{
		userMsg("Problem: " + prompt + "\n\nFull reasoning process:\n" + fullReasoning),
	}
	return apiCall(distillerSystem, messages, 1.0, 2048)
}

func extractTechnique(prompt, fullReasoning string) (string, error) {
	messages := []ChatMessage{
		userMsg("Problem: " + prompt + "\n\nFull reasoning process:\n" + fullReasoning),
	}
	return apiCall(techniqueSystem, messages, 1.0, 1024)
}

// === Process one problem ===
func processProblem(item Problem, ckpt *Checkpoint) {
	pid := item.ID
	prompt := item.Prompt
	answer := item.Answer

	// Phase 1: Solve
	if _, ok := ckpt.Solved[pid]; !ok {
		log.Printf("[%s] Solving: %s...", pid, trunc(prompt, 80))
		result, err := solveProblem(prompt, answer)
		if err != nil {
			ckptMu.Lock()
			ckpt.Solved[pid] = &SolveInfo{Attempts: maxSolveRetries, Failed: true}
			ckptMu.Unlock()
			saveCheckpoint(ckpt)
			log.Printf("[%s] ❌ Failed to solve: %v", pid, err)
			return
		}
		result.ID = pid
		rawPath := filepath.Join(rawCotDir, pid+".json")
		data, _ := json.MarshalIndent(result, "", "  ")
		os.WriteFile(rawPath, data, 0644)

		ckptMu.Lock()
		ckpt.Solved[pid] = &SolveInfo{RawPath: rawPath, Attempts: 1}
		ckptMu.Unlock()
		saveCheckpoint(ckpt)
		log.Printf("[%s] ✅ Saved raw CoT", pid)
	}

	// Skip if failed
	if ckpt.Solved[pid].Failed {
		return
	}

	// Load raw reasoning
	rawPath := ckpt.Solved[pid].RawPath
	if rawPath == "" {
		return
	}
	data, err := os.ReadFile(rawPath)
	if err != nil {
		return
	}
	var raw RawCOT
	json.Unmarshal(data, &raw)
	fullReasoning := raw.FullReasoning

	// Phase 2a: Distill
	if _, ok := ckpt.Distilled[pid]; !ok {
		log.Printf("[%s] Distilling...", pid)
		distilled, err := distillReasoning(prompt, fullReasoning)
		if err != nil {
			log.Printf("[%s] Distillation failed: %v", pid, err)
			return
		}
		distillPath := filepath.Join(distilledCotDir, pid+".json")
		d := DistilledCOT{ID: pid, Prompt: prompt, Answer: answer, DistilledCOT: distilled}
		data, _ := json.MarshalIndent(d, "", "  ")
		os.WriteFile(distillPath, data, 0644)

		ckptMu.Lock()
		ckpt.Distilled[pid] = distillPath
		ckptMu.Unlock()
		saveCheckpoint(ckpt)
		log.Printf("[%s] ✅ Distilled CoT saved", pid)
	}

	// Phase 2b: Extract technique
	if _, ok := ckpt.TechniquesExtracted[pid]; !ok {
		log.Printf("[%s] Extracting technique...", pid)
		technique, err := extractTechnique(prompt, fullReasoning)
		if err != nil {
			log.Printf("[%s] Technique extraction failed: %v", pid, err)
			return
		}
		techPath := filepath.Join(techniquesDir, pid+".json")
		t := TechniqueData{ID: pid, Prompt: prompt, Answer: answer, Technique: technique}
		data, _ := json.MarshalIndent(t, "", "  ")
		os.WriteFile(techPath, data, 0644)

		ckptMu.Lock()
		ckpt.TechniquesExtracted[pid] = techPath
		ckptMu.Unlock()
		saveCheckpoint(ckpt)
		log.Printf("[%s] ✅ Technique saved", pid)
	}
}

// === Merge outputs ===
func mergeOutputs(ckpt *Checkpoint) {
	var fullData []RawCOT
	for _, info := range ckpt.Solved {
		if info.Failed || info.RawPath == "" {
			continue
		}
		data, err := os.ReadFile(info.RawPath)
		if err != nil {
			continue
		}
		var r RawCOT
		json.Unmarshal(data, &r)
		fullData = append(fullData, r)
	}

	var distilledData []DistilledCOT
	for _, path := range ckpt.Distilled {
		data, err := os.ReadFile(path)
		if err != nil {
			continue
		}
		var d DistilledCOT
		json.Unmarshal(data, &d)
		distilledData = append(distilledData, d)
	}

	data, _ := json.MarshalIndent(fullData, "", "  ")
	os.WriteFile(filepath.Join(outputDir, "full_cot_dataset.json"), data, 0644)

	data, _ = json.MarshalIndent(distilledData, "", "  ")
	os.WriteFile(filepath.Join(outputDir, "distilled_cot_dataset.json"), data, 0644)

	log.Printf("Merged: %d raw CoT, %d distilled CoT", len(fullData), len(distilledData))
}

func trunc(s string, n int) string {
	if len(s) <= n {
		return s
	}
	return s[:n]
}

// === Main ===
func main() {
	limitFlag := flag.Int("limit", 0, "Only process first N problems (0=all)")
	workersFlag := flag.Int("workers", 5, "Number of concurrent goroutines")
	envFlag := flag.String("env", ".env", "Env file to load (.env or .env2)")
	flag.Parse()

	// Setup dirs
	if wd, err := os.Getwd(); err == nil {
		baseDir = wd
	} else {
		exe, _ := os.Executable()
		baseDir = filepath.Dir(exe)
	}

	outputDir = filepath.Join(baseDir, "cot_output")
	checkpointFile = filepath.Join(outputDir, "checkpoint.json")
	rawCotDir = filepath.Join(outputDir, "raw_cot")
	distilledCotDir = filepath.Join(outputDir, "distilled_cot")
	techniquesDir = filepath.Join(outputDir, "techniques")
	apiConfigCache = filepath.Join(outputDir, "api_config_cache.json")

	for _, d := range []string{outputDir, rawCotDir, distilledCotDir, techniquesDir} {
		_ = os.MkdirAll(d, 0755)
	}

	// Load env file
	envFile := *envFlag
	envPath := filepath.Join(baseDir, envFile)
	if _, err := os.Stat(envPath); err != nil && envFile == ".env" {
		fallback := filepath.Join(baseDir, ".env2")
		if _, ferr := os.Stat(fallback); ferr == nil {
			envPath = fallback
		}
	}
	env, extraKeys := loadEnv(envPath)
	apiBase := env["API_BASE_URL"]
	if apiBase == "" {
		apiBase = os.Getenv("API_BASE_URL")
	}
	apiModel = env["API_MODEL"]
	if apiModel == "" {
		apiModel = os.Getenv("API_MODEL")
	}
	primary := env["API_KEY"]
	if primary == "" {
		primary = os.Getenv("API_KEY")
	}

	// Build key pool: primary key + all extra keys
	apiKeys = []string{}
	if strings.TrimSpace(primary) != "" {
		apiKeys = append(apiKeys, strings.TrimSpace(primary))
	}
	for _, k := range extraKeys {
		if kk := strings.TrimSpace(k); kk != "" {
			apiKeys = append(apiKeys, kk)
		}
	}

	formatHint := env["API_FORMAT"]
	if formatHint == "" {
		formatHint = os.Getenv("API_FORMAT")
	}
	endpointHint := env["API_ENDPOINT"]
	if endpointHint == "" {
		endpointHint = os.Getenv("API_ENDPOINT")
	}

	if apiBase == "" || apiModel == "" || len(apiKeys) == 0 {
		log.Fatal("Please fill in .env/.env2 with API_BASE_URL, API_MODEL, API_KEY")
	}

	// Setup logging before endpoint probing so discovery logs are persisted.
	logFile, err := os.OpenFile(filepath.Join(outputDir, "generation.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err == nil {
		log.SetOutput(io.MultiWriter(os.Stdout, logFile))
	}
	log.SetFlags(log.Ldate | log.Ltime)

	// HTTP client with connection pooling
	transport := &http.Transport{
		MaxIdleConns:        100,
		MaxIdleConnsPerHost: 100,
		IdleConnTimeout:     90 * time.Second,
	}
	httpClient = &http.Client{Timeout: apiTimeout, Transport: transport}

	resolvedFormat, resolvedEndpoint, err := resolveAPIConfig(apiBase, apiModel, formatHint, endpointHint)
	if err != nil {
		log.Fatalf("Failed to resolve API endpoint/format: %v", err)
	}
	setAPIConfig(resolvedFormat, resolvedEndpoint)
	log.Printf("Loaded env file: %s", envPath)
	log.Printf("Resolved API config: %s @ %s", resolvedFormat, resolvedEndpoint)

	// Graceful shutdown
	sigCh := make(chan os.Signal, 1)
	signal.Notify(sigCh, os.Interrupt)
	go func() {
		<-sigCh
		log.Println("Shutdown requested, finishing current tasks...")
		shutdown.Store(true)
	}()

	// Load data
	log.Printf("Loading training data from %s", filepath.Join(baseDir, "train.csv"))
	problems, err := loadTrainData()
	if err != nil {
		log.Fatalf("Failed to load train data: %v", err)
	}
	if *limitFlag > 0 {
		problems = problems[:min(*limitFlag, len(problems))]
	}
	total := len(problems)
	log.Printf("Loaded %d problems (limit=%d, workers=%d)", total, *limitFlag, *workersFlag)
	_, activeEndpoint := getAPIConfig()
	log.Printf("Using model: %s @ %s (with %d API keys)", apiModel, activeEndpoint, len(apiKeys))

	ckpt := loadCheckpoint()
	log.Printf("Checkpoint: %d solved, %d distilled, %d techniques",
		len(ckpt.Solved), len(ckpt.Distilled), len(ckpt.TechniquesExtracted))

	// Filter to todo items
	var todo []Problem
	for _, p := range problems {
		info, solved := ckpt.Solved[p.ID]
		if !solved {
			todo = append(todo, p)
			continue
		}
		if info.Failed {
			todo = append(todo, p)
			continue
		}
		if _, ok := ckpt.Distilled[p.ID]; !ok {
			todo = append(todo, p)
			continue
		}
		if _, ok := ckpt.TechniquesExtracted[p.ID]; !ok {
			todo = append(todo, p)
			continue
		}
	}
	log.Printf("Remaining to process: %d/%d", len(todo), total)

	if len(todo) == 0 {
		log.Println("Nothing to do!")
		mergeOutputs(ckpt)
		return
	}

	// Worker pool
	ch := make(chan Problem, *workersFlag)
	var wg sync.WaitGroup
	var completed atomic.Int64

	for i := 0; i < *workersFlag; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for item := range ch {
				if shutdown.Load() {
					return
				}
				processProblem(item, ckpt)
				done := completed.Add(1)
				log.Printf("Progress: %d/%d (%.1f%%)", done, int64(len(todo)), float64(done)/float64(len(todo))*100)
			}
		}()
	}

	for _, item := range todo {
		if shutdown.Load() {
			break
		}
		ch <- item
	}
	close(ch)
	wg.Wait()

	// Merge
	log.Println("Merging all outputs...")
	mergeOutputs(ckpt)
	log.Println("Done!")
}
