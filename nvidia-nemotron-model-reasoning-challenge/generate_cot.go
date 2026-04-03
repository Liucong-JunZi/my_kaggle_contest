package main

import (
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
	apiKey      string
	baseDir     string
)

const (
	maxSolveRetries = 999999 // unlimited — keep going until correct
	maxAPIRetries   = 999    // unlimited API retries
	retryDelay      = 2 * time.Second
	apiTimeout      = 600 * time.Second // 10 min for very long reasoning
)

var (
	outputDir       string
	checkpointFile  string
	rawCotDir       string
	distilledCotDir string
	techniquesDir   string
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

// Anthropic API request/response types
type AnthropicContent struct {
	Type string `json:"type"`
	Text string `json:"json:"text,omitempty"`
}

type AnthropicMessage struct {
	Role    string      `json:"role"`
	Content interface{} `json:"content"` // string or []AnthropicContent
}

type AnthropicRequest struct {
	Model       string             `json:"model"`
	MaxTokens   int                `json:"max_tokens"`
	System      string             `json:"system,omitempty"`
	Messages    []AnthropicMessage `json:"messages"`
	Temperature float64            `json:"temperature"`
}

type AnthropicResponseContent struct {
	Type     string `json:"type"`
	Text     string `json:"text,omitempty"`
	Thinking string `json:"thinking,omitempty"`
}

type AnthropicResponse struct {
	Content []AnthropicResponseContent `json:"content"`
}

// === Globals ===
var (
	shutdown   atomic.Bool
	ckptMu     sync.Mutex
	httpClient *http.Client
)

// === Load .env ===
func loadEnv(path string) map[string]string {
	env := map[string]string{}
	f, err := os.Open(path)
	if err != nil {
		return env
	}
	defer f.Close()
	buf, _ := io.ReadAll(f)
	for _, line := range strings.Split(string(buf), "\n") {
		line = strings.TrimSpace(line)
		if line == "" || strings.HasPrefix(line, "#") {
			continue
		}
		parts := strings.SplitN(line, "=", 2)
		if len(parts) == 2 {
			env[strings.TrimSpace(parts[0])] = strings.TrimSpace(parts[1])
		}
	}
	return env
}

// === Anthropic API call ===
// messages: only user/assistant pairs (no system — passed separately)
func apiCall(systemPrompt string, messages []AnthropicMessage, temperature float64, maxTokens int) (string, error) {
	reqBody := AnthropicRequest{
		Model:       apiModel,
		MaxTokens:   maxTokens,
		System:      systemPrompt,
		Messages:    messages,
		Temperature: temperature,
	}
	body, _ := json.Marshal(reqBody)

	for attempt := 0; attempt < maxAPIRetries; attempt++ {
		if shutdown.Load() {
			return "", fmt.Errorf("shutdown requested")
		}
		req, err := http.NewRequest("POST", apiEndpoint, strings.NewReader(string(body)))
		if err != nil {
			return "", err
		}
		req.Header.Set("x-api-key", apiKey)
		req.Header.Set("anthropic-version", "2023-06-01")
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

		respBody, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		if resp.StatusCode != 200 {
			snippet := string(respBody)
			if len(snippet) > 200 {
				snippet = snippet[:200]
			}
			delay := retryDelay * time.Duration(math.Pow(2, float64(attempt)))
			log.Printf("  API error %d (attempt %d/%d): %s. Retrying in %v...", resp.StatusCode, attempt+1, maxAPIRetries, snippet, delay)
			time.Sleep(delay)
			continue
		}

		var apiResp AnthropicResponse
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
		result := thinking
		if result != "" && text != "" {
			result += "\n\n" + text
		} else if text != "" {
			result = text
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

// === Answer extraction ===
var boxedRe = regexp.MustCompile(`\\boxed\{([^}]+)\}`)

func extractBoxedAnswer(text string) string {
	matches := boxedRe.FindAllStringSubmatch(text, -1)
	if len(matches) > 0 {
		return strings.TrimSpace(matches[len(matches)-1][1])
	}
	return ""
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

// helper: wrap string into Anthropic content message
func userMsg(text string) AnthropicMessage {
	return AnthropicMessage{Role: "user", Content: text}
}

func assistantMsg(text string) AnthropicMessage {
	return AnthropicMessage{Role: "assistant", Content: text}
}

// === Agent 1: Solver ===
const solverSystem = `You are a math and logic reasoning expert. Solve the given problem step by step.

Rules:
1. Think carefully and show your complete reasoning process
2. Put your final answer inside \boxed{...}
3. Show all intermediate steps and calculations
4. If you make a mistake, you will be told to try again without the correct answer`

func solveProblem(prompt, correctAnswer string) (*RawCOT, error) {
	messages := []AnthropicMessage{
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

		extracted := extractBoxedAnswer(response)
		if extracted != "" && normalizeAnswer(extracted) == normalizeAnswer(correctAnswer) {
			log.Printf("  ✅ Correct answer on attempt %d", attempt+1)
			return &RawCOT{
				Prompt:        prompt,
				Answer:        correctAnswer,
				FullReasoning: response,
			}, nil
		}

		log.Printf("  ❌ Wrong: got '%s', expected '%s'", extracted, correctAnswer)
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
	messages := []AnthropicMessage{
		userMsg("Problem: " + prompt + "\n\nFull reasoning process:\n" + fullReasoning),
	}
	return apiCall(distillerSystem, messages, 1.0, 2048)
}

func extractTechnique(prompt, fullReasoning string) (string, error) {
	messages := []AnthropicMessage{
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
	flag.Parse()

	// Setup dirs
	if wd, err := os.Getwd(); err == nil {
		baseDir = wd
	} else {
		exe, _ := os.Executable()
		baseDir = filepath.Dir(exe)
	}

	// Load .env
	env := loadEnv(filepath.Join(baseDir, ".env"))
	apiBase := env["API_BASE_URL"]
	if apiBase == "" {
		apiBase = os.Getenv("API_BASE_URL")
	}
	apiModel = env["API_MODEL"]
	if apiModel == "" {
		apiModel = os.Getenv("API_MODEL")
	}
	apiKey = env["API_KEY"]
	if apiKey == "" {
		apiKey = os.Getenv("API_KEY")
	}
	// Anthropic API endpoint
	apiEndpoint = strings.TrimRight(apiBase, "/") + "/v1/messages"

	if apiBase == "" || apiModel == "" || apiKey == "" {
		log.Fatal("Please fill in .env file with API_BASE_URL, API_MODEL, API_KEY")
	}

	outputDir = filepath.Join(baseDir, "cot_output")
	checkpointFile = filepath.Join(outputDir, "checkpoint.json")
	rawCotDir = filepath.Join(outputDir, "raw_cot")
	distilledCotDir = filepath.Join(outputDir, "distilled_cot")
	techniquesDir = filepath.Join(outputDir, "techniques")

	for _, d := range []string{outputDir, rawCotDir, distilledCotDir, techniquesDir} {
		os.MkdirAll(d, 0755)
	}

	// Setup logging
	logFile, err := os.OpenFile(filepath.Join(outputDir, "generation.log"), os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0644)
	if err == nil {
		log.SetOutput(io.MultiWriter(os.Stdout, logFile))
	}
	log.SetFlags(log.Ldate | log.Ltime)

	// HTTP client with connection pooling
	transport := &http.Transport{
		MaxIdleConns:        20,
		MaxIdleConnsPerHost: 20,
		IdleConnTimeout:     90 * time.Second,
	}
	httpClient = &http.Client{Timeout: apiTimeout, Transport: transport}

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
	log.Printf("Using model: %s @ %s", apiModel, apiEndpoint)

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
