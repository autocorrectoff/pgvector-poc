package main

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"log"
	"net/http"
	"os"
	"strings"
	"sync"
	"sync/atomic"
	"time"
)

type OllamaRequest struct {
	Model  string `json:"model"`
	Prompt string `json:"prompt"`
}

type OllamaResponse struct {
	Embedding []float32 `json:"embedding"`
}

type Entry struct {
	Word      string    `json:"word"`
	Embedding []float32 `json:"embedding"`
}

const (
	ollamaURL     = "http://localhost:11434/api/embeddings"
	modelName     = "nomic-embed-text"
	maxGoroutines = 50 // control concurrency level
)

func fetchEmbedding(
	word string,
	wg *sync.WaitGroup,
	sem chan struct{},
	results *sync.Map,
	progress *int32,
	total int32,
) {
	defer wg.Done()
	sem <- struct{}{}        // acquire slot
	defer func() { <-sem }() // release slot

	payload := OllamaRequest{
		Model:  modelName,
		Prompt: word,
	}
	data, _ := json.Marshal(payload)

	resp, err := http.Post(ollamaURL, "application/json", bytes.NewBuffer(data))
	if err != nil {
		log.Printf("❌ Request failed for word '%s': %v", word, err)
	} else {
		defer resp.Body.Close()
		if resp.StatusCode == http.StatusOK {
			var res OllamaResponse
			if err := json.NewDecoder(resp.Body).Decode(&res); err == nil && len(res.Embedding) > 0 {
				results.Store(word, res.Embedding)
			}
		} else {
			body, _ := io.ReadAll(resp.Body)
			log.Printf("❌ Ollama error for '%s': %s, %s", word, resp.Status, body)
		}
	}

	// update progress
	current := atomic.AddInt32(progress, 1)
	fmt.Printf("\rProgress: %d/%d", current, total)
}

func CreateEmbeddings() {
	start := time.Now()

	// Step 1: Read words from file
	file, err := os.Open("wordDictionary.txt")
	if err != nil {
		log.Fatalf("Failed to open file: %v", err)
	}
	defer file.Close()

	var words []string
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		word := strings.TrimSpace(scanner.Text())
		if word != "" {
			words = append(words, word)
		}
	}
	if err := scanner.Err(); err != nil {
		log.Fatalf("Error reading file: %v", err)
	}
	totalWords := int32(len(words))

	// Step 2: Concurrent embedding requests with progress
	var wg sync.WaitGroup
	sem := make(chan struct{}, maxGoroutines)
	results := &sync.Map{}
	var progress int32

	fmt.Println("🚀 Generating embeddings...")

	for _, word := range words {
		wg.Add(1)
		go fetchEmbedding(word, &wg, sem, results, &progress, totalWords)
	}

	wg.Wait()
	fmt.Print("\n✅ All embeddings processed.\n")

	// Step 3: Write results to JSONL file
	outFile, err := os.Create("embeddings.jsonl")
	if err != nil {
		log.Fatalf("Failed to create output file: %v", err)
	}
	defer outFile.Close()

	encoder := json.NewEncoder(outFile)
	count := 0

	results.Range(func(key, value any) bool {
		entry := Entry{
			Word:      key.(string),
			Embedding: value.([]float32),
		}
		if err := encoder.Encode(entry); err != nil {
			log.Printf("Failed to write entry for '%s': %v", entry.Word, err)
			return true
		}
		count++
		return true
	})

	fmt.Printf("📁 %d embeddings written to embeddings.jsonl in %v\n", count, time.Since(start))
}
