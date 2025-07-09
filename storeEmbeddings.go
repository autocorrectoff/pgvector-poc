package main

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"log"
	"os"
	"sync"

	_ "github.com/jackc/pgx/v4/stdlib"
	"github.com/jmoiron/sqlx"
)

type WordEmbedding struct {
    Word      string    `json:"word"`
    Embedding []float32 `json:"embedding"`
}

func InsertEmbeddingsFromJSONL(filePath string, db *sqlx.DB, workerCount int) error {
	file, err := os.Open(filePath)
	if err != nil {
		return fmt.Errorf("failed to open file: %w", err)
	}
	defer file.Close()

	scanner := bufio.NewScanner(file)
	jobs := make(chan WordEmbedding, 100)
	wg := sync.WaitGroup{}

	ctx := context.Background()

	// Worker pool
	for i := 0; i < workerCount; i++ {
		wg.Add(1)
		go func(workerID int) {
			defer wg.Done()
			for we := range jobs {
				_, err := db.ExecContext(ctx,
					"INSERT INTO words (word, embedding) VALUES ($1, $2::vector)",
					we.Word, pgvectorString(we.Embedding),
				)
				if err != nil {
					log.Printf("[Worker %d] failed to insert word %q: %v", workerID, we.Word, err)
				}
			}
		}(i)
	}

	// Feed jobs
	for scanner.Scan() {
		line := scanner.Bytes()
		var we WordEmbedding
		if err := json.Unmarshal(line, &we); err != nil {
			log.Printf("Failed to parse line: %v", err)
			continue
		}
		jobs <- we
	}
	close(jobs)

	wg.Wait()

	if err := scanner.Err(); err != nil {
		return fmt.Errorf("error reading file: %w", err)
	}

	return nil
}

func pgvectorString(vec []float32) string {
    str := "["
    for i, v := range vec {
        str += fmt.Sprintf("%f", v)
        if i < len(vec)-1 {
            str += ","
        }
    }
    str += "]"
    return str
}
