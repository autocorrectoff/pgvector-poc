package main

import (
	"fmt"
	"strconv"
	"strings"

	_ "github.com/jackc/pgx/v4/stdlib"
	"github.com/jmoiron/sqlx"
)

type Words struct {
    Word      string    `json:"word"`
    ID        int       `json:"id"`
	Distance  float32   `json:"distance"`
}

func QuerySimilarWords(db *sqlx.DB, target []float32, limit int) ([]Words, error) {
	query := `
        SELECT id, word, embedding <-> $1::vector AS distance
        FROM words
        ORDER BY distance
        LIMIT $2
    `

	embeddingStr := pgvectorString(target)

	rows, err := db.Queryx(query, embeddingStr, limit)
	if err != nil {
		return nil, fmt.Errorf("query failed: %w", err)
	}
	defer rows.Close()

	var results []Words
	for rows.Next() {
		var word string
		var id int
		var distance float32

		if err := rows.Scan(&id, &word, &distance); err != nil {
			return nil, fmt.Errorf("scan failed: %w", err)
		}

		results = append(results, Words{
			Word:    word,
			Distance: distance,
			ID:   id,
		})
	}

	return results, nil
}

func parseVectorString(s string) ([]float32, error) {
	s = strings.Trim(s, "[]'")
	parts := strings.Split(s, ",")
	vec := make([]float32, 0, len(parts))
	for _, p := range parts {
		f, err := strconv.ParseFloat(strings.TrimSpace(p), 32)
		if err != nil {
			return nil, err
		}
		vec = append(vec, float32(f))
	}
	return vec, nil
}