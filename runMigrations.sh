#!/bin/sh

# This script is used to run migrations on the database.
# Example usage: ./runMigrations.sh "postgres://user:password@host:5432/dbname?sslmode=disable"

#./runMigrations.sh "postgres://postgres:password@localhost:5432/vectordb?sslmode=disable"

BD_CONN_STR=$1

go install -tags 'postgres' github.com/golang-migrate/migrate/v4/cmd/migrate@v4.15.2

migrate -database $BD_CONN_STR -path db/migrations up