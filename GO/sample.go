package main

import (
	"fmt"
	"io"
	"log"
	"net/http"
	"time"
)

func main() {
	//resp, err := http.Get("https://go.dev")
	resp, err := http.Get("https://httpbin.org/get")
	if err != nil {
		log.Fatalln(err)
	}

	start := time.Now()
	//We Read the response body on the line below.
	body, err := io.ReadAll(resp.Body)
	if err != nil {
		log.Fatalln(err)
	}

	log.Println("Everything:", time.Since(start))
	fmt.Printf("client: status code: %d\n", resp.StatusCode)
	//Convert the body to type string
	sb := string(body)
	log.Printf(sb)
}
