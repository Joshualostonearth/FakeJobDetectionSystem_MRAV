import json
from scrapegraphai.graphs import SmartScraperGraph
#Local llm
# Local pipeline configuration pointing strictly to Ollama
graph_config = {
    "llm": {
        "model": "ollama/llama3.2",
        "temperature": 0,
        "format": "json",
        "base_url": "http://localhost:11434",
    },
    "embeddings": {
        "model": "ollama/nomic-embed-text",
        "base_url": "http://localhost:11434",
    },
    "verbose": True,
    "headless": True,
}

# Define your extraction prompt and target web source
smart_scraper_graph = SmartScraperGraph(
    prompt="Extract the official company name, primary contact phone number, and corporate office address.",
    source="https://www.google.com/about/careers/applications/",
    config=graph_config
)
#Scraping is based on prompt given 
# Run the local graph execution
result = smart_scraper_graph.run()
print("\n--- EXTRACTED JSON OUTPUT ---")
print(json.dumps(result, indent=2))