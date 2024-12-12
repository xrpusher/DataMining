# Python Auction Server

This project implements a Python server that handles auction logic and manages probabilities, revenue calculations, and optimization using LightGBM.

---

## Features

- Modularized Python server structure.
- Uses Flask for API endpoints.
- Implements LightGBM for probability learning and optimization.
- Supports configuration via `.env` file.

---

## Requirements

- Python 3.10 or newer.
- Dependencies are listed in `requirements.txt`.

---

## Project Structure

```
server/
├── app/
│   ├── __init__.py           # Initializes the Flask app
│   ├── config.py             # Loads configurations from .env and defaults
│   ├── handlers.py           # API routes and logic for handling HTTP requests
│   ├── logger.py             # Logging setup
│   ├── main.py               # Entry point for the server
│   ├── misc.py               # Helper utilities and error handling
│   ├── model.py              # Data models for auction requests and responses
│   └── space.py              # Core logic for bucket learning and optimization
├── data/
│   └── spaces_desc.json      # Example auction spaces data
├── .env                      # Environment variables
├── .gitignore                # Ignored files for git
├── logs.txt                  # Server logs (ignored in git)
├── requirements.txt          # Project dependencies
└── README.md                 # This file
```

---

## Installation and Setup

### Clone the Repository

```bash
git clone <repository_url>
cd server
```

### Create a Virtual Environment

Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### Install Dependencies

Install the required dependencies:

```bash
pip install -r requirements.txt
```

### Configuration

The server uses environment variables for configuration. These are set in the `.env` file. Below is an example `.env` file:

```
BUFFER_SIZE=100
DISCOUNT=0.25
DESIRED_EXPLORATION_SPEED=2
LOG_LEVEL=trace
LEVEL_SIZE=5
BUCKET_SIZE=30
SPACE_DESC_FILE=""
CACHE_TTL=10
```

#### Key `.env` Variables

- **BUFFER_SIZE**: Maximum size of the bucket buffer.
- **DISCOUNT**: Discount factor for probabilities.
- **DESIRED_EXPLORATION_SPEED**: Speed for exploration.
- **LOG_LEVEL**: Logging level (trace, debug, info, error).
- **LEVEL_SIZE**: Number of auction levels.
- **BUCKET_SIZE**: Number of buckets per level.
- **SPACE_DESC_FILE**: Path to the `spaces_desc.json` file. If left blank, the server will automatically set it to `server/data/spaces_desc.json`.
- **CACHE_TTL**: Time-to-live for cache in seconds.

---

## Running the Server

Ensure the `.env` file is correctly configured.

Start the server using:

```bash
python -m app.main
```

---

## Testing the Server

### Endpoints

- **`/optimize`**: POST request to optimize auction price.
- **`/feedback`**: POST feedback about impressions.
- **`/space`**: GET request to retrieve auction data.

---

## Troubleshooting

### Common Issues

#### FileNotFoundError for `spaces_desc.json`

Ensure that the `SPACE_DESC_FILE` variable in `.env` points to the correct location of the `spaces_desc.json` file. By default, the server looks for `server/data/spaces_desc.json`.

#### Missing Dependencies

Ensure all dependencies are installed. You can install them using:

```bash
pip install -r requirements.txt
```

#### Port Already in Use

If another process is using the default port (8000), you can run the server on a different port by modifying `app.run()` in `main.py`:

```python
app.run(host="0.0.0.0", port=8080)
```

---

## Notes

- Ensure the server runs in the root directory of the `server` folder.
- Update `requirements.txt` whenever new dependencies are added.
- Use `.gitignore` to exclude unnecessary files from version control (e.g., `venv/`, `logs.txt`).

---

## Authors

This server application is a proprietary counterpart to the client application based on the repository [https://github.com/edvgha/Tail/tree/main](https://github.com/edvgha/Tail/tree/main).
