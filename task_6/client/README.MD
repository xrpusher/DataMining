# Client Application for Auction Simulation

This project is a client application for visualizing auction simulation data. It connects to a server application and provides real-time visualization using `matplotlib` and `seaborn`.

---

## Prerequisites

- Python 3.10 or higher
- A running server application

---

## Setup Instructions

### 1. Clone the Repository

First, clone the repository and navigate to the `client` directory:

```bash
git clone <repository-url>
cd <repository-name>/client
```

### 2. Create a Virtual Environment

Create and activate a Python virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

Install the required dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

If `requirements.txt` is missing or outdated, you can generate it by running:

```bash
pip freeze > requirements.txt
```

### 4. Install or Upgrade Additional Dependencies

Ensure the following are installed and upgraded for proper functioning:

```bash
pip install --upgrade seaborn
pip install --upgrade PyQt5
```

### 5. Verify Installed Packages (Optional)

To verify all dependencies are installed correctly:

```bash
pip list
```

---

## Usage

### Run the Client

Navigate to the client directory (if not already there) and run the application:

```bash
python main.py
```

### Visualization

The client visualizes auction data using matplotlib and seaborn. Ensure the server is running and accessible. If there are connection errors, check the host and port specified in the code.

---

## Troubleshooting

### Matplotlib Backend Error

If you encounter an error related to the `Qt5Agg` backend, ensure PyQt5 is installed:

```bash
pip install PyQt5
```

Alternatively, switch to a non-GUI backend (if interactive plots are not required):

```python
import matplotlib as mpl
mpl.use('Agg')
```

### Missing Dependencies

If any library is missing, install it using:

```bash
pip install <library-name>
```

---

## Project Structure

```
client/
├── __pycache__/           # Compiled Python files
├── data/                  # Contains data files required by the client
│   ├── buckets.json
│   ├── contexts.json
│   ├── us-east4gcp_banner_120747...
│   ├── us-east4gcp_banner_com-p...
│   ├── us-east4gcp_native_591560...
│   ├── us-east4gcp_native_com-pix...
│   ├── us-east4gcp_video_74519_...
│   └── us-east4gcp_video_com-pe...
├── venv/                  # Virtual environment (not committed to Git)
├── .gitignore             # Git ignore rules
├── animate.py             # Handles visualization using matplotlib and seaborn
├── auction.py             # Core auction logic
├── client.py              # Client-side application logic
├── context.py             # Context handling for auctions
├── log.py                 # Logging setup
├── main.py                # Entry point for the client
├── simulator.py           # Simulates auction scenarios
├── Pipfile                # Pipenv dependency file
├── Pipfile.lock           # Locked dependencies
├── README.md              # Documentation
└── requirements.txt       # Python dependencies
```

---

## Notes

- Server Dependency: Ensure the server-side application is running before starting the client.
- Updating Dependencies: Update requirements.txt whenever new dependencies are added.
- `.gitignore` Usage: Use `.gitignore` to avoid committing unnecessary files like `venv/` or `.log` files.

### Example `.gitignore`

Add the following `.gitignore` file to the client directory:

```
# Virtual environment
venv/
__pycache__/

# Logs
*.log

# Python-specific ignores
*.pyc
*.pyo
```

---

## Authors

This client application is based on the repository [https://github.com/edvgha/Tail/tree/main](https://github.com/edvgha/Tail/tree/main).