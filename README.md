# SiratGPT

SiratGPT is an AI chatbot designed to provide information about Islam by searching through Hadith and Quran databases. The application leverages modern AI techniques to provide accurate and detailed responses to user queries about Islamic topics.

## Features

- Search through Hadith database using MongoDB
- Extract relevant information from the Quran using semantic search
- Option for deep search with extended AI responses
- Professional, modern UI with chat interface
- Source selection (Hadith, Quran, or both)
- Conversation history tracking
- Mobile-responsive design
- Dark/light theme toggle

## Requirements

- Python 3.8+
- MongoDB running locally
- Required Python packages (see `requirements.txt`)

## Installation

1. Clone this repository
2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Ensure MongoDB is running locally at `mongodb://localhost:27017/` with the SiratGPT database and Hadiths collection
4. Ensure you have the Quran PDF file (`quran-english-translation.pdf`) in the root directory

## Quick Start

### On Windows:
Simply double-click the `start.bat` file or run it from the command line:
```
start.bat
```

### On Linux/Mac:
Make the script executable and run it:
```bash
chmod +x start.sh
./start.sh
```

The startup scripts will:
1. Start the Streamlit backend (app.py)
2. Start the Flask server (server.py)
3. Open your default browser to http://localhost:5000

## Manual Usage

If you prefer to start the services manually:

### Step 1: Start the Streamlit Backend

```bash
python -m streamlit run app.py
```

### Step 2: Start the Flask Server

```bash
python server.py
```

### Step 3: Open the Interface

Open your browser and navigate to http://localhost:5000

## Using the Interface

- **Ask Questions**: Type your questions about Islamic topics in the input field
- **Source Selection**: Choose between Hadith, Quran, or Both sources using the selector at the top
- **Deep Search**: Toggle Deep Search mode for more comprehensive answers
- **Quick Prompts**: Click on the suggestion cards for quick access to common topics
- **New Conversation**: Click the "New Conversation" button to start fresh
- **Theme Toggle**: Click the sun/moon icon to switch between light and dark themes

## Developer Information

### File Structure

- `app.py` - Original Streamlit application with core functionality
- `server.py` - Flask server that connects the frontend to the backend
- `index.html` - Single HTML file containing all UI components (HTML/CSS/JS)
- `start.bat` - Windows startup script
- `start.sh` - Linux/Mac startup script
- `requirements.txt` - List of Python dependencies

### Customization

#### Modifying the Backend

The backend functionality is in `app.py` with the connection layer in `server.py`:

1. Add new functions to `app.py` as needed
2. Update the integration section in `server.py` to call these functions
3. Modify the frontend JavaScript to send additional parameters as needed

#### Customizing the UI

The UI is contained in a single HTML file (`index.html`) with embedded CSS and JavaScript:

1. Edit the CSS variables in the `:root` section to change colors and theme
2. Modify the HTML structure to add or remove elements
3. Update the JavaScript to change functionality

## Troubleshooting

### Common Issues

- **MongoDB Connection**: Ensure MongoDB is running and accessible at localhost:27017
- **Missing Quran PDF**: Make sure the `quran-english-translation.pdf` file is in the root directory
- **Port Conflicts**: If port 5000 or 8501 is already in use, modify the port in `server.py` or use Streamlit's options

### Error Messages

- "I'm having trouble connecting to the database": Check that both app.py and server.py are running
- "Network response was not ok": Verify that the API endpoint is accessible
- Streamlit access errors: Make sure you're accessing the UI through the Flask server (port 5000), not directly through Streamlit

## Creator

SiratGPT was created by Zaid, a visionary AI engineer and entrepreneur who is passionate about fusing technology with Islamic knowledge. As the Founder of HatchUp.ai, Zaid built SiratGPT to bring deep Islamic insights to the digital world, combining modern AI techniques with timeless wisdom. 