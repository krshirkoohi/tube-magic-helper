# Tube Magic Helper

Open source lightweight console YouTube AI assistant similar to Tube Magic.

## Features

- Search for YouTube videos from the command line
- Analyze YouTube videos with AI assistance
- Get video summaries and insights
- View video metadata and comments

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/tube-magic-helper.git
   cd tube-magic-helper
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.env` file with your API keys (see `.env.example` for reference):
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

## API Keys

This application requires two API keys:

1. **YouTube API Key**: Get it from the [Google Cloud Console](https://console.cloud.google.com/)
2. **OpenAI API Key**: Get it from the [OpenAI Platform](https://platform.openai.com/account/api-keys)

## Usage

### Analyze a YouTube video

```bash
python tube_magic_helper.py analyze https://www.youtube.com/watch?v=VIDEO_ID
```

Options:
- `--comments` / `-c`: Include comments in the analysis
- `--summary` / `-s`: Generate video summary (default: enabled)
- `--insights` / `-i`: Generate insights about the video (default: enabled)

### Search for YouTube videos

```bash
python tube_magic_helper.py search "your search query"
```

Options:
- `--max` / `-m`: Maximum number of results (default: 5)

## License

MIT
