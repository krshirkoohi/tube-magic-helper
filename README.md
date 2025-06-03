# Tube Wizard

Open source lightweight console YouTube AI assistant for content creators.

## Features

- Search for YouTube videos from the command line
- Analyze YouTube videos with AI assistance
- Get video summaries and insights
- View video metadata and comments

## Installation

1. Clone this repository:
   ```bash
   git clone https://github.com/krshirkoohi/tube-wizard.git
   cd tube-wizard
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
python tube_wizard.py analyze https://www.youtube.com/watch?v=VIDEO_ID
```

Options:
- `--comments` / `-c`: Include comments in the analysis
- `--summary` / `-s`: Generate video summary (default: enabled)
- `--insights` / `-i`: Generate insights about the video (default: enabled)

### Search for YouTube videos

```bash
python tube_wizard.py search "your search query"
```

Options:
- `--max` / `-m`: Maximum number of results (default: 5)

### Generate video ideas

```bash
python tube_wizard.py ideas "your niche or topic"
```

Options:
- `--num` / `-n`: Number of ideas to generate (default: 10)

### Generate a video script

```bash
python tube_wizard.py script "your video title or topic"
```

Options:
- `--minutes` / `-m`: Approximate length of the video in minutes (default: 5)

### Optimize video metadata for SEO

```bash
python tube_wizard.py optimize "your video title" --desc "your description" --tags "tag1,tag2,tag3" --niche "your niche"
```

Options:
- `--desc` / `-d`: Current video description (optional)
- `--tags` / `-t`: Current video tags, comma-separated (optional)
- `--niche` / `-n`: Video niche/topic (optional)

### Research keywords

```bash
python tube_wizard.py keywords "keyword1" "keyword2" "keyword3"
```

Options:
- `--limit` / `-l`: Maximum number of results to return (default: 10)
- `--suggestions` / `-s`: Maximum suggestions per keyword (default: 5)
- `--export` / `-e`: Export results to files (default: false)
- `--format` / `-f`: Export format: txt, csv, or both (default: both)

## License

MIT
