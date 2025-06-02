#!/usr/bin/env python3
"""
Tube Magic Helper - Lightweight console YouTube AI assistant
"""
import os
import sys
import json
import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich import print as rprint
from dotenv import load_dotenv
import openai
from pytube import YouTube
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import datetime
from pathlib import Path
import requests
import math
import urllib.parse

# Load environment variables
load_dotenv()

# Initialize console
console = Console()
app = typer.Typer(help="Tube Magic Helper - YouTube AI Assistant")

# Configuration
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize OpenAI client
openai_client = None
if OPENAI_API_KEY:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Initialize YouTube API client
youtube_client = None
if YOUTUBE_API_KEY:
    youtube_client = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

# Data directory to store generated content
data_dir = Path(__file__).parent / "data"
data_dir.mkdir(exist_ok=True)

def check_api_keys():
    """Check if API keys are set and provide guidance if not."""
    missing_keys = []
    
    if not YOUTUBE_API_KEY:
        missing_keys.append("YOUTUBE_API_KEY")
    
    if not OPENAI_API_KEY:
        missing_keys.append("OPENAI_API_KEY")
    
    if missing_keys:
        console.print(Panel(
            f"[bold red]Missing API keys: {', '.join(missing_keys)}[/bold red]\n\n"
            "Please create a .env file in the project directory with the following content:\n\n"
            "YOUTUBE_API_KEY=your_youtube_api_key\n"
            "OPENAI_API_KEY=your_openai_api_key\n\n"
            "You can get a YouTube API key from the Google Cloud Console.\n"
            "You can get an OpenAI API key from https://platform.openai.com/account/api-keys",
            title="API Keys Required",
            border_style="red"
        ))
        return False
    return True


def get_video_id(url):
    """Extract video ID from YouTube URL."""
    try:
        if "youtu.be" in url:
            return url.split("/")[-1].split("?")[0]
        elif "youtube.com" in url:
            if "v=" in url:
                return url.split("v=")[1].split("&")[0]
        return url  # Assume it's already a video ID if not matching patterns
    except Exception as e:
        console.print(f"[red]Error parsing YouTube URL: {e}[/red]")
        return None


def get_video_info(video_id):
    """Get video information using YouTube API."""
    try:
        if not youtube_client:
            console.print("[yellow]YouTube API key not set. Using limited pytube functionality.[/yellow]")
            yt = YouTube(f"https://www.youtube.com/watch?v={video_id}")
            return {
                "title": yt.title,
                "channel": yt.author,
                "description": yt.description,
                "views": yt.views,
                "publish_date": str(yt.publish_date),
                "url": f"https://www.youtube.com/watch?v={video_id}"
            }
        
        request = youtube_client.videos().list(
            part="snippet,contentDetails,statistics",
            id=video_id
        )
        response = request.execute()
        
        if not response["items"]:
            console.print(f"[red]Video not found: {video_id}[/red]")
            return None
        
        video = response["items"][0]
        snippet = video["snippet"]
        statistics = video["statistics"]
        
        return {
            "title": snippet["title"],
            "channel": snippet["channelTitle"],
            "description": snippet["description"],
            "views": statistics.get("viewCount", "N/A"),
            "likes": statistics.get("likeCount", "N/A"),
            "comments": statistics.get("commentCount", "N/A"),
            "publish_date": snippet["publishedAt"],
            "url": f"https://www.youtube.com/watch?v={video_id}"
        }
    except Exception as e:
        console.print(f"[red]Error fetching video info: {e}[/red]")
        return None


def get_video_comments(video_id, max_results=10):
    """Get video comments using YouTube API."""
    if not youtube_client:
        console.print("[yellow]YouTube API key required to fetch comments.[/yellow]")
        return []
    
    try:
        request = youtube_client.commentThreads().list(
            part="snippet",
            videoId=video_id,
            maxResults=max_results,
            order="relevance"
        )
        response = request.execute()
        
        comments = []
        for item in response["items"]:
            comment = item["snippet"]["topLevelComment"]["snippet"]
            comments.append({
                "author": comment["authorDisplayName"],
                "text": comment["textDisplay"],
                "likes": comment["likeCount"],
                "published_at": comment["publishedAt"]
            })
        
        return comments
    except HttpError as e:
        if "commentsDisabled" in str(e):
            console.print("[yellow]Comments are disabled for this video.[/yellow]")
        else:
            console.print(f"[red]Error fetching comments: {e}[/red]")
        return []
    except Exception as e:
        console.print(f"[red]Error fetching comments: {e}[/red]")
        return []


def analyze_with_ai(content, prompt_template, max_tokens=500):
    """Analyze content using OpenAI API."""
    if not openai_client:
        console.print("[yellow]OpenAI API key not set. AI analysis unavailable.[/yellow]")
        return None
    
    try:
        prompt = prompt_template.format(content=content)
        
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        
        return response.choices[0].message.content
    except Exception as e:
        console.print(f"[red]Error during AI analysis: {e}[/red]")
        return None


def display_video_info(video_info):
    """Display video information in a rich panel."""
    if not video_info:
        return
    
    info_text = f"""
# {video_info['title']}

**Channel:** {video_info['channel']}
**Views:** {video_info['views']}
**Published:** {video_info['publish_date']}

## Description
{video_info['description'][:500]}{'...' if len(video_info['description']) > 500 else ''}
"""
    
    console.print(Panel(Markdown(info_text), title=f"Video Information", border_style="blue"))


def display_ai_analysis(analysis, title):
    """Display AI analysis in a rich panel."""
    if not analysis:
        return
    
    console.print(Panel(Markdown(analysis), title=title, border_style="green"))


def openai_chat(prompt: str, max_tokens: int = 800):
    """Simple wrapper around OpenAI chat completion."""
    if not openai_client:
        console.print("[yellow]OpenAI API key not set. AI functionality unavailable.[/yellow]")
        return None
    try:
        response = openai_client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        console.print(f"[red]Error from OpenAI: {e}[/red]")
        return None


def generate_video_ideas(niche: str, num_ideas: int = 10):
    """Generate video ideas for a given niche using AI."""
    prompt = (
        f"Provide {num_ideas} unique YouTube video ideas for a channel about '{niche}'. "
        "Return them as a numbered list."
    )
    ideas_text = openai_chat(prompt)
    if not ideas_text:
        return []
    ideas = []
    for line in ideas_text.splitlines():
        line = line.strip()
        if not line:
            continue
        # Remove leading numbering if present
        if line[0].isdigit():
            parts = line.split(".", 1)
            if len(parts) == 2:
                idea = parts[1].strip()
            else:
                idea = line
        else:
            idea = line
        ideas.append(idea)
    return ideas


def generate_script(title: str, duration_minutes: int = 5):
    """Generate a video script for the given title."""
    prompt = (
        f"Write a detailed YouTube script (~{duration_minutes} minutes) for a video titled '{title}'. "
        "Include an engaging introduction, well-structured main points, and a strong conclusion."
    )
    return openai_chat(prompt, max_tokens=1200)


def optimize_metadata(topic: str, script: str | None = None):
    """Generate SEO-optimised metadata (title, description, tags)."""
    prompt = (
        f"Generate an SEO-optimised YouTube title, a 200-word description, and exactly 15 one-word keyword tags for a video about '{topic}'. "
        "Return the result strictly as valid JSON with keys: title, description, tags."
    )
    if script:
        prompt += "\n\nHere is the draft script for additional context:\n" + script[:3000]
    meta_text = openai_chat(prompt, max_tokens=800)
    if not meta_text:
        return None
    try:
        return json.loads(meta_text)
    except Exception:
        console.print("[yellow]Unable to parse JSON metadata. Showing raw output instead.[/yellow]")
        return {"raw": meta_text}


# --- Keyword research helpers ---

def _keyword_suggestions(keyword: str, max_suggestions: int = 5) -> list[str]:
    """Return related long-tail keywords using YouTube autosuggest."""
    try:
        url = (
            "https://suggestqueries.google.com/complete/search?" +
            urllib.parse.urlencode({"client": "firefox", "ds": "yt", "q": keyword})
        )
        resp = requests.get(url, timeout=4)
        data = resp.json()
        # data[1] is list of suggestions
        return [s.lower() for s in data[1][:max_suggestions]]
    except Exception:
        return []


def _keyword_stats(keyword: str, max_videos: int = 50):
    """Compute stats for a keyword using up-to-`max_videos` top-viewed results."""
    if not youtube_client:
        return None

    # ── get total result count (competition) ──
    try:
        meta_resp = youtube_client.search().list(
            q=keyword,
            part="id",
            type="video",
            maxResults=1,
        ).execute()
        competition = meta_resp.get("pageInfo", {}).get("totalResults", 0)
    except Exception as e:
        console.print(f"[red]Keyword meta error for '{keyword}': {e}[/red]")
        return None

    # ── collect IDs ordered by viewCount ──
    video_ids = []
    page_token = None
    while len(video_ids) < max_videos:
        try:
            search_page = youtube_client.search().list(
                q=keyword,
                part="id",
                type="video",
                order="viewCount",
                maxResults=min(50, max_videos - len(video_ids)),
                pageToken=page_token,
            ).execute()
        except Exception as e:
            console.print(f"[red]Search error for '{keyword}': {e}[/red]")
            break
        video_ids.extend([it["id"]["videoId"] for it in search_page.get("items", [])])
        page_token = search_page.get("nextPageToken")
        if not page_token:
            break

    if not video_ids:
        return None

    # ── total & average views for collected videos ──
    total_views = 0
    for i in range(0, len(video_ids), 50):
        try:
            vids_resp = youtube_client.videos().list(
                id=",".join(video_ids[i : i + 50]),
                part="statistics",
            ).execute()
            total_views += sum(int(v["statistics"].get("viewCount", 0)) for v in vids_resp.get("items", []))
        except Exception as e:
            console.print(f"[red]Stats error for '{keyword}': {e}[/red]")

    avg_views = total_views / max(len(video_ids), 1)
    # Normalised competition (0..1). YouTube caps totalResults at 1,000,000.
    competition_norm = min(competition, 1_000_000) / 1_000_000

    # Composite score: favour high avg views & low competition.
    score = avg_views / (1 + competition_norm * 1_000_000)

    # Human-friendly competition label
    competition_display = f"{competition:,}" if competition < 1_000_000 else "1M+"

    return {
        "keyword": keyword,
        "views": total_views,
        "avg_views": round(avg_views),
        "competition": competition_display,
        "competition_norm": competition_norm,
        "score": round(score, 2),
    }


def keyword_research(seed_keywords: list[str], limit: int = 10):
    """Keyword research with autosuggest expansion and improved scoring."""
    if not youtube_client:
        console.print("[red]YouTube API key required for keyword research.[/red]")
        return []

    all_keywords: set[str] = {kw.lower() for kw in seed_keywords}
    for kw in seed_keywords:
        all_keywords.update(_keyword_suggestions(kw))

    results = []
    for kw in all_keywords:
        stats = _keyword_stats(kw)
        if stats:
            results.append(stats)

    results.sort(key=lambda x: x["score"], reverse=True)
    # Trim and return
    return results[:limit]


@app.command("analyze")
def analyze_video(
    url: str = typer.Argument(..., help="YouTube video URL or ID"),
    comments: bool = typer.Option(False, "--comments", "-c", help="Include comments in analysis"),
    summary: bool = typer.Option(True, "--summary", "-s", help="Generate video summary"),
    insights: bool = typer.Option(True, "--insights", "-i", help="Generate insights about the video")
):
    """Analyze a YouTube video with AI assistance."""
    if not check_api_keys():
        return
    
    with console.status("[bold blue]Processing video...[/bold blue]"):
        video_id = get_video_id(url)
        if not video_id:
            console.print("[red]Invalid YouTube URL or ID[/red]")
            return
        
        video_info = get_video_info(video_id)
        if not video_info:
            return
    
    display_video_info(video_info)
    
    # Get comments if requested
    video_comments = []
    if comments:
        with console.status("[bold blue]Fetching comments...[/bold blue]"):
            video_comments = get_video_comments(video_id)
    
    # Prepare content for AI analysis
    content_for_analysis = f"""
Title: {video_info['title']}
Channel: {video_info['channel']}
Description: {video_info['description']}
"""
    
    if comments and video_comments:
        content_for_analysis += "\nTop Comments:\n"
        for i, comment in enumerate(video_comments[:5], 1):
            content_for_analysis += f"{i}. {comment['author']}: {comment['text']}\n"
    
    # Generate summary if requested
    if summary:
        with console.status("[bold green]Generating summary...[/bold green]"):
            summary_prompt = "Provide a concise summary of this YouTube video based on the following information:\n\n{content}"
            summary_text = analyze_with_ai(content_for_analysis, summary_prompt)
            display_ai_analysis(summary_text, "Video Summary")
    
    # Generate insights if requested
    if insights:
        with console.status("[bold green]Generating insights...[/bold green]"):
            insights_prompt = "Provide interesting insights and key takeaways from this YouTube video based on the following information:\n\n{content}"
            insights_text = analyze_with_ai(content_for_analysis, insights_prompt)
            display_ai_analysis(insights_text, "Video Insights")


@app.command("search")
def search_videos(
    query: str = typer.Argument(..., help="Search query"),
    max_results: int = typer.Option(5, "--max", "-m", help="Maximum number of results")
):
    """Search for YouTube videos."""
    if not YOUTUBE_API_KEY:
        console.print("[red]YouTube API key required for search functionality.[/red]")
        return
    
    with console.status(f"[bold blue]Searching for '{query}'...[/bold blue]"):
        try:
            request = youtube_client.search().list(
                q=query,
                part="snippet",
                maxResults=max_results,
                type="video"
            )
            response = request.execute()
            
            if not response["items"]:
                console.print("[yellow]No videos found for your query.[/yellow]")
                return
            
            console.print(Panel(f"[bold]Search Results for '{query}'[/bold]", border_style="blue"))
            
            for i, item in enumerate(response["items"], 1):
                video_id = item["id"]["videoId"]
                snippet = item["snippet"]
                
                console.print(f"[bold blue]{i}.[/bold blue] [bold]{snippet['title']}[/bold]")
                console.print(f"   Channel: {snippet['channelTitle']}")
                console.print(f"   Published: {snippet['publishedAt']}")
                console.print(f"   URL: https://www.youtube.com/watch?v={video_id}")
                console.print()
            
        except Exception as e:
            console.print(f"[red]Error during search: {e}[/red]")


@app.command("ideas")
def ideas_cmd(
    niche: str = typer.Argument(..., help="Niche/topic to generate video ideas for"),
    num: int = typer.Option(10, "--num", "-n", help="Number of ideas to generate")
):
    """Generate video ideas for a niche."""
    if not check_api_keys():
        return
    console.status("[bold blue]Generating ideas...[/bold blue]")
    ideas = generate_video_ideas(niche, num)
    if not ideas:
        return
    ideas_panel = "\n".join(f"{i+1}. {idea}" for i, idea in enumerate(ideas))
    display_ai_analysis(ideas_panel, f"Video Ideas for {niche}")
    save_path = data_dir / f"ideas_{niche.replace(' ', '_')}.json"
    with open(save_path, "w") as f:
        json.dump(ideas, f, indent=2)
    console.print(f"[green]Ideas saved to {save_path}[/green]")


@app.command("script")
def script_cmd(
    title: str = typer.Argument(..., help="Title or topic of the video"),
    minutes: int = typer.Option(5, "--minutes", "-m", help="Approximate length in minutes")
):
    """Generate a full YouTube video script."""
    if not check_api_keys():
        return
    console.status("[bold blue]Generating script...[/bold blue]")
    script_text = generate_script(title, minutes)
    if not script_text:
        return
    display_ai_analysis(script_text, "Generated Script")
    scripts_dir = data_dir / "scripts"
    scripts_dir.mkdir(exist_ok=True)
    with open(scripts_dir / f"{title.replace(' ', '_')}.txt", "w") as f:
        f.write(script_text)


@app.command("optimize")
def optimize_cmd(
    topic: str = typer.Argument(..., help="Video topic/title"),
    script_file: str = typer.Option(None, "--script-file", "-s", help="Path to a script file for context")
):
    """Generate SEO-optimised metadata for a video."""
    if not check_api_keys():
        return
    script_content = None
    if script_file and os.path.exists(script_file):
        script_content = Path(script_file).read_text()
    console.status("[bold blue]Generating metadata...[/bold blue]")
    meta = optimize_metadata(topic, script_content)
    if not meta:
        return
    if "raw" in meta:
        display_ai_analysis(meta["raw"], "Metadata (raw)")
    else:
        tags_line = ", ".join(meta.get("tags", []))
        meta_text = (
            f"**Title:** {meta['title']}\n\n"
            f"**Description:**\n{meta['description']}\n\n"
            f"**Tags:** {tags_line}"
        )
        display_ai_analysis(meta_text, "Optimised Metadata")


@app.command("research")
def research_cmd(
    keywords: str = typer.Argument(..., help="Comma-separated seed keywords"),
    limit: int = typer.Option(10, "--limit", "-l", help="Number of top keywords to return")
):
    """Perform keyword research based on seed keywords."""
    if not check_api_keys():
        return
    kw_list = [k.strip() for k in keywords.split(",") if k.strip()]
    console.status("[bold blue]Researching keywords...[/bold blue]")
    results = keyword_research(kw_list, limit)
    if not results:
        return
    table_lines = [
        f"{r['keyword']}: Views={r['views']}, AvgViews={r['avg_views']}, Competition={r['competition']}, Score={r['score']}" for r in results
    ]
    display_ai_analysis("\n".join(table_lines), "Keyword Research Results")


@app.callback()
def main():
    """Tube Magic Helper - Lightweight console YouTube AI assistant."""
    console.print(Panel.fit(
        "[bold blue]Tube Magic Helper[/bold blue]\n"
        "[italic]Lightweight console YouTube AI assistant[/italic]",
        border_style="blue"
    ))


if __name__ == "__main__":
    app()
