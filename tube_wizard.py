#!/usr/bin/env python3
"""
Tube Wizard - Lightweight console YouTube AI assistant
"""
import os
import sys
import json
import typer
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown
from rich.table import Table
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
import re
from collections import defaultdict

# Load environment variables
load_dotenv()

# Initialize console
console = Console()
app = typer.Typer(help="Tube Wizard - YouTube AI Assistant")

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

def _keyword_suggestions(keyword: str, max_suggestions: int = 10) -> list[str]:
    """Return related long-tail keywords using YouTube autosuggest.
    
    Args:
        keyword: The seed keyword to get suggestions for
        max_suggestions: Maximum number of suggestions to return
        
    Returns:
        List of suggested keywords (lowercase)
    """
    try:
        url = (
            "https://suggestqueries.google.com/complete/search?" +
            urllib.parse.urlencode({"client": "firefox", "ds": "yt", "q": keyword})
        )
        resp = requests.get(url, timeout=4)
        data = resp.json()
        # data[1] is list of suggestions, remove duplicates
        seen = set()
        suggestions = []
        for s in data[1]:
            s_lower = s.lower()
            if s_lower not in seen and s_lower != keyword.lower():
                seen.add(s_lower)
                suggestions.append(s_lower)
                if len(suggestions) >= max_suggestions:
                    break
        return suggestions
    except Exception as e:
        console.print(f"[yellow]Warning: Could not get suggestions for '{keyword}': {e}[/yellow]")
        return []


def _parse_duration(duration_str):
    """Parse ISO 8601 duration format (PT1H2M3S) to minutes."""
    duration_regex = re.compile(r'PT(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?')
    match = duration_regex.match(duration_str)
    if not match:
        return 0
    
    parts = match.groupdict()
    hours = int(parts.get('hours', 0) or 0)
    minutes = int(parts.get('minutes', 0) or 0)
    seconds = int(parts.get('seconds', 0) or 0)
    
    return hours * 60 + minutes + seconds / 60

def _categorize_keyword(keyword, channel_data, category_data):
    """Categorize a keyword as channel-specific, category-specific, or video-specific."""
    if keyword.lower() in channel_data:
        return "channel"
    elif keyword.lower() in category_data:
        return "category"
    else:
        return "video"

def _keyword_stats(keyword: str, max_videos: int = 50):
    """Compute stats for a keyword using up-to-`max_videos` top-viewed results.
    
    Implements the Magic Score formula: M = V/C where:
    - V = Search volume (views)
    - C = Competition level
    """
    if not youtube_client:
        return None

    # Get total result count (competition)
    try:
        meta_resp = youtube_client.search().list(
            q=keyword,
            part="id,snippet",
            type="video",
            maxResults=1,
        ).execute()
        competition = meta_resp.get("pageInfo", {}).get("totalResults", 0)
        
        # Get channel and category info for categorization
        channel_info = None
        category_id = None
        if meta_resp.get("items"):
            channel_info = meta_resp["items"][0]["snippet"].get("channelTitle")
            video_id = meta_resp["items"][0]["id"].get("videoId")
            if video_id:
                try:
                    video_resp = youtube_client.videos().list(
                        id=video_id,
                        part="snippet"
                    ).execute()
                    if video_resp.get("items"):
                        category_id = video_resp["items"][0]["snippet"].get("categoryId")
                except Exception:
                    pass
            
    except Exception as e:
        console.print(f"[red]Keyword meta error for '{keyword}': {e}[/red]")
        return None

    # Collect IDs ordered by viewCount
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

    # Calculate total & average views, likes, comments for collected videos
    total_views = 0
    total_likes = 0
    total_comments = 0
    total_duration = 0
    for i in range(0, len(video_ids), 50):
        try:
            vids_resp = youtube_client.videos().list(
                id=",".join(video_ids[i : i + 50]),
                part="statistics,contentDetails",
            ).execute()
            
            for v in vids_resp.get("items", []):
                stats = v.get("statistics", {})
                total_views += int(stats.get("viewCount", 0))
                total_likes += int(stats.get("likeCount", 0))
                total_comments += int(stats.get("commentCount", 0))
                
                # Calculate duration
                duration_str = v.get("contentDetails", {}).get("duration", "PT0M0S")
                duration_minutes = _parse_duration(duration_str)
                total_duration += duration_minutes
                
        except Exception as e:
            console.print(f"[red]Stats error for '{keyword}': {e}[/red]")

    video_count = len(video_ids)
    avg_views = total_views / max(video_count, 1)
    avg_likes = total_likes / max(video_count, 1)
    avg_comments = total_comments / max(video_count, 1)
    avg_duration = total_duration / max(video_count, 1)
    
    # Calculate engagement rate (likes + comments per view)
    engagement_rate = ((total_likes + total_comments) / total_views * 100) if total_views > 0 else 0
    
    # Normalize competition (0..1). YouTube caps totalResults at 1,000,000.
    competition_norm = min(competition, 1_000_000) / 1_000_000

    # Calculate Magic Score (M = V/C)
    magic_score = avg_views / (competition_norm * 1_000_000 + 1)

    # Human-friendly competition label
    if competition < 1000:
        competition_display = str(competition)
    elif competition < 1_000_000:
        competition_display = f"{competition/1000:.1f}K"
    else:
        competition_display = f"{competition/1_000_000:.1f}M"
    
    # Determine keyword type
    keyword_type = "video"
    if channel_info:
        keyword_type = "channel"
    elif category_id:
        keyword_type = "category"

    return {
        "keyword": keyword,
        "views": total_views,
        "avg_views": round(avg_views),
        "avg_likes": round(avg_likes),
        "avg_comments": round(avg_comments),
        "avg_duration": round(avg_duration, 1),
        "engagement_rate": round(engagement_rate, 2),
        "competition": competition_display,
        "competition_raw": competition,
        "competition_norm": round(competition_norm, 3),
        "magic_score": round(magic_score, 2),
        "keyword_type": keyword_type,
        "channel_info": channel_info,
        "category_id": category_id
    }


def keyword_research(seed_keywords: list[str], limit: int = 10, max_suggestions: int = 5):
    """Keyword research with autosuggest expansion and Magic Score calculation.
    
    Implements the 'secret keyword process' with:
    - Seed keyword expansion via YouTube autosuggest
    - Magic Score calculation (M = V/C)
    - Keyword categorization (channel, category, video specific)
    
    Args:
        seed_keywords: Initial keywords to research
        limit: Maximum number of results to return
        max_suggestions: Maximum suggestions to get per seed keyword
        
    Returns:
        Dictionary with all keywords and categorized results
    """
    if not youtube_client:
        console.print("[red]YouTube API key required for keyword research.[/red]")
        return {}

    # Step 1: Collect all keywords (seeds + suggestions)
    all_keywords = set()
    for kw in seed_keywords:
        kw_lower = kw.lower().strip()
        all_keywords.add(kw_lower)
        suggestions = _keyword_suggestions(kw_lower, max_suggestions)
        all_keywords.update(suggestions)

    console.print(f"[green]Found {len(all_keywords)} keywords to analyze[/green]")
    
    # Step 2: Analyze each keyword
    results = []
    with console.status("[bold blue]Analyzing keywords...[/bold blue]") as status:
        for i, kw in enumerate(all_keywords, 1):
            status.update(f"[bold blue]Analyzing keywords... [{i}/{len(all_keywords)}][/bold blue]")
            stats = _keyword_stats(kw)
            if stats:
                results.append(stats)
    
    # Step 3: Sort by Magic Score
    results.sort(key=lambda x: x["magic_score"], reverse=True)
    
    # Step 4: Categorize keywords
    categorized = {
        "channel_specific": [],
        "category_specific": {},
        "video_specific": []
    }
    
    for kw_data in results:
        if kw_data["keyword_type"] == "channel" and kw_data["channel_info"]:
            categorized["channel_specific"].append(kw_data)
        elif kw_data["keyword_type"] == "category" and kw_data["category_id"]:
            category_id = kw_data["category_id"]
            if category_id not in categorized["category_specific"]:
                categorized["category_specific"][category_id] = []
            categorized["category_specific"][category_id].append(kw_data)
        else:
            categorized["video_specific"].append(kw_data)
    
    # Return all results and categorized data
    return {
        "all_keywords": results[:limit],
        "categorized": categorized,
        "stats": {
            "total": len(results),
            "channel_specific": len(categorized["channel_specific"]),
            "category_specific": sum(len(v) for v in categorized["category_specific"].values()),
            "video_specific": len(categorized["video_specific"])
        }
    }


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
    title: str = typer.Argument(..., help="Current video title"),
    description: str = typer.Option(None, "--desc", "-d", help="Current video description"),
    tags: str = typer.Option(None, "--tags", "-t", help="Current video tags (comma-separated)"),
    niche: str = typer.Option(None, "--niche", "-n", help="Video niche/topic")
):
    """Optimize video metadata (title, description, tags) for SEO."""
    if not check_api_keys():
        return
    
    # Prepare current metadata
    current_metadata = {
        "title": title,
        "description": description or "",
        "tags": tags.split(",") if tags else []
    }
    
    with console.status("[bold blue]Optimizing metadata...[/bold blue]"):
        optimized = optimize_metadata(current_metadata, niche)
    
    if not optimized:
        return
    
    # Display optimized metadata
    console.print(Panel("[bold]Optimized Video Metadata[/bold]", border_style="green"))
    
    console.print("[bold blue]Title:[/bold blue]")
    console.print(Markdown(f"# {optimized['title']}\n"))
    
    console.print("[bold blue]Description:[/bold blue]")
    console.print(Markdown(optimized['description']))
    
    console.print("[bold blue]Tags:[/bold blue]")
    tags_text = ", ".join(optimized['tags'])
    console.print(f"[green]{tags_text}[/green]")
    
    # Save optimized metadata to file
    metadata_dir = data_dir / "metadata"
    metadata_dir.mkdir(exist_ok=True)
    
    filename = f"optimized_{title[:30].replace(' ', '_')}.json"
    with open(metadata_dir / filename, "w") as f:
        json.dump(optimized, f, indent=2)
    
    console.print(f"\n[green]Optimized metadata saved to {metadata_dir / filename}[/green]")


@app.command("keywords")
def keywords_cmd(
    keywords: list[str] = typer.Argument(..., help="Seed keywords to research"),
    limit: int = typer.Option(10, "--limit", "-l", help="Maximum number of results"),
    suggestions: int = typer.Option(5, "--suggestions", "-s", help="Maximum suggestions per keyword"),
    export: bool = typer.Option(False, "--export", "-e", help="Export results to files"),
    format: str = typer.Option("both", "--format", "-f", help="Export format: txt, csv, or both")
):
    """Research keywords with YouTube autosuggest and Magic Score."""
    if not YOUTUBE_API_KEY:
        console.print("[red]YouTube API key required for keyword research.[/red]")
        return
    
    with console.status(f"[bold blue]Researching {len(keywords)} seed keywords...[/bold blue]"):
        results = keyword_research(keywords, limit, suggestions)
    
    if not results or not results.get("all_keywords"):
        console.print("[yellow]No keyword results found.[/yellow]")
        return
    
    # Display results in a table
    table = Table(title=f"Keyword Research Results (Top {limit})")
    table.add_column("Keyword", style="cyan")
    table.add_column("Magic Score", justify="right", style="green")
    table.add_column("Avg Views", justify="right")
    table.add_column("Competition", justify="right")
    table.add_column("Type", style="magenta")
    
    for kw in results["all_keywords"]:
        table.add_row(
            kw["keyword"],
            str(kw["magic_score"]),
            f"{kw['avg_views']:,}",
            kw["competition"],
            kw["keyword_type"]
        )
    
    console.print(table)
    
    # Display stats
    stats = results["stats"]
    console.print(f"\n[bold]Total Keywords:[/bold] {stats['total']}")
    console.print(f"[bold]Channel-specific:[/bold] {stats['channel_specific']}")
    console.print(f"[bold]Category-specific:[/bold] {stats['category_specific']}")
    console.print(f"[bold]Video-specific:[/bold] {stats['video_specific']}")
    
    # Export if requested
    if export:
        keywords_dir = data_dir / "keywords"
        keywords_dir.mkdir(exist_ok=True)
        
        # Create filename from seed keywords
        seed_text = "_".join(k.replace(" ", "") for k in keywords[:2])
        if len(keywords) > 2:
            seed_text += "_etc"
        
        if format in ["txt", "both"]:
            # Export to text file
            with open(keywords_dir / f"{seed_text}.txt", "w") as f:
                f.write(f"Tube Wizard Keyword Research Results\n")
                f.write(f"Seed Keywords: {', '.join(keywords)}\n\n")
                
                f.write("Top Keywords by Magic Score:\n")
                for i, kw in enumerate(results["all_keywords"], 1):
                    f.write(f"{i}. {kw['keyword']} - Score: {kw['magic_score']} - Views: {kw['avg_views']} - Competition: {kw['competition']}\n")
                
                f.write("\nStats:\n")
                for k, v in stats.items():
                    f.write(f"{k.replace('_', ' ').title()}: {v}\n")
            
            console.print(f"[green]Exported to {keywords_dir / f'{seed_text}.txt'}[/green]")
        
        if format in ["csv", "both"]:
            # Export to CSV file
            with open(keywords_dir / f"{seed_text}.csv", "w", newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Keyword", "Magic Score", "Avg Views", "Competition", "Type"])
                for kw in results["all_keywords"]:
                    writer.writerow([kw["keyword"], kw["magic_score"], kw["avg_views"], kw["competition"], kw["keyword_type"]])
            
            console.print(f"[green]Exported to {keywords_dir / f'{seed_text}.csv'}[/green]")


if __name__ == "__main__":
    # Create data directory if it doesn't exist
    data_dir.mkdir(exist_ok=True)
    
    # Show welcome message
    console.print(Panel.fit(
        "[bold blue]Tube Wizard[/bold blue] - YouTube AI Assistant\n" +
        "[italic]Your AI-powered YouTube companion for content creation and optimization[/italic]",
        border_style="green"
    ))
    
    # Run the app
    app()
