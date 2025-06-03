"""
Tube Wizard - Streamlit App
A YouTube keyword research and content optimization tool

This app implements the 'secret keyword process' for YouTube SEO:
1. Research and expand seed keywords
2. Calculate Magic Score (M = V/C)
3. Categorize keywords (channel, category, video specific)
4. Export results for analysis
"""
import os
import sys
import json
import datetime
import math
import re
import random
import csv
import io
import urllib.parse
import requests
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from collections import defaultdict

import streamlit as st

# Streamlit page configuration - MUST be the first Streamlit command
st.set_page_config(
    page_title="Tube Wizard", 
    page_icon="🎬", 
    layout="wide",
    initial_sidebar_state="expanded"
)

import pandas as pd
import altair as alt
import plotly.express as px
from dotenv import load_dotenv
from googleapiclient.discovery import build
from googleapiclient.errors import HttpError
import openai
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

# Adjust Python path so we can import from the project root
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

# Import functions from tube_wizard module
try:
    # First try direct import
    from tube_wizard import (
        get_video_id,
        get_video_info,
        get_video_comments,
        analyze_with_ai,
        keyword_research,
        _keyword_suggestions,
        _keyword_stats,
        _parse_duration,
        _categorize_keyword,
        generate_video_ideas,
        generate_script,
        optimize_metadata
    )
    # Flag to indicate if imports were successful
    TUBE_WIZARD_IMPORTS = True
except ImportError as e:
    # Flag to indicate if imports failed
    TUBE_WIZARD_IMPORTS = False
    st.warning(f"Could not import functions from tube_wizard.py: {e}. Some features may be limited.")
    pass

# -----------------------------------------------------------------------------
# Environment / configuration
# -----------------------------------------------------------------------------
load_dotenv()

# API Keys
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Initialize API clients
youtube_client = None
if YOUTUBE_API_KEY:
    youtube_client = build("youtube", "v3", developerKey=YOUTUBE_API_KEY)

openai_client = None
if OPENAI_API_KEY:
    openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)

# Data directory
DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# -----------------------------------------------------------------------------
# Helper functions for keyword research
# -----------------------------------------------------------------------------

def check_api_keys() -> bool:
    """Check if required API keys are set and show warnings if not."""
    missing_keys = []
    
    if not YOUTUBE_API_KEY:
        missing_keys.append("YOUTUBE_API_KEY")
    
    if not OPENAI_API_KEY:
        missing_keys.append("OPENAI_API_KEY")
    
    if missing_keys:
        st.error(
            f"Missing API keys: {', '.join(missing_keys)}\n\n"
            "Please create a .env file in the project directory with the following content:\n\n"
            "YOUTUBE_API_KEY=your_youtube_api_key\n"
            "OPENAI_API_KEY=your_openai_api_key\n\n"
            "You can get a YouTube API key from the Google Cloud Console.\n"
            "You can get an OpenAI API key from https://platform.openai.com/account/api-keys"
        )
        return False
    return True


def keyword_suggestions(keyword: str, max_suggestions: int = 10) -> List[str]:
    """Return related long-tail keywords using YouTube autosuggest.
    
    Args:
        keyword: The seed keyword to get suggestions for
        max_suggestions: Maximum number of suggestions to return
        
    Returns:
        List of suggested keywords (lowercase)
    """
    # Use imported function if available
    if TUBE_WIZARD_IMPORTS:
        return _keyword_suggestions(keyword, max_suggestions)
        
    # Fallback implementation
    try:
        # YouTube autosuggest API URL
        base_url = "http://suggestqueries.google.com/complete/search"
        params = {
            "client": "youtube",
            "ds": "yt",
            "q": keyword,
            "alt": "json"
        }
        response = requests.get(base_url, params=params)
        if response.status_code != 200:
            st.warning(f"Error getting suggestions for '{keyword}': {response.status_code}")
            return []
            
        suggestions = response.json()[1]
        # Return unique, non-empty suggestions (lowercase for consistency)
        return list(set([s.lower() for s in suggestions if s.strip()]))[:max_suggestions]
    except Exception as e:
        st.warning(f"Error getting suggestions for '{keyword}': {str(e)}")
        return []


def parse_duration(duration_str: str) -> float:
    """Parse ISO 8601 duration format (PT1H2M3S) to minutes."""
    if not duration_str or not duration_str.startswith('PT'):
        return 0.0
    
    hours = re.search(r'(\d+)H', duration_str)
    minutes = re.search(r'(\d+)M', duration_str)
    seconds = re.search(r'(\d+)S', duration_str)
    
    total_minutes = 0.0
    if hours:
        total_minutes += int(hours.group(1)) * 60
    if minutes:
        total_minutes += int(minutes.group(1))
    if seconds:
        total_minutes += int(seconds.group(1)) / 60
        
    return round(total_minutes, 2)


def get_video_id(url: str) -> str:
    """Extract the video ID from a YouTube URL.
    
    Args:
        url: YouTube video URL in various possible formats
        
    Returns:
        Video ID or empty string if no match found
    """
    # Common YouTube URL patterns
    patterns = [
        r'(?:https?://)?(?:www\.)?youtube\.com/watch\?v=([\w-]+)',  # Standard watch URL
        r'(?:https?://)?(?:www\.)?youtube\.com/embed/([\w-]+)',       # Embed URL
        r'(?:https?://)?(?:www\.)?youtube\.com/v/([\w-]+)',           # Old embed URL
        r'(?:https?://)?(?:www\.)?youtube\.com/shorts/([\w-]+)',      # Shorts URL
        r'(?:https?://)?(?:www\.)?youtu\.be/([\w-]+)'                 # Short URL
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    return ""





def get_competition_label(competition_norm: float) -> str:
    """Convert normalized competition value to human-readable label."""
    if competition_norm < 0.01:
        return "Very Low"
    elif competition_norm < 0.1:
        return "Low"
    elif competition_norm < 0.3:
        return "Medium"
    elif competition_norm < 0.6:
        return "High"
    else:
        return "Very High"


def parse_duration(duration_str: str) -> float:
    """Parse YouTube API duration string (ISO 8601) to minutes.
    
    Example: 'PT1H30M15S' -> 90.25 minutes
    """
    # Use imported function if available
    if TUBE_WIZARD_IMPORTS:
        return _parse_duration(duration_str)
    
    # Fallback implementation
    match = re.search(r'PT(?:(?P<hours>\d+)H)?(?:(?P<minutes>\d+)M)?(?:(?P<seconds>\d+)S)?', duration_str)
    if not match:
        return 0.0
        
    hours = int(match.group('hours') or 0)
    minutes = int(match.group('minutes') or 0)
    seconds = int(match.group('seconds') or 0)
    
    return hours * 60 + minutes + seconds / 60


def categorize_keyword(keyword: str, channel_data: Dict, category_data: Dict) -> str:
    """Categorize a keyword as channel-specific, category-specific, or video-specific.
    
    Args:
        keyword: The keyword to categorize
        channel_data: Dictionary with channel match info
        category_data: Dictionary with category match info
        
    Returns:
        String: 'channel', 'category', or 'video'
    """
    # Use imported function if available
    if TUBE_WIZARD_IMPORTS:
        return _categorize_keyword(keyword, channel_data, category_data)
    
    # Fallback implementation
    if channel_data.get("channel_match", False):
        return "channel"
    elif category_data.get("category_match", False):
        return "category"
    else:
        return "video"


def get_competition_label(competition_norm: float) -> str:
    """Convert normalized competition value to human-readable label."""
    if competition_norm < 0.01:
        return "Very Low"
    elif competition_norm < 0.1:
        return "Low"
    elif competition_norm < 0.3:
        return "Medium"
    elif competition_norm < 0.6:
        return "High"
    else:
        return "Very High"


def keyword_stats(keyword: str, max_videos: int = 50) -> Dict[str, Any]:
    """Compute stats for a keyword using up-to-`max_videos` top-viewed results.
    
    Implements the Magic Score formula: M = V/C where:
    - V = Search volume (views)
    - C = Competition level
    """
    # Use imported function if available
    if TUBE_WIZARD_IMPORTS:
        return _keyword_stats(keyword, max_videos)
        
    # Fallback implementation
    if not youtube_client:
        st.error("YouTube API client not initialized. Check your API key.")
        return None
        
    try:
        # Step 1: Search for videos with this keyword
        meta_resp = youtube_client.search().list(
            part="id,snippet",
            q=keyword,
            type="video",
            maxResults=min(max_videos, 50),  # API limit is 50
            order="viewCount"
        ).execute()
        
        # Get competition (total results count)
        competition = meta_resp.get("pageInfo", {}).get("totalResults", 0)
        if competition == 0:
            return None
            
        # Extract video IDs for detailed stats
        video_ids = [item["id"]["videoId"] for item in meta_resp.get("items", [])]
        if not video_ids:
            return None
            
        # Step 2: Get detailed video stats
        video_resp = youtube_client.videos().list(
            part="statistics,contentDetails,snippet",
            id=",".join(video_ids)
        ).execute()
        
        # Process video data
        total_views = 0
        total_likes = 0
        total_comments = 0
        total_duration = 0.0
        channel_titles = set()
        category_ids = set()
        
        for item in video_resp.get("items", []):
            # Get statistics
            stats = item.get("statistics", {})
            views = int(stats.get("viewCount", 0))
            likes = int(stats.get("likeCount", 0))
            comments = int(stats.get("commentCount", 0))
            
            # Get duration
            duration_str = item.get("contentDetails", {}).get("duration", "PT0S")
            duration_min = parse_duration(duration_str)
            
            # Get channel and category info
            snippet = item.get("snippet", {})
            channel_title = snippet.get("channelTitle", "").lower()
            category_id = snippet.get("categoryId", "")
            
            # Add to totals
            total_views += views
            total_likes += likes
            total_comments += comments
            total_duration += duration_min
            channel_titles.add(channel_title)
            category_ids.add(category_id)
        
        # Calculate averages
        num_videos = len(video_resp.get("items", []))
        if num_videos == 0:
            return None
            
        avg_views = total_views / num_videos
        avg_likes = total_likes / num_videos
        avg_comments = total_comments / num_videos
        avg_duration = round(total_duration / num_videos, 2)
        
        # Calculate engagement rate (likes + comments per view)
        engagement_rate = round((total_likes + total_comments) / max(total_views, 1) * 100, 2)
        
        # Normalize competition (log scale to handle wide range)
        competition_norm = min(competition, 1_000_000) / 1_000_000
        
        # Calculate Magic Score (M = V/C)
        # Higher views and lower competition = higher score
        magic_score = avg_views / (competition_norm * 1_000_000 + 1)
        
        # Check for channel and category matches
        keyword_lower = keyword.lower()
        channel_match = any(channel in keyword_lower or keyword_lower in channel for channel in channel_titles)
        category_match = len(category_ids) == 1  # All videos in same category
        
        channel_data = {"channel_match": channel_match, "channels": list(channel_titles)}
        category_data = {"category_match": category_match, "categories": list(category_ids)}
        
        # Determine keyword type
        keyword_type = categorize_keyword(keyword, channel_data, category_data)
        
        # Get human-friendly competition label
        competition_display = get_competition_label(competition_norm)
        
        return {
            "keyword": keyword,
            "views": total_views,
            "avg_views": round(avg_views),
            "competition": competition_display,
            "competition_norm": competition_norm,
            "competition_raw": competition,
            "magic_score": round(magic_score, 2),
            "engagement_rate": str(engagement_rate),
            "avg_likes": round(avg_likes),
            "avg_comments": round(avg_comments),
            "avg_duration": avg_duration,
            "keyword_type": keyword_type,
            "channel_data": channel_data,
            "category_data": category_data
        }
    except HttpError as e:
        st.error(f"YouTube API error: {str(e)}")
        return None
    except Exception as e:
        st.error(f"Error analyzing keyword '{keyword}': {str(e)}")
        return None


def process_keywords(seed_keywords: List[str], max_suggestions: int = 10, limit: int = 50, use_cache: bool = True, cache_dir: Path = None) -> Dict[str, Any]:
    """Process seed keywords and return results.
    
    Args:
        seed_keywords: List of seed keywords to expand and analyze
        max_suggestions: Maximum number of suggestions to get for each seed keyword
        limit: Maximum number of results to return
        use_cache: Whether to use cached results
        cache_dir: Directory to store cache files
        
    Returns:
        Dictionary with results and metadata
    """
    # Use imported function if available
    if TUBE_WIZARD_IMPORTS and not use_cache:  # Skip cache when using imported function
        results = keyword_research(seed_keywords, limit, max_suggestions)
        return {
            "all_keywords": results.get("all_keywords", []),
            "categorized": results.get("categorized", {}),
            "stats": results.get("stats", {}),
            "timestamp": datetime.datetime.now().isoformat(),
            "seed_keywords": seed_keywords,
            "from_cache": False
        }
    try:
        # Create cache directory if it doesn't exist and caching is enabled
        if use_cache and cache_dir is not None:
            cache_dir.mkdir(exist_ok=True, parents=True)
        
        # Step 1: Expand seed keywords with YouTube suggestions
        progress_text = "Step 1/4: Expanding seed keywords with YouTube suggestions..."
        progress_bar = st.progress(0, text=progress_text)
        
        all_keywords: set[str] = {kw.lower() for kw in seed_keywords}
        total_seeds = len(seed_keywords)
        
        for i, kw in enumerate(seed_keywords):
            # Try to get suggestions from cache first
            suggestions = []
            cache_hit = False
            
            if use_cache and cache_dir is not None:
                suggestion_cache_file = cache_dir / f"suggestions_{kw.replace(' ', '_')}.json"
                if suggestion_cache_file.exists():
                    try:
                        with open(suggestion_cache_file, 'r') as f:
                            suggestions = json.load(f)
                            cache_hit = True
                            st.sidebar.info(f"Using cached suggestions for '{kw}'")
                    except Exception:
                        # If cache read fails, proceed with API call
                        cache_hit = False
            
            # If not in cache or cache disabled, get from API
            if not cache_hit:
                try:
                    suggestions = keyword_suggestions(kw, max_suggestions)
                    
                    # Save to cache if enabled
                    if use_cache and cache_dir is not None:
                        suggestion_cache_file = cache_dir / f"suggestions_{kw.replace(' ', '_')}.json"
                        with open(suggestion_cache_file, 'w') as f:
                            json.dump(suggestions, f)
                except Exception as e:
                    st.warning(f"Error getting suggestions for '{kw}': {str(e)}")
                    # Continue with other keywords
                    continue
            
            all_keywords.update(suggestions)
            progress_bar.progress((i + 1) / total_seeds, text=progress_text)
        
        # Step 2: Get stats for each keyword
        progress_text = "Step 2/4: Analyzing keyword statistics..."
        progress_bar.progress(0, text=progress_text)
        
        results = []
        all_keywords_list = list(all_keywords)
        total_keywords = len(all_keywords_list)
        
        for i, kw in enumerate(all_keywords_list[:limit]):
            # Try to get stats from cache first
            stats = None
            cache_hit = False
            
            if use_cache and cache_dir is not None:
                stats_cache_file = cache_dir / f"stats_{kw.replace(' ', '_')}.json"
                if stats_cache_file.exists():
                    try:
                        with open(stats_cache_file, 'r') as f:
                            stats = json.load(f)
                            cache_hit = True
                            st.sidebar.info(f"Using cached stats for '{kw}'")
                    except Exception:
                        # If cache read fails, proceed with API call
                        cache_hit = False
            
            # If not in cache or cache disabled, get from API
            if not cache_hit:
                try:
                    stats = keyword_stats(kw)
                    
                    # Save to cache if enabled
                    if stats and use_cache and cache_dir is not None:
                        stats_cache_file = cache_dir / f"stats_{kw.replace(' ', '_')}.json"
                        with open(stats_cache_file, 'w') as f:
                            json.dump(stats, f, default=str)
                except Exception as e:
                    if "quota" in str(e).lower():
                        # If we hit quota limit, raise the exception to be handled by caller
                        raise e
                    st.warning(f"Error getting stats for '{kw}': {str(e)}")
                    # Continue with other keywords
                    continue
            
            if stats:
                results.append(stats)
            
            progress_bar.progress((i + 1) / min(total_keywords, limit), text=progress_text)
        
        # Step 3: Calculate Magic Scores
        progress_text = "Step 3/4: Calculating Magic Scores..."
        progress_bar.progress(0, text=progress_text)
        
        for i, kw_data in enumerate(results):
            # Calculate Magic Score (M = V/C)
            views = kw_data["avg_views"]
            competition = kw_data["competition_norm"]
            
            # Avoid division by zero
            if competition == 0:
                competition = 0.0001
                
            # Calculate Magic Score
            magic_score = views / (competition * 1_000_000)
            kw_data["magic_score"] = round(magic_score, 2)
            
            # Add competition label
            kw_data["competition_label"] = get_competition_label(kw_data["competition_norm"])
            
            # Categorize keyword
            # Handle both old and new cache formats
            if "channel_data" in kw_data and "category_data" in kw_data:
                kw_data["keyword_type"] = categorize_keyword(kw_data["keyword"], kw_data["channel_data"], kw_data["category_data"])
            elif "keyword_type" not in kw_data:
                # For older cache format, create the necessary data structures
                channel_match = "channel_info" in kw_data and bool(kw_data.get("channel_info"))
                category_match = "category_id" in kw_data and bool(kw_data.get("category_id"))
                
                channel_data = {"channel_match": channel_match, "channels": [kw_data.get("channel_info", "")] if channel_match else []}
                category_data = {"category_match": category_match, "categories": [kw_data.get("category_id", "")] if category_match else []}
                
                # Store the new format data for future use
                kw_data["channel_data"] = channel_data
                kw_data["category_data"] = category_data
                
                kw_data["keyword_type"] = categorize_keyword(kw_data["keyword"], channel_data, category_data)
            
            progress_bar.progress((i + 1) / len(results), text=progress_text)
        
        # Step 4: Sort by Magic Score and categorize
        progress_text = "Step 4/4: Finalizing results..."
        progress_bar.progress(0.5, text=progress_text)
        
        # Sort by Magic Score
        results.sort(key=lambda x: x["magic_score"], reverse=True)
        
        # Categorize keywords
        categories = {
            "channel": [],
            "category": [],
            "video": []
        }
        
        for kw_data in results:
            categories[kw_data["keyword_type"]].append(kw_data)
        
        progress_bar.progress(1.0, text="Done!")
        
        return {
            "success": True,
            "all_keywords": results,
            "categories": categories,
            "stats": {
                "total": len(results),
                "channel": len(categories["channel"]),
                "category": len(categories["category"]),
                "video": len(categories["video"]),
            }
        }
    except Exception as e:
        # Handle exceptions and provide meaningful error messages
        if "quota" in str(e).lower():
            return {
                "success": False,
                "message": "YouTube API quota exceeded. Please try again later or use cached results."
            }
        else:
            return {
                "success": False,
                "message": f"Error processing keywords: {str(e)}"
            }


# -----------------------------------------------------------------------------
# Streamlit UI components
# -----------------------------------------------------------------------------

def show_keyword_results(results: Dict[str, Any]):
    """Display keyword research results in the Streamlit UI."""
    if not results.get("success", False):
        st.error(results.get("message", "An error occurred during keyword research."))
        return
    
    # Create a DataFrame for easier display
    df = pd.DataFrame(results["all_keywords"])
    
    # Summary statistics
    st.success(f"✅ Found {results['stats']['total']} high-potential keywords")
    
    # Display category counts
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Channel-specific", results["stats"]["channel"])
    with col2:
        st.metric("Category-specific", results["stats"]["category"])
    with col3:
        st.metric("Video-specific", results["stats"]["video"])
    
    # Create tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["All Keywords", "By Category", "Charts", "Export"])
    
    with tab1:
        # Display all keywords in a table
        st.subheader("All Keywords by Magic Score")
        
        # Select columns to display
        display_df = df[["keyword", "magic_score", "avg_views", "competition", 
                        "engagement_rate", "keyword_type"]].copy()
        
        # Format columns
        display_df["avg_views"] = display_df["avg_views"].apply(lambda x: f"{x:,}")
        display_df["engagement_rate"] = display_df["engagement_rate"].astype(str) + "%"
        display_df["keyword_type"] = display_df["keyword_type"].apply(lambda x: x.capitalize())
        
        # Rename columns for display
        display_df.columns = ["Keyword", "Magic Score", "Avg. Views", "Competition", 
                             "Engagement", "Type"]
        
        st.dataframe(display_df, use_container_width=True)
    
    with tab2:
        # Create tabs for each category
        cat_tab1, cat_tab2, cat_tab3 = st.tabs(["Channel Keywords", "Category Keywords", "Video Keywords"])
        
        with cat_tab1:
            if results["stats"]["channel"] > 0:
                channel_df = pd.DataFrame(results["categories"]["channel"])
                st.dataframe(
                    channel_df[["keyword", "magic_score", "avg_views", "competition"]],
                    use_container_width=True
                )
            else:
                st.info("No channel-specific keywords found.")
        
        with cat_tab2:
            if results["stats"]["category"] > 0:
                category_df = pd.DataFrame(results["categories"]["category"])
                st.dataframe(
                    category_df[["keyword", "magic_score", "avg_views", "competition"]],
                    use_container_width=True
                )
            else:
                st.info("No category-specific keywords found.")
        
        with cat_tab3:
            if results["stats"]["video"] > 0:
                video_df = pd.DataFrame(results["categories"]["video"])
                st.dataframe(
                    video_df[["keyword", "magic_score", "avg_views", "competition"]],
                    use_container_width=True
                )
            else:
                st.info("No video-specific keywords found.")
    
    with tab3:
        # Create visualizations
        st.subheader("Keyword Analysis Charts")
        
        # Magic Score by keyword (top 15)
        top_df = df.sort_values("magic_score", ascending=False).head(15)
        
        # Bar chart of Magic Scores
        st.subheader("Top 15 Keywords by Magic Score")
        fig = px.bar(
            top_df, 
            x="keyword", 
            y="magic_score",
            color="keyword_type",
            labels={"keyword": "Keyword", "magic_score": "Magic Score", "keyword_type": "Type"},
            color_discrete_map={"channel": "#2ca02c", "category": "#d62728", "video": "#1f77b4"}
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Scatter plot of views vs competition
        st.subheader("Views vs Competition")
        fig2 = px.scatter(
            df, 
            x="competition_norm", 
            y="avg_views",
            size="magic_score",
            color="keyword_type",
            hover_name="keyword",
            log_y=True,
            labels={
                "competition_norm": "Competition Level", 
                "avg_views": "Average Views",
                "magic_score": "Magic Score",
                "keyword_type": "Type"
            },
            color_discrete_map={"channel": "#2ca02c", "category": "#d62728", "video": "#1f77b4"}
        )
        st.plotly_chart(fig2, use_container_width=True)
    
    with tab4:
        # Export options
        st.subheader("Export Results")
        
        # Generate filename based on timestamp
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        filename_base = f"keyword_research_{timestamp}"
        
        # CSV export
        csv_data = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="Download CSV",
            data=csv_data,
            file_name=f"{filename_base}.csv",
            mime="text/csv",
            help="Download all keyword data as CSV for use in spreadsheets"
        )
        
        # JSON export
        json_data = json.dumps(results, default=str, indent=2)
        st.download_button(
            label="Download JSON",
            data=json_data,
            file_name=f"{filename_base}.json",
            mime="application/json",
            help="Download complete keyword research data as JSON"
        )
        
        # Show usage instructions
        with st.expander("How to use these results"):
            st.markdown("""
            ### Next Steps
            
            1. **Import the CSV file** into Google Sheets or Excel
            2. **Sort and filter keywords** by Magic Score and category
            3. **Remove irrelevant keywords** that don't fit your content
            4. **Organize into groups** for different content types:
               - Channel keywords for your overall channel strategy
               - Category keywords for playlists and series
               - Video keywords for specific videos
            5. **Use these keywords** in your video titles, descriptions, and tags
            
            Remember that the Magic Score (M = V/C) represents the balance between search volume (views) 
            and competition. Higher scores indicate better keyword opportunities.
            """)


def save_results_to_file(results, filepath):
    try:
        with open(filepath, "w") as f:
            json.dump(results, f, indent=2, default=str)
        return True
    except Exception as e:
        st.error(f"Error saving results: {str(e)}")
        return False

# -----------------------------------------------------------------------------
# Main Application UI
# -----------------------------------------------------------------------------

# Show warning if API keys are missing, but don't stop execution
if not YOUTUBE_API_KEY or not OPENAI_API_KEY:
    missing_keys = []
    if not YOUTUBE_API_KEY:
        missing_keys.append("YOUTUBE_API_KEY")
    if not OPENAI_API_KEY:
        missing_keys.append("OPENAI_API_KEY")
        
    st.warning(
        f"Missing API keys: {', '.join(missing_keys)}\n\n"
        "Some features may not work properly. Please create a .env file with your API keys."
    )

# App header
st.title("🎬 Tube Wizard")
st.subheader("YouTube AI Assistant & Keyword Research Tool")

# Sidebar navigation - Use functions already imported from tube_wizard
MODE = st.sidebar.radio(
    "Choose a function",
    [
        "Keyword Research Process",
        "Analyze Video",
        "Video Ideas",
        "Generate Script",
        "Optimize Metadata",
        "Basic Keyword Research",
    ],
)

# Sidebar configuration
st.sidebar.title("⚙️ Configuration")

# Handle different modes
if MODE == "Keyword Research Process":
    # Show app description for keyword research process
    st.markdown("""
    ### The Secret Keyword Process for YouTube SEO
    
    This tool helps you find high-potential keywords for your YouTube videos using the 'secret keyword process':
    
    1. **Research & expand** seed keywords using YouTube's autosuggest
    2. **Calculate Magic Score** (M = V/C) where V = search volume and C = competition
    3. **Categorize keywords** as channel-specific, category-specific, or video-specific
    4. **Export results** for use in your content strategy
    """)
    
    # Seed keywords input
    st.sidebar.subheader("Step 1: Enter Seed Keywords")
    seed_keywords = st.sidebar.text_area(
        "Enter seed keywords (one per line)",
        height=100,
        help="These are your starting keywords. The tool will expand these with YouTube suggestions."
    )
    
    # Research parameters
    st.sidebar.subheader("Step 2: Set Parameters")
    suggestions_per_keyword = st.sidebar.slider(
        "Suggestions per keyword", 
        min_value=5, 
        max_value=20, 
        value=10,
        help="Number of YouTube autosuggest results to get for each seed keyword"
    )
    
    max_keywords = st.sidebar.slider(
        "Maximum keywords to analyze", 
        min_value=10, 
        max_value=100, 
        value=30,  # Reduced default to help with API quota
        help="Limit the total number of keywords to analyze (higher = more comprehensive but slower)"
    )
    
    # Advanced options
    with st.sidebar.expander("Advanced Options"):
        save_to_file = st.checkbox("Save results to file", value=True)
        file_format = st.radio("File format", ["CSV", "JSON", "Both"])
        use_cache = st.checkbox("Use cached results when available", value=True, 
                              help="Reduces API calls by using previously fetched data when available")
        
    # Main content area for Keyword Research Process
    if st.sidebar.button("Run Keyword Research", type="primary"):
        # Process seed keywords
        if not seed_keywords.strip():
            st.error("Please enter at least one seed keyword")
            st.stop()
            
        # Parse seed keywords
        seed_list = [k.strip() for k in seed_keywords.split("\n") if k.strip()]
        
        if not seed_list:
            st.error("Please enter at least one valid seed keyword")
            st.stop()
        
        # Show seed keywords
        st.subheader("Starting Keyword Research Process")
        st.write(f"Seed Keywords: {', '.join(seed_list)}")
        
        # Create a cache directory if it doesn't exist
        cache_dir = DATA_DIR / "cache"
        cache_dir.mkdir(exist_ok=True, parents=True)
        
        try:
            # Run the keyword research process with API quota handling
            with st.spinner("Researching keywords... This may take a few minutes."):
                results = process_keywords(
                    seed_keywords=seed_list,
                    max_suggestions=suggestions_per_keyword,
                    limit=max_keywords,
                    use_cache=use_cache,
                    cache_dir=cache_dir
                )
            
            # Display results
            if results.get("success", False):
                # Show results in the UI
                show_keyword_results(results)
                
                # Save results to file if requested
                if save_to_file:
                    # Create timestamp for filename
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename_base = f"keyword_research_{timestamp}"
                    
                    # Create data directory if it doesn't exist
                    data_path = DATA_DIR / "keyword_research"
                    data_path.mkdir(exist_ok=True, parents=True)
                    
                    # Save files based on selected format
                    if file_format in ["CSV", "Both"]:
                        csv_path = data_path / f"{filename_base}.csv"
                        df = pd.DataFrame(results["all_keywords"])
                        df.to_csv(csv_path, index=False)
                        st.sidebar.success(f"CSV saved to: {csv_path}")
                        
                    if file_format in ["JSON", "Both"]:
                        json_path = data_path / f"{filename_base}.json"
                        save_results_to_file(results, json_path)
                        st.sidebar.success(f"JSON saved to: {json_path}")
            else:
                st.error(results.get("message", "An error occurred during keyword research."))
        
        except Exception as e:
            if "quota" in str(e).lower():
                st.error("""
                ⚠️ **YouTube API Quota Exceeded**
                
                You've reached your daily limit for YouTube API requests. Here are some options:
                
                1. Try again tomorrow when your quota resets
                2. Use a different API key
                3. Reduce the number of keywords you're analyzing
                4. Use cached results if available
                """)
            else:
                st.error(f"Error: {str(e)}")
    
    else:
        # Show instructions when the app first loads
        st.info("""
        👈 **How to use this tool:**
        
        1. Enter your seed keywords in the sidebar (one per line)
        2. Adjust the parameters as needed
        3. Click "Run Keyword Research"
        4. Review the results and export to CSV/JSON
        
        The tool will expand your seed keywords, analyze them, and calculate a Magic Score
        to help you find the best keywords for your YouTube content.
        """)
        
        # Show example results
        with st.expander("See example results"):
            st.markdown("""
            The results will include:
            
            - **Magic Score**: A proprietary score (M = V/C) that balances search volume and competition
            - **Keyword Categories**: Channel-specific, category-specific, and video-specific keywords
            - **Engagement Metrics**: Views, likes, comments, and engagement rates
            - **Competition Level**: From "Very Low" to "Very High"
            
            You can export the results to CSV for use in Google Sheets or Excel.
            """)

elif MODE == "Analyze Video":
    st.markdown("### Analyze YouTube Video with AI")
    st.markdown("Get insights, summaries, and metadata for any YouTube video.")
    
    video_url = st.text_input("Enter YouTube video URL or ID", placeholder="https://www.youtube.com/watch?v=...")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        include_comments = st.checkbox("Include comments", value=False)
    with col2:
        generate_summary = st.checkbox("Generate summary", value=True)
    with col3:
        generate_insights = st.checkbox("Generate insights", value=True)
        
    if st.button("Analyze Video"):
        # Use imported functions if available
        if TUBE_WIZARD_IMPORTS and video_url:
            with st.status("Processing video...") as status:
                # Get video ID
                video_id = get_video_id(video_url)
                if not video_id:
                    st.error("Invalid YouTube URL or ID")
                    status.update(label="Error: Invalid YouTube URL", state="error")
                    st.stop()
                
                # Get video info
                status.update(label="Fetching video information...")
                video_info = get_video_info(video_id)
                if not video_info:
                    st.error("Could not retrieve video information")
                    status.update(label="Error: Could not retrieve video info", state="error")
                    st.stop()
                
                # Display video info
                st.subheader(video_info['title'])
                col1, col2 = st.columns([3, 1])
                with col1:
                    st.markdown(f"**Channel:** {video_info['channel']}")
                    st.markdown(f"**Published:** {video_info['published']}")
                    st.markdown(f"**Duration:** {video_info['duration']} minutes")
                with col2:
                    st.markdown(f"**Views:** {video_info['views']:,}")
                    st.markdown(f"**Likes:** {video_info['likes']:,}")
                    st.markdown(f"**Comments:** {video_info['comments']:,}")
                
                # Get comments if requested
                video_comments = []
                if include_comments:
                    status.update(label="Fetching comments...")
                    video_comments = get_video_comments(video_id)
                    if video_comments:
                        with st.expander("Top Comments"):
                            for comment in video_comments[:10]:
                                st.markdown(f"**{comment['author']}:** {comment['text']}")
                                st.markdown("---")
                
                # Prepare content for AI analysis
                content_for_analysis = f"""
Title: {video_info['title']}
Channel: {video_info['channel']}
Description: {video_info['description']}
"""
                
                if include_comments and video_comments:
                    content_for_analysis += "\nTop Comments:\n"
                    for i, comment in enumerate(video_comments[:5], 1):
                        content_for_analysis += f"{i}. {comment['author']}: {comment['text']}\n"
                
                # Generate summary if requested
                if generate_summary:
                    status.update(label="Generating summary...")
                    summary_prompt = "Provide a concise summary of this YouTube video based on the following information:\n\n{content}"
                    summary_text = analyze_with_ai(content_for_analysis, summary_prompt)
                    if summary_text:
                        st.subheader("Video Summary")
                        st.markdown(summary_text)
                
                # Generate insights if requested
                if generate_insights:
                    status.update(label="Generating insights...")
                    insights_prompt = "Provide interesting insights and key takeaways from this YouTube video based on the following information:\n\n{content}"
                    insights_text = analyze_with_ai(content_for_analysis, insights_prompt)
                    if insights_text:
                        st.subheader("Video Insights")
                        st.markdown(insights_text)
                
                status.update(label="Analysis complete", state="complete")
        else:
            try:
                with st.spinner("Processing video..."):
                    # Extract video ID from URL
                    video_id = get_video_id(video_url)
                    if not video_id:
                        st.error("Invalid YouTube URL. Please enter a valid YouTube video URL.")
                        st.stop()
                    
                    # Check cache first if use_cache is enabled
                    cache_hit = False
                    cache_dir = DATA_DIR / "cache"
                    cache_dir.mkdir(exist_ok=True, parents=True)
                    cache_file = cache_dir / f"video_analysis_{video_id}.json"
                    
                    if use_cache and cache_file.exists():
                        try:
                            with open(cache_file, "r") as f:
                                video_data = json.load(f)
                            st.success(f"Found cached analysis for this video")
                            cache_hit = True
                        except Exception as e:
                            st.warning(f"Error reading cache: {str(e)}. Will fetch fresh data.")
                            cache_hit = False
                    
                    if not cache_hit:
                        # Get video info
                        video_request = youtube_client.videos().list(
                            id=video_id,
                            part="snippet,statistics,contentDetails"
                        )
                        video_response = video_request.execute()
                        
                        if not video_response.get("items"):
                            st.error("Video not found. Please check the URL and try again.")
                            st.stop()
                        
                        video_info = video_response["items"][0]
                        
                        # Get comments if requested
                        comments = []
                        if include_comments:
                            try:
                                comments_request = youtube_client.commentThreads().list(
                                    videoId=video_id,
                                    part="snippet",
                                    maxResults=10,
                                    order="relevance"
                                )
                                comments_response = comments_request.execute()
                                
                                for item in comments_response.get("items", []):
                                    comment = item["snippet"]["topLevelComment"]["snippet"]
                                    comments.append({
                                        "author": comment["authorDisplayName"],
                                        "text": comment["textDisplay"],
                                        "likes": comment["likeCount"],
                                        "published": comment["publishedAt"]
                                    })
                            except Exception as e:
                                st.warning(f"Could not fetch comments: {str(e)}")
                        
                        # Prepare video data for caching and analysis
                        video_data = {
                            "video_id": video_id,
                            "info": video_info,
                            "comments": comments
                        }
                        
                        # Cache the results
                        with open(cache_file, "w") as f:
                            json.dump(video_data, f)
                    
                    # Extract relevant information for display
                    snippet = video_data["info"]["snippet"]
                    statistics = video_data["info"]["statistics"]
                    content_details = video_data["info"]["contentDetails"]
                    
                    # Display video information
                    st.markdown(f"## {snippet['title']}")
                    
                    col1, col2 = st.columns([1, 2])
                    
                    with col1:
                        thumbnail_url = snippet.get("thumbnails", {}).get("high", {}).get("url", "")
                        if thumbnail_url:
                            st.image(thumbnail_url, use_column_width=True)
                    
                    with col2:
                        st.markdown(f"**Channel:** {snippet.get('channelTitle', 'Unknown')}")
                        st.markdown(f"**Published:** {snippet.get('publishedAt', 'Unknown')[:10]}")
                        
                        # Format statistics
                        views = int(statistics.get("viewCount", 0))
                        likes = int(statistics.get("likeCount", 0))
                        comments_count = int(statistics.get("commentCount", 0))
                        
                        # Parse duration
                        duration_str = content_details.get("duration", "PT0M0S")
                        duration_mins = parse_duration(duration_str)
                        
                        st.markdown(f"**Duration:** {duration_mins:.1f} minutes")
                        st.markdown(f"**Views:** {views:,}")
                        st.markdown(f"**Likes:** {likes:,}")
                        st.markdown(f"**Comments:** {comments_count:,}")
                        
                        # Calculate engagement rate
                        engagement_rate = (likes + comments_count) / max(views, 1) * 100
                        st.markdown(f"**Engagement Rate:** {engagement_rate:.2f}%")
                    
                    # Display description
                    with st.expander("Video Description", expanded=False):
                        st.markdown(snippet.get("description", "No description available."))
                    
                    # Display comments if available
                    if video_data.get("comments"):
                        with st.expander("Top Comments", expanded=False):
                            for comment in video_data["comments"]:
                                st.markdown(f"**{comment['author']}:** {comment['text']}")
                                st.markdown(f"Likes: {comment['likes']} | Published: {comment['published'][:10]}")
                                st.markdown("---")
                    
                    # Generate AI analysis if OpenAI API key is available
                    if OPENAI_API_KEY:
                        generate_summary = st.checkbox("Generate summary", value=True)
                        generate_insights = st.checkbox("Generate insights", value=True)
                        
                        if generate_summary or generate_insights:
                            # Prepare content for AI analysis
                            content_for_analysis = f"""
                            Title: {snippet['title']}
                            Channel: {snippet['channelTitle']}
                            Description: {snippet['description']}
                            """
                            
                            if video_data.get("comments"):
                                content_for_analysis += "\nTop Comments:\n"
                                for i, comment in enumerate(video_data["comments"][:5], 1):
                                    content_for_analysis += f"{i}. {comment['author']}: {comment['text']}\n"
                            
                            # Generate summary if requested
                            if generate_summary:
                                with st.spinner("Generating summary..."):
                                    summary_prompt = f"Provide a concise summary of this YouTube video based on the following information:\n\n{content_for_analysis}"
                                    
                                    # Call OpenAI API
                                    response = openai_client.chat.completions.create(
                                        model="gpt-3.5-turbo",
                                        messages=[{"role": "user", "content": summary_prompt}],
                                        max_tokens=500
                                    )
                                    
                                    summary_text = response.choices[0].message.content.strip()
                                    
                                    # Display summary
                                    st.markdown("### Video Summary")
                                    st.markdown(summary_text)
                            
                            # Generate insights if requested
                            if generate_insights:
                                with st.spinner("Generating insights..."):
                                    insights_prompt = f"Provide interesting insights and key takeaways from this YouTube video based on the following information:\n\n{content_for_analysis}"
                                    
                                    # Call OpenAI API
                                    response = openai_client.chat.completions.create(
                                        model="gpt-3.5-turbo",
                                        messages=[{"role": "user", "content": insights_prompt}],
                                        max_tokens=500
                                    )
                                    
                                    insights_text = response.choices[0].message.content.strip()
                                    
                                    # Display insights
                                    st.markdown("### Video Insights")
                                    st.markdown(insights_text)
                    
                    # Add a link to watch the video
                    st.markdown(f"[Watch on YouTube](https://www.youtube.com/watch?v={video_id})")
                    
                    # Export options
                    st.markdown("### Export Analysis")
                    
                    # Add download button for JSON data
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    st.download_button(
                        label="Download Analysis as JSON",
                        data=json.dumps(video_data, indent=2),
                        file_name=f"video_analysis_{video_id}_{timestamp}.json",
                        mime="application/json"
                    )
                    
            except HttpError as e:
                error_details = json.loads(e.content.decode("utf-8"))
                error_reason = error_details.get("error", {}).get("errors", [{}])[0].get("reason", "")
                
                if error_reason == "quotaExceeded":
                    st.error("YouTube API quota exceeded. Please try again tomorrow or use cached results.")
                    st.info("Tip: YouTube provides a daily quota of API requests. To avoid hitting the limit, use cached results when possible.")
                else:
                    st.error(f"YouTube API error: {str(e)}")
            except Exception as e:
                st.error(f"Error analyzing video: {str(e)}")


elif MODE == "Video Ideas":
    st.markdown("### Generate Video Ideas")
    st.markdown("Get AI-generated video ideas for your YouTube channel.")
    
    topic = st.text_input("Enter a topic or niche", placeholder="e.g., Python programming, fitness, cooking")
    num_ideas = st.slider("Number of ideas to generate", min_value=3, max_value=10, value=5)
    
    if st.button("Generate Ideas"):
        # Use imported function if available
        if TUBE_WIZARD_IMPORTS and topic:
            with st.status("Generating video ideas...") as status:
                ideas = generate_video_ideas(topic, num_ideas)
                if ideas:
                    st.subheader(f"Video Ideas for: {topic}")
                    for i, idea in enumerate(ideas, 1):
                        st.markdown(f"**{i}. {idea}**")
                        st.markdown("---")
                    status.update(label="Ideas generated successfully", state="complete")
                else:
                    st.error("Failed to generate ideas. Please check your OpenAI API key.")
                    status.update(label="Error generating ideas", state="error")
        else:
            if not topic:
                st.error("Please enter a topic or niche")
            elif not OPENAI_API_KEY:
                st.error("OpenAI API key is required for generating video ideas. Please add it to your .env file.")
            else:
                try:
                    with st.spinner(f"Generating {num_ideas} video ideas for '{topic}'..."):
                        # Generate video ideas using OpenAI
                        prompt = (
                            f"Provide {num_ideas} unique YouTube video ideas for a channel about '{topic}'. "
                            "Return them as a numbered list."
                        )
                        
                        # Call OpenAI API
                        response = openai_client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=800
                        )
                        
                        ideas_text = response.choices[0].message.content.strip()
                        
                        # Display ideas
                        st.subheader(f"Video Ideas for: {topic}")
                        st.markdown(ideas_text)
                except Exception as e:
                    error_msg = str(e).lower()
                    st.error(f"Error generating ideas: {str(e)}")
                    
                    if "rate limit" in error_msg:
                        st.warning("You've hit the OpenAI API rate limit. Please try again later.")
                    elif "billing" in error_msg:
                        st.warning("Your OpenAI account may have billing issues. Please check your account.")
                    elif "authentication" in error_msg or "api key" in error_msg:
                        st.warning("Authentication error with OpenAI. Please check your API key.")
                    else:
                        st.info("If you're seeing API errors, you might need to check your OpenAI API key or account status.")


elif MODE == "Generate Script":
    st.markdown("### Generate Video Script")
    st.markdown("Get a complete AI-generated script for your YouTube video.")
    
    title = st.text_input("Enter a video title or topic", placeholder="e.g., How to Build a Website with React")
    duration = st.slider("Approximate video duration (minutes)", min_value=3, max_value=15, value=5)
    
    if st.button("Generate Script"):
        if not title:
            st.error("Please enter a video title or topic")
        elif not OPENAI_API_KEY:
            st.error("OpenAI API key is required for generating scripts. Please add it to your .env file.")
        else:
            # Use imported function if available
            if TUBE_WIZARD_IMPORTS:
                with st.status("Generating script...") as status:
                    script = generate_script(title, duration)
                    if script:
                        st.subheader(f"Script for: {title}")
                        st.markdown(script)
                        
                        # Add download button
                        script_text = f"# Script for: {title}\n\n{script}"
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"script_{title.replace(' ', '_')}_{timestamp}.md"
                        
                        st.download_button(
                            label="Download Script",
                            data=script_text,
                            file_name=filename,
                            mime="text/markdown"
                        )
                        
                        status.update(label="Script generated successfully", state="complete")
                    else:
                        st.error("Failed to generate script. Please check your OpenAI API key.")
                        status.update(label="Error generating script", state="error")
            else:
                try:
                    with st.spinner(f"Generating a {duration}-minute script for '{title}'..."):
                        # Generate script using OpenAI
                        prompt = (
                            f"Write a detailed YouTube script (~{duration} minutes) for a video titled '{title}'. "
                            "Include an engaging introduction, well-structured main points, and a strong conclusion."
                        )
                        
                        # Call OpenAI API
                        response = openai_client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=1500
                        )
                        
                        script_text = response.choices[0].message.content.strip()
                        
                        # Display script
                        st.subheader(f"Script for: {title}")
                        st.markdown(script_text)
                        
                        # Add download button
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"script_{title.replace(' ', '_')}_{timestamp}.md"
                        
                        st.download_button(
                            label="Download Script",
                            data=script_text,
                            file_name=filename,
                            mime="text/markdown"
                        )
                        
                        # Create script directory if it doesn't exist
                        script_dir = DATA_DIR / "scripts"
                        script_dir.mkdir(exist_ok=True, parents=True)
                        file_path = script_dir / filename
                        
                        # Save script to file
                        with open(file_path, "w") as f:
                            f.write(f"Script for: {title}\n\n")
                            f.write(script_text)
                        
                        # Add button to optimize metadata based on this script
                        if st.button("Generate SEO Metadata from this Script"):
                            st.session_state.optimize_metadata_topic = title
                            st.session_state.optimize_metadata_script = script_text
                            st.session_state.app_mode = "Optimize Metadata"
                            st.rerun()
                except Exception as e:
                    error_msg = str(e).lower()
                    st.error(f"Error generating script: {str(e)}")
                    
                    if "rate limit" in error_msg:
                        st.warning("You've hit the OpenAI API rate limit. Please try again later.")
                    elif "billing" in error_msg:
                        st.warning("Your OpenAI account may have billing issues. Please check your account.")
                    elif "authentication" in error_msg or "api key" in error_msg:
                        st.warning("Authentication error with OpenAI. Please check your API key.")
                    else:
                        st.info("If you're seeing API errors, you might need to check your OpenAI API key or account status.")


elif MODE == "Optimize Metadata":
    st.markdown("### YouTube Metadata Optimizer")
    
    # Metadata input
    topic = st.text_input("Video Topic/Title")
    current_description = st.text_area("Current Description (optional)", height=150)
    script_file = st.file_uploader("Upload Script File (optional)", type=["txt"])
    
    # Check if we have a script from the script generator
    if hasattr(st.session_state, 'optimize_metadata_topic') and hasattr(st.session_state, 'optimize_metadata_script'):
        topic = st.session_state.optimize_metadata_topic
        script_content = st.session_state.optimize_metadata_script
        st.info(f"Using script for '{topic}' from Script Generator")
        # Clear the session state
        del st.session_state.optimize_metadata_topic
        del st.session_state.optimize_metadata_script
    else:
        script_content = None
        if script_file is not None:
            script_content = script_file.getvalue().decode("utf-8")
            st.success("Script file uploaded successfully")
    
    if st.button("Optimize Metadata", type="primary"):
        if not topic:
            st.error("Please enter a video topic or title")
        elif not OPENAI_API_KEY:
            st.error("OpenAI API key is required for optimizing metadata. Please add it to your .env file.")
        else:
            # Use imported function if available
            if TUBE_WIZARD_IMPORTS:
                with st.status("Optimizing metadata...") as status:
                    metadata = optimize_metadata(topic, script_content, current_description)
                    if metadata:
                        st.subheader("Optimized Metadata")
                        
                        st.markdown("#### Title")
                        st.code(metadata["title"], language="")
                        
                        st.markdown("#### Description")
                        st.code(metadata["description"], language="")
                        
                        st.markdown("#### Tags")
                        st.code(", ".join(metadata["tags"]), language="")
                        
                        # Add download button
                        metadata_text = f"# Optimized Metadata for: {topic}\n\n"
                        metadata_text += f"## Title\n{metadata['title']}\n\n"
                        metadata_text += f"## Description\n{metadata['description']}\n\n"
                        metadata_text += f"## Tags\n{', '.join(metadata['tags'])}\n"
                        
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        filename = f"metadata_{topic.replace(' ', '_')}_{timestamp}.md"
                        
                        st.download_button(
                            label="Download Metadata",
                            data=metadata_text,
                            file_name=filename,
                            mime="text/markdown"
                        )
                        
                        status.update(label="Metadata optimized successfully", state="complete")
                    else:
                        st.error("Failed to optimize metadata. Please check your OpenAI API key.")
                        status.update(label="Error optimizing metadata", state="error")
            else:
                try:
                    with st.spinner(f"Optimizing metadata for '{topic}'..."):
                        # Generate optimized metadata using OpenAI
                        prompt = (
                            f"Generate an SEO-optimised YouTube title, a 200-word description, and exactly 15 one-word keyword tags for a video about '{topic}'. "
                            "Return the result strictly as valid JSON with keys: title, description, tags."
                        )
                        
                        if script_content:
                            prompt += f"\n\nHere is the draft script for additional context:\n{script_content[:3000]}"
                        
                        if current_description:
                            prompt += f"\n\nHere is the current description for reference:\n{current_description}"
                        
                        # Call OpenAI API
                        response = openai_client.chat.completions.create(
                            model="gpt-3.5-turbo",
                            messages=[{"role": "user", "content": prompt}],
                            max_tokens=800
                        )
                    
                        meta_text = response.choices[0].message.content.strip()
                        
                        try:
                            meta = json.loads(meta_text)
                        except Exception:
                            st.warning("Unable to parse JSON metadata. Showing raw output instead.")
                            st.text(meta_text)
                            meta = {"raw": meta_text}
                        
                        # Display optimized metadata
                        st.success("Generated optimized metadata")
                        
                        # Create a nice display for the metadata
                        st.markdown("### Optimized Title")
                        st.markdown(f"**{meta.get('title', 'No title generated')}**")
                        
                        st.markdown("### Optimized Description")
                        st.text_area("Copy this description", meta.get('description', 'No description generated'), height=200)
                        
                        st.markdown("### Suggested Tags")
                        if 'tags' in meta and isinstance(meta['tags'], list):
                            tags_str = ', '.join(meta['tags'])
                            st.text_area("Copy these tags", tags_str, height=100)
                        else:
                            st.info("No tags generated or tags were not in the expected format")
                    
                        # Save metadata to file
                        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                        metadata_dir = DATA_DIR / "metadata"
                        metadata_dir.mkdir(exist_ok=True, parents=True)
                        
                        filename = f"metadata_{topic.replace(' ', '_')}_{timestamp}.json"
                        file_path = metadata_dir / filename
                        
                        with open(file_path, "w") as f:
                            json.dump(meta, f, indent=2)
                        
                        # Add download button
                        st.download_button(
                            label="Download Metadata as JSON",
                            data=json.dumps(meta, indent=2),
                            file_name=filename,
                            mime="application/json"
                        )
                        
                        st.success(f"Metadata saved to {file_path}")
                except Exception as e:
                    error_msg = str(e).lower()
                    st.error(f"Error optimizing metadata: {str(e)}")
                    
                    if "rate limit" in error_msg:
                        st.warning("You've hit the OpenAI API rate limit. Please try again later.")
                    elif "billing" in error_msg:
                        st.warning("Your OpenAI account may have billing issues. Please check your account.")
                    elif "authentication" in error_msg or "api key" in error_msg:
                        st.warning("Authentication error with OpenAI. Please check your API key.")
                    else:
                        st.info("If you're seeing API errors, you might need to check your OpenAI API key or account status.")


elif MODE == "Basic Keyword Research":
    st.markdown("### Basic YouTube Keyword Research")
    
    # Basic keyword research input
    keyword = st.text_input("Enter a keyword to research")
    max_results = st.slider("Maximum results to show", 5, 50, 20)
    use_cache = st.checkbox("Use cached results if available", value=True)
    
    if st.button("Research Keyword", type="primary"):
        if not keyword:
            st.error("Please enter a keyword to research")
        elif not YOUTUBE_API_KEY:
            st.error("YouTube API key is required for keyword research. Please add it to your .env file.")
        else:
            try:
                with st.spinner(f"Researching keyword: '{keyword}'..."):
                    # Check cache first if use_cache is enabled
                    cache_hit = False
                    cache_dir = DATA_DIR / "cache"
                    cache_dir.mkdir(exist_ok=True, parents=True)
                    cache_file = cache_dir / f"basic_keyword_{keyword.replace(' ', '_')}.json"
                    
                    if use_cache and cache_file.exists():
                        try:
                            with open(cache_file, "r") as f:
                                results = json.load(f)
                            st.success(f"Found cached results for '{keyword}'")
                            cache_hit = True
                        except Exception as e:
                            st.warning(f"Error reading cache: {str(e)}. Will fetch fresh data.")
                            cache_hit = False
                    
                    if not cache_hit:
                        # Get keyword suggestions
                        suggestions = keyword_suggestions(keyword)
                        
                        # Get search results for the keyword
                        search_request = youtube_client.search().list(
                            q=keyword,
                            part="id,snippet",
                            type="video",
                            maxResults=max_results
                        )
                        search_response = search_request.execute()
                        
                        # Extract video IDs
                        video_ids = [item["id"]["videoId"] for item in search_response.get("items", [])]
                        
                        # Get video statistics
                        if video_ids:
                            videos_request = youtube_client.videos().list(
                                id=",".join(video_ids),
                                part="statistics,contentDetails,snippet"
                            )
                            videos_response = videos_request.execute()
                        else:
                            videos_response = {"items": []}
                        
                        # Prepare results
                        results = {
                            "keyword": keyword,
                            "suggestions": suggestions,
                            "search_results": search_response.get("items", []),
                            "video_stats": videos_response.get("items", [])
                        }
                        
                        # Cache the results
                        with open(cache_file, "w") as f:
                            json.dump(results, f)
                    
                    # Display results
                    st.markdown(f"## Results for '{keyword}'")
                    
                    # Display keyword suggestions
                    if results.get("suggestions"):
                        st.markdown("### Related Keywords")
                        suggestions_cols = st.columns(3)
                        for i, suggestion in enumerate(results["suggestions"]):
                            col_idx = i % 3
                            with suggestions_cols[col_idx]:
                                st.write(f"- {suggestion}")
                    
                    # Display video results with stats
                    st.markdown("### Top Videos")
                    
                    # Create a dictionary mapping video IDs to their statistics
                    video_stats = {}
                    for item in results.get("video_stats", []):
                        video_stats[item["id"]] = item
                    
                    # Display each video with its stats
                    for i, item in enumerate(results.get("search_results", []), 1):
                        video_id = item["id"]["videoId"]
                        snippet = item["snippet"]
                        stats = video_stats.get(video_id, {})
                        
                        col1, col2 = st.columns([1, 3])
                        
                        with col1:
                            thumbnail_url = snippet.get("thumbnails", {}).get("medium", {}).get("url", "")
                            if thumbnail_url:
                                st.image(thumbnail_url, use_column_width=True)
                        
                        with col2:
                            st.markdown(f"**{i}. {snippet.get('title', 'No title')}**")
                            st.markdown(f"Channel: {snippet.get('channelTitle', 'Unknown')}")
                            st.markdown(f"Published: {snippet.get('publishedAt', 'Unknown')[:10]}")
                            
                            # Display statistics if available
                            if stats and "statistics" in stats:
                                statistics = stats["statistics"]
                                views = int(statistics.get("viewCount", 0))
                                likes = int(statistics.get("likeCount", 0))
                                comments = int(statistics.get("commentCount", 0))
                                
                                st.markdown(f"Views: {views:,} | Likes: {likes:,} | Comments: {comments:,}")
                            
                            # Add a link to the video
                            st.markdown(f"[Watch on YouTube](https://www.youtube.com/watch?v={video_id})")
                        
                        st.markdown("---")
                    
                    # Calculate and display summary statistics
                    if results.get("video_stats"):
                        st.markdown("### Summary Statistics")
                        
                        total_views = sum(int(item.get("statistics", {}).get("viewCount", 0)) for item in results["video_stats"])
                        total_likes = sum(int(item.get("statistics", {}).get("likeCount", 0)) for item in results["video_stats"])
                        total_comments = sum(int(item.get("statistics", {}).get("commentCount", 0)) for item in results["video_stats"])
                        
                        avg_views = total_views / max(len(results["video_stats"]), 1)
                        avg_likes = total_likes / max(len(results["video_stats"]), 1)
                        avg_comments = total_comments / max(len(results["video_stats"]), 1)
                        
                        engagement_rate = (total_likes + total_comments) / max(total_views, 1) * 100
                        
                        col1, col2, col3, col4 = st.columns(4)
                        col1.metric("Avg. Views", f"{avg_views:,.0f}")
                        col2.metric("Avg. Likes", f"{avg_likes:,.0f}")
                        col3.metric("Avg. Comments", f"{avg_comments:,.0f}")
                        col4.metric("Engagement Rate", f"{engagement_rate:.2f}%")
                    
                    # Add export options
                    st.markdown("### Export Results")
                    
                    # Convert results to CSV format for download
                    csv_data = io.StringIO()
                    csv_writer = csv.writer(csv_data)
                    
                    # Write header
                    csv_writer.writerow(["Title", "Channel", "Published", "Views", "Likes", "Comments", "URL"])
                    
                    # Write data
                    for item in results.get("search_results", []):
                        video_id = item["id"]["videoId"]
                        snippet = item["snippet"]
                        stats = video_stats.get(video_id, {})
                        
                        views = int(stats.get("statistics", {}).get("viewCount", 0))
                        likes = int(stats.get("statistics", {}).get("likeCount", 0))
                        comments = int(stats.get("statistics", {}).get("commentCount", 0))
                        
                        csv_writer.writerow([
                            snippet.get("title", ""),
                            snippet.get("channelTitle", ""),
                            snippet.get("publishedAt", "")[:10],
                            views,
                            likes,
                            comments,
                            f"https://www.youtube.com/watch?v={video_id}"
                        ])
                    
                    # Add download buttons
                    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
                    
                    st.download_button(
                        label="Download as CSV",
                        data=csv_data.getvalue(),
                        file_name=f"keyword_research_{keyword.replace(' ', '_')}_{timestamp}.csv",
                        mime="text/csv"
                    )
                    
                    st.download_button(
                        label="Download as JSON",
                        data=json.dumps(results, indent=2),
                        file_name=f"keyword_research_{keyword.replace(' ', '_')}_{timestamp}.json",
                        mime="application/json"
                    )
                    
            except HttpError as e:
                error_details = json.loads(e.content.decode("utf-8"))
                error_reason = error_details.get("error", {}).get("errors", [{}])[0].get("reason", "")
                
                if error_reason == "quotaExceeded":
                    st.error("YouTube API quota exceeded. Please try again tomorrow or use cached results.")
                    st.info("Tip: YouTube provides a daily quota of API requests. To avoid hitting the limit, use cached results when possible.")
                else:
                    st.error(f"YouTube API error: {str(e)}")
            except Exception as e:
                st.error(f"Error: {str(e)}")


# About section in sidebar for all modes
with st.sidebar.expander("About Tube Wizard"):
    st.markdown("""
    **Tube Wizard** is a lightweight YouTube AI assistant that helps content creators with:
    
    - Keyword research and SEO optimization
    - Content strategy and planning
    - Video metadata optimization
    - Video analysis and insights
    - Script generation
    
    Built with ❤️ using Python, Streamlit, and the YouTube Data API.
    """)

# Add API quota information
with st.sidebar.expander("API Quota Information"):
    st.markdown("""
    **YouTube API Quota Usage**
    
    The YouTube Data API has daily quota limits. If you encounter quota errors:
    
    1. Use the cache option to reduce API calls
    2. Reduce the number of keywords analyzed
    3. Try again tomorrow when your quota resets
    4. Consider using a different API key
    
    [Learn more about YouTube API quotas](https://developers.google.com/youtube/v3/getting-started#quota)
    """)


# Footer
st.markdown("---")
st.caption("Tube Wizard • Built with ❤️ using Streamlit and YouTube Data API • 2025")
