"""
Streamlit web app version of Tube Magic Helper.
Provides a browser-based UI that wraps the CLI functionality.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List

import streamlit as st
from dotenv import load_dotenv

# -----------------------------------------------------------------------------
# Adjust Python path so we can import the CLI module when the app is run via
# `streamlit run streamlit_app/app.py` from project root.
# -----------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from tube_magic_helper import (
    check_api_keys,
    get_video_id,
    get_video_info,
    get_video_comments,
    analyze_with_ai,
    generate_video_ideas,
    generate_script,
    optimize_metadata,
    keyword_research,
)

# -----------------------------------------------------------------------------
# Environment / configuration
# -----------------------------------------------------------------------------
load_dotenv()

if not check_api_keys():
    st.warning(
        "Missing API keys. Please set YOUTUBE_API_KEY and OPENAI_API_KEY in the environment."
    )
    st.stop()

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="Tube Magic Helper", page_icon="🎬", layout="wide")

# -----------------------------------------------------------------------------
# Helper display functions (Streamlit primitives)
# -----------------------------------------------------------------------------

def show_video_info(info: dict):
    """Display video metadata."""
    st.markdown(f"## {info['title']}")
    stats_line = (
        f"**Channel:** {info['channel']}  |  **Views:** {info['views']}  |  **Published:** {info['publish_date']}"
    )
    st.write(stats_line)
    st.markdown("### Description")
    st.write(info["description"])


def show_ai_panel(text: str, title: str):
    """Show AI-generated content in an expandable panel."""
    if not text:
        return
    with st.expander(title, expanded=True):
        st.markdown(text)


# -----------------------------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------------------------
MODE = st.sidebar.radio(
    "Choose a function",
    [
        "Analyze Video",
        "Video Ideas",
        "Generate Script",
        "Optimize Metadata",
        "Keyword Research",
    ],
)

# -----------------------------------------------------------------------------
# Mode: Analyze Video
# -----------------------------------------------------------------------------
if MODE == "Analyze Video":
    st.header("📊 Analyze a YouTube Video")
    url = st.text_input("YouTube URL or ID", placeholder="https://www.youtube.com/watch?v=dQw4w9WgXcQ")

    cols = st.columns(3)
    with cols[0]:
        include_comments = st.checkbox("Include Comments", value=False)
    with cols[1]:
        summary_opt = st.checkbox("Generate Summary", value=True)
    with cols[2]:
        insights_opt = st.checkbox("Generate Insights", value=True)

    if st.button("Analyze", type="primary") and url:
        with st.spinner("Fetching video information..."):
            vid_id = get_video_id(url)
            if not vid_id:
                st.error("Invalid YouTube URL or ID")
                st.stop()
            info = get_video_info(vid_id)
            if not info:
                st.error("Unable to retrieve video information")
                st.stop()

        show_video_info(info)

        comments: List[dict] = []
        if include_comments:
            with st.spinner("Loading comments..."):
                comments = get_video_comments(vid_id)

        # Build analysis content
        analysis_content = (
            f"Title: {info['title']}\n"
            f"Channel: {info['channel']}\n"
            f"Description: {info['description']}\n"
        )
        if include_comments and comments:
            analysis_content += "\nTop Comments:\n"
            for i, c in enumerate(comments[:5], 1):
                analysis_content += f"{i}. {c['author']}: {c['text']}\n"

        if summary_opt:
            with st.spinner("Generating summary..."):
                summary_prompt = (
                    "Provide a concise summary of this YouTube video based on the following information:\n\n{content}"
                )
                summary_text = analyze_with_ai(analysis_content, summary_prompt)
            show_ai_panel(summary_text, "Video Summary")

        if insights_opt:
            with st.spinner("Generating insights..."):
                insights_prompt = (
                    "Provide interesting insights and key takeaways from this YouTube video based on the following information:\n\n{content}"
                )
                insights_text = analyze_with_ai(analysis_content, insights_prompt)
            show_ai_panel(insights_text, "Video Insights")

# -----------------------------------------------------------------------------
# Mode: Video Ideas
# -----------------------------------------------------------------------------
elif MODE == "Video Ideas":
    st.header("💡 Generate Video Ideas")
    niche = st.text_input("Niche / Topic", placeholder="personal finance")
    num = st.slider("Number of ideas", min_value=5, max_value=30, value=10)
    if st.button("Generate", type="primary") and niche:
        with st.spinner("Generating ideas..."):
            ideas = generate_video_ideas(niche, num)
        if ideas:
            st.markdown("### Ideas")
            st.write("\n".join(f"{i+1}. {idea}" for i, idea in enumerate(ideas)))
            fp = DATA_DIR / f"ideas_{niche.replace(' ', '_')}.txt"
            fp.write_text("\n".join(ideas))
            st.success(f"Ideas saved to {fp}")

# -----------------------------------------------------------------------------
# Mode: Generate Script
# -----------------------------------------------------------------------------
elif MODE == "Generate Script":
    st.header("📝 AI Script Writer")
    title = st.text_input("Video Title / Topic")
    minutes = st.slider("Approximate length (minutes)", 3, 15, 5)
    if st.button("Create Script", type="primary") and title:
        with st.spinner("Generating script..."):
            script_text = generate_script(title, minutes)
        if script_text:
            st.markdown("### Generated Script")
            st.write(script_text)
            scripts_dir = DATA_DIR / "scripts"
            scripts_dir.mkdir(exist_ok=True)
            fn = scripts_dir / f"{title.replace(' ', '_')}.txt"
            fn.write_text(script_text)
            st.success(f"Script saved to {fn}")

# -----------------------------------------------------------------------------
# Mode: Optimize Metadata
# -----------------------------------------------------------------------------
elif MODE == "Optimize Metadata":
    st.header("🔧 Optimize Video Metadata")
    topic = st.text_input("Video Topic / Title")
    script_input = st.text_area("Optional: Paste script text for context")
    if st.button("Generate Metadata", type="primary") and topic:
        with st.spinner("Optimizing metadata..."):
            meta = optimize_metadata(topic, script_input or None)
        if not meta:
            st.error("Failed to generate metadata")
        elif "raw" in meta:
            st.write(meta["raw"])
        else:
            st.subheader("Title")
            st.write(meta["title"])
            st.subheader("Description")
            st.write(meta["description"])
            st.subheader("Tags (15)")
            st.write(", ".join(meta.get("tags", [])))

# -----------------------------------------------------------------------------
# Mode: Keyword Research
# -----------------------------------------------------------------------------
elif MODE == "Keyword Research":
    st.header("🔍 Keyword Research")
    seeds = st.text_input("Seed keywords (comma-separated)")
    limit = st.slider("Number of results", 5, 20, 10)
    if st.button("Research", type="primary") and seeds:
        seed_list = [k.strip() for k in seeds.split(",") if k.strip()]
        with st.spinner("Researching..."):
            results = keyword_research(seed_list, limit)
        if results:
            st.markdown("### Results")
            st.table([
                {
                    "Keyword": r["keyword"],
                    "Total Views": f"{r['views']:,}",
                    "Avg Views": f"{r['avg_views']:,}",
                    "Competition": r["competition"],
                    "Score": r["score"],
                }
                for r in results
            ])

            # Simple bar chart of scores
            st.bar_chart({r["keyword"]: r["score"] for r in results})

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Tube Magic Helper CLI • Streamlit front-end")
