### Streamlit Book Recommendation System - Interactive Demo - REVISION 2.0

"""
Hệ thống gợi ý sách - Ứng dụng demo tương tác

Ứng dụng web Streamlit cho gợi ý sách theo hướng lai,
kết hợp lọc cộng tác và ghép nối theo nội dung.

Author: Quang
GitHub: @OuyangXuelili
"""

from __future__ import annotations

import base64
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import gzip
import heapq
import html
import json
import os
import re
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from collections import Counter, defaultdict
from typing import List
import random
import requests
from scipy import sparse
from sklearn.decomposition import TruncatedSVD

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import GoodreadsLoader

# Page configuration
st.set_page_config(
    page_title="BRS | Hệ gợi ý sách",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="auto"
)

# ============================================================================
# COLOR SCHEME - Warm Book Theme (Coral, Amber, Terracotta)
# ============================================================================
COLORS = {
    "primary": "#E94560",
    "secondary": "#F18F01",
    "accent": "#C44536",
    "highlight": "#F4A261",
    "background": "#FFF8F0",
    "card_bg": "#FDF6EC",
    "text_dark": "#2D2A32",
    "text_light": "#6B5B6E",
}

GENRE_LABELS = {
    "Classic Fiction": "Văn học kinh điển",
    "Dystopian Fiction": "Giả tưởng phản địa đàng",
    "Romance": "Lãng mạn",
    "Magical Realism": "Hiện thực huyền ảo",
    "Fantasy": "Kỳ ảo",
    "Science Fiction": "Khoa học viễn tưởng",
    "Mystery": "Bí ẩn",
    "Thriller": "Kịch tính",
    "True Crime": "Tội phạm có thật",
    "Non-Fiction": "Phi hư cấu",
    "Self-Help": "Tự lực",
    "Psychology": "Tâm lý học",
    "Biography": "Tiểu sử",
    "Business": "Kinh doanh",
    "Philosophy": "Triết học",
    "Memoir": "Hồi ký",
    "Historical Fiction": "Tiểu thuyết lịch sử",
    "Contemporary Fiction": "Tiểu thuyết đương đại",
    "General Fiction": "Tiểu thuyết phổ thông",
    "Horror": "Kinh dị",
    "Young Adult": "Thiếu niên",
    "Graphic Novels": "Truyện tranh",
    "Poetry": "Thơ",
    "Science": "Khoa học",
    "Education": "Giáo dục",
    "Cooking": "Ẩm thực",
    "Travel": "Du lịch",
    "Art": "Nghệ thuật",
    "Music": "Âm nhạc",
    "Health": "Sức khỏe",
}

MOOD_LABELS = {
    "Adventurous": "Phiêu lưu",
    "Romantic": "Lãng mạn",
    "Intellectual": "Trí tuệ",
    "Thrilling": "Hồi hộp",
    "Classic Vibes": "Cổ điển",
    "Emotional": "Cảm xúc",
    "Escapist": "Thoát ly",
}

TAB_LABELS = [
    "Gợi ý cá nhân",
    "Sách phổ biến",
    "Tìm sách tương tự",
    "Khám phá dữ liệu",
    "Hiệu suất mô hình",
]

DEFAULT_HYBRID_WEIGHTS = {
    "genre": 0.46,
    "rating": 0.20,
    "popularity": 0.16,
    "author": 0.12,
    "novelty": 0.06,
}


def translate_genre(genre: str) -> str:
    normalized_genre = _normalize_recommendation_text(genre)
    genre_aliases = {
        _normalize_recommendation_text(key): value
        for key, value in GENRE_LABELS.items()
    }
    genre_aliases.update({
        "classics": "Văn học kinh điển",
        "classic": "Văn học kinh điển",
        "historical fiction": "Tiểu thuyết lịch sử",
        "historical fiction ": "Tiểu thuyết lịch sử",
        "historical-fiction": "Tiểu thuyết lịch sử",
        "non fiction": "Phi hư cấu",
        "non-fiction": "Phi hư cấu",
        "young adult": "Thiếu niên",
        "young-adult": "Thiếu niên",
        "ya": "Thiếu niên",
        "chick lit": "Lãng mạn",
        "chick-lit": "Lãng mạn",
        "contemporary": "Tiểu thuyết đương đại",
        "general fiction": "Tiểu thuyết phổ thông",
        "literary fiction": "Tiểu thuyết đương đại",
        "women s fiction": "Tiểu thuyết đương đại",
        "graphic novels": "Truyện tranh",
        "graphic-novels": "Truyện tranh",
        "comics": "Truyện tranh",
        "manga": "Truyện tranh",
        "spiritual": "Triết học",
        "spirituality": "Triết học",
    })
    return genre_aliases.get(normalized_genre, genre)


def translate_mood(mood: str) -> str:
    return MOOD_LABELS.get(mood, mood)


def format_user_profile_label(user_id: str, rating_count=None) -> str:
    clean_id = str(user_id)
    if clean_id.startswith("user_"):
        clean_id = clean_id.replace("user_", "", 1)
    label = f"Hồ sơ {clean_id}"
    if rating_count is not None:
        return f"{label} ({int(rating_count)} đánh giá)"
    return label


def translate_reason(reason: str) -> str:
    if reason.startswith("Same genre:"):
        genre = reason.split(":", 1)[1].strip()
        return f"Cùng thể loại: {translate_genre(genre)}"
    if reason.startswith("Same author:"):
        author = reason.split(":", 1)[1].strip()
        return f"Cùng tác giả: {author}"
    if reason.startswith("Based on your interest in"):
        genre = reason.replace("Based on your interest in", "", 1).strip()
        return f"Dựa trên sở thích của bạn với {translate_genre(genre)}"

    replacements = {
        "Same genre": "Cùng thể loại",
        "Same author": "Cùng tác giả",
        "Matches your": "Phù hợp với tâm trạng",
        "Based on your interest in": "Dựa trên sở thích của bạn với",
    }
    for english, vietnamese in replacements.items():
        if reason.startswith(english):
            return reason.replace(english, vietnamese, 1)
    return reason


def build_cover_mark(title: str) -> str:
    words = [word for word in title.split() if word]
    if not words:
        return "BK"

    initials = "".join(word[0] for word in words[:2]).upper()
    return initials[:2] if len(initials) > 1 else f"{initials}K"


def get_cover_colors(genre: str) -> tuple[str, str]:
    palette = {
        "Classic Fiction": ("#243b53", "#d4a373"),
        "Dystopian Fiction": ("#2f3e46", "#84a98c"),
        "Romance": ("#8f2d56", "#f2b5d4"),
        "Magical Realism": ("#2d6a4f", "#b7e4c7"),
        "Fantasy": ("#3a0ca3", "#f9c74f"),
        "Science Fiction": ("#005f73", "#94d2bd"),
        "Mystery": ("#232946", "#eebbc3"),
        "Thriller": ("#3d405b", "#e07a5f"),
        "True Crime": ("#1f2937", "#d97706"),
        "Non-Fiction": ("#0f766e", "#f4d35e"),
        "Self-Help": ("#1d4ed8", "#a7f3d0"),
        "Psychology": ("#7c3aed", "#c4b5fd"),
        "Biography": ("#0f172a", "#93c5fd"),
        "Business": ("#164e63", "#facc15"),
        "Philosophy": ("#312e81", "#c7d2fe"),
        "Memoir": ("#be123c", "#fecdd3"),
        "Historical Fiction": ("#6b4226", "#f2cc8f"),
        "Contemporary Fiction": ("#0f766e", "#fed7aa"),
        "Horror": ("#111827", "#ef4444"),
        "Young Adult": ("#9a3412", "#fdba74"),
    }
    return palette.get(genre, ("#334155", "#cbd5e1"))


def escape_html(value: object) -> str:
    return html.escape(str(value), quote=True)


def _cover_text_lines(value: str, max_chars: int = 19, max_lines: int = 4) -> list[str]:
    words = str(value).split()
    lines: list[str] = []
    current: list[str] = []

    for word in words:
        candidate = " ".join([*current, word])
        if current and len(candidate) > max_chars:
            lines.append(" ".join(current))
            current = [word]
            if len(lines) == max_lines:
                break
        else:
            current.append(word)

    if current and len(lines) < max_lines:
        lines.append(" ".join(current))

    if len(lines) == max_lines and len(words) > len(" ".join(lines).split()):
        lines[-1] = f"{lines[-1][: max_chars - 1].rstrip()}..."

    return lines or ["Untitled"]


def build_cover_svg_data_uri(title: str, genre: str, author: str = "") -> str:
    palette_one, palette_two = get_cover_colors(genre)
    title_lines = _cover_text_lines(title)
    title_svg = "\n".join(
        f"<text x='34' y='{154 + idx * 34}' class='title'>{escape_html(line)}</text>"
        for idx, line in enumerate(title_lines)
    )
    author_line = escape_html(author or "Unknown author")
    genre_line = escape_html(translate_genre(genre))
    mark = escape_html(build_cover_mark(title))

    svg = f"""
    <svg xmlns='http://www.w3.org/2000/svg' width='420' height='640' viewBox='0 0 420 640'>
        <defs>
            <linearGradient id='coverGradient' x1='0%' y1='0%' x2='100%' y2='100%'>
                <stop offset='0%' stop-color='{palette_one}'/>
                <stop offset='100%' stop-color='{palette_two}'/>
            </linearGradient>
            <linearGradient id='paper' x1='0%' y1='0%' x2='0%' y2='100%'>
                <stop offset='0%' stop-color='rgba(255,255,255,0.26)'/>
                <stop offset='100%' stop-color='rgba(255,255,255,0.08)'/>
            </linearGradient>
            <style>
                .label {{ font: 700 18px Inter, Arial, sans-serif; letter-spacing: 0; fill: rgba(255,255,255,0.78); }}
                .title {{ font: 800 31px Georgia, serif; letter-spacing: 0; fill: white; }}
                .author {{ font: 600 19px Inter, Arial, sans-serif; letter-spacing: 0; fill: rgba(255,255,255,0.84); }}
                .mark {{ font: 900 54px Inter, Arial, sans-serif; letter-spacing: 0; fill: rgba(255,255,255,0.94); }}
            </style>
        </defs>
        <rect width='420' height='640' rx='28' fill='url(#coverGradient)'/>
        <rect x='28' y='28' width='364' height='584' rx='22' fill='url(#paper)' stroke='rgba(255,255,255,0.22)'/>
        <rect x='0' y='0' width='42' height='640' fill='rgba(0,0,0,0.14)'/>
        <path d='M72 96 H342' stroke='rgba(255,255,255,0.44)' stroke-width='2'/>
        <path d='M72 116 H244' stroke='rgba(255,255,255,0.22)' stroke-width='2'/>
        <text x='34' y='82' class='label'>{genre_line}</text>
        {title_svg}
        <text x='34' y='466' class='author'>{author_line}</text>
        <text x='34' y='558' class='mark'>{mark}</text>
    </svg>
    """.strip()
    encoded = base64.b64encode(svg.encode("utf-8")).decode("ascii")
    return f"data:image/svg+xml;base64,{encoded}"


def _normalize_recommendation_text(value: str) -> str:
    return str(value).casefold().replace("-", " ").replace("_", " ").strip()


NOISY_SHELF_LABELS = {
    "",
    "nan",
    "none",
    "null",
    "book",
    "books",
    "default",
    "library",
    "my library",
    "shelf",
    "shelves",
    "read",
    "to read",
    "currently reading",
    "did not finish",
    "didn t finish",
    "dnf",
    "unfinished",
    "abandoned",
    "owned",
    "owned books",
    "books i own",
    "i own",
    "my books",
    "collection",
    "collections",
    "favorites",
    "favourites",
    "favorite",
    "favourite",
    "wish list",
    "wishlist",
    "to buy",
    "buy",
    "kindle",
    "ebook",
    "e book",
    "ebooks",
    "audio",
    "audiobook",
    "audiobooks",
    "audio book",
    "paperback",
    "hardcover",
    "hardback",
    "mass market paperback",
    "borrowed",
    "library books",
    "netgalley",
    "arc",
    "review copy",
    "review copies",
    "reread",
    "re read",
    "p",
}

GENRE_SHELF_ALIASES = {
    "fiction": "General Fiction",
    "general fiction": "General Fiction",
    "literary fiction": "Contemporary Fiction",
    "adult fiction": "Contemporary Fiction",
    "realistic fiction": "Contemporary Fiction",
    "contemporary": "Contemporary Fiction",
    "contemporary fiction": "Contemporary Fiction",
    "women s fiction": "Contemporary Fiction",
    "womens fiction": "Contemporary Fiction",
    "chick lit": "Romance",
    "chicklit": "Romance",
    "romance": "Romance",
    "historical romance": "Romance",
    "paranormal romance": "Romance",
    "fantasy": "Fantasy",
    "urban fantasy": "Fantasy",
    "high fantasy": "Fantasy",
    "epic fantasy": "Fantasy",
    "paranormal": "Fantasy",
    "magic": "Fantasy",
    "magical realism": "Magical Realism",
    "science fiction": "Science Fiction",
    "sci fi": "Science Fiction",
    "scifi": "Science Fiction",
    "sf": "Science Fiction",
    "dystopia": "Dystopian Fiction",
    "dystopian": "Dystopian Fiction",
    "dystopian fiction": "Dystopian Fiction",
    "mystery": "Mystery",
    "mysteries": "Mystery",
    "crime": "Mystery",
    "detective": "Mystery",
    "thriller": "Thriller",
    "thrillers": "Thriller",
    "suspense": "Thriller",
    "true crime": "True Crime",
    "non fiction": "Non-Fiction",
    "nonfiction": "Non-Fiction",
    "biography": "Biography",
    "biographies": "Biography",
    "autobiography": "Biography",
    "memoir": "Memoir",
    "memoirs": "Memoir",
    "business": "Business",
    "economics": "Business",
    "finance": "Business",
    "self help": "Self-Help",
    "self improvement": "Self-Help",
    "personal development": "Self-Help",
    "psychology": "Psychology",
    "philosophy": "Philosophy",
    "spiritual": "Philosophy",
    "spirituality": "Philosophy",
    "religion": "Philosophy",
    "theology": "Philosophy",
    "history": "Non-Fiction",
    "historical": "Historical Fiction",
    "historical fiction": "Historical Fiction",
    "classics": "Classic Fiction",
    "classic": "Classic Fiction",
    "classic literature": "Classic Fiction",
    "horror": "Horror",
    "young adult": "Young Adult",
    "ya": "Young Adult",
    "teen": "Young Adult",
    "juvenile": "Young Adult",
    "children s": "Young Adult",
    "childrens": "Young Adult",
    "middle grade": "Young Adult",
    "graphic novels": "Graphic Novels",
    "graphic novel": "Graphic Novels",
    "comics": "Graphic Novels",
    "comic books": "Graphic Novels",
    "manga": "Graphic Novels",
    "poetry": "Poetry",
    "poems": "Poetry",
    "science": "Science",
    "popular science": "Science",
    "education": "Education",
    "teaching": "Education",
    "cookbooks": "Cooking",
    "cooking": "Cooking",
    "food": "Cooking",
    "travel": "Travel",
    "art": "Art",
    "music": "Music",
    "health": "Health",
    "fitness": "Health",
}


def _canonical_genre_from_shelf(value: str) -> str:
    text = str(value).strip()
    if not text:
        return ""

    normalized = _normalize_recommendation_text(text.replace("&", " and ").replace("/", " "))
    normalized = re.sub(r"\s+", " ", normalized).strip()
    if not normalized or normalized in NOISY_SHELF_LABELS:
        return ""
    if normalized.isdigit() or re.fullmatch(r"(19|20)\d{2}", normalized):
        return ""
    if re.match(r"^(read|books read|read in|read for|read on)\s+(19|20)?\d{2}", normalized):
        return ""
    if len(normalized) < 3 and normalized not in GENRE_SHELF_ALIASES:
        return ""

    if normalized in GENRE_SHELF_ALIASES:
        return GENRE_SHELF_ALIASES[normalized]

    for shelf_label, canonical_genre in GENRE_SHELF_ALIASES.items():
        if len(shelf_label) >= 6 and shelf_label in normalized:
            return canonical_genre
    return ""


def _iter_text_candidates(value):
    if value is None:
        return

    if isinstance(value, dict):
        for key in ("name", "genre", "tag", "label"):
            candidate = value.get(key)
            if candidate:
                yield candidate
        return

    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                for key in ("name", "genre", "tag", "label"):
                    candidate = item.get(key)
                    if candidate:
                        yield candidate
                        break
            else:
                yield item
        return

    text = str(value).strip()
    for candidate in text.split(","):
        yield candidate


def _extract_genre_value(value, default: str = "General Fiction") -> str:
    broad_fallback = ""
    for candidate in _iter_text_candidates(value) or []:
        canonical_genre = _canonical_genre_from_shelf(candidate)
        if not canonical_genre:
            continue
        if canonical_genre == "General Fiction":
            broad_fallback = broad_fallback or canonical_genre
            continue
        return canonical_genre
    return broad_fallback or default


def _book_matches_mood(book_genre: str, target_genres: set[str]) -> bool:
    normalized_genre = _normalize_recommendation_text(book_genre)
    if not normalized_genre:
        return False

    for target_genre in target_genres:
        normalized_target = _normalize_recommendation_text(target_genre)
        if normalized_target in normalized_genre or normalized_genre in normalized_target:
            return True
    return False


def is_missing_author(author: str) -> bool:
    normalized = _normalize_recommendation_text(author)
    return normalized in {"", "unknown", "unknown author", "nan", "none", "n a", "na"}


def _clean_title_for_lookup(title: str) -> str:
    clean_title = str(title).strip()
    clean_title = re.sub(r"\s*\([^)]*#\d+[^)]*\)\s*$", "", clean_title).strip()
    return clean_title or str(title).strip()


def _title_lookup_key(title: str) -> str:
    normalized = _normalize_recommendation_text(_clean_title_for_lookup(title))
    return re.sub(r"[^a-z0-9]+", " ", normalized).strip()


KNOWN_AUTHOR_BY_TITLE = {
    "a game of thrones": "George R.R. Martin",
    "bridget jones s diary": "Helen Fielding",
    "divergent": "Veronica Roth",
    "eat pray love": "Elizabeth Gilbert",
    "gone girl": "Gillian Flynn",
    "harry potter and the chamber of secrets": "J.K. Rowling",
    "harry potter and the deathly hallows": "J.K. Rowling",
    "harry potter and the goblet of fire": "J.K. Rowling",
    "harry potter and the half blood prince": "J.K. Rowling",
    "harry potter and the order of the phoenix": "J.K. Rowling",
    "harry potter and the prisoner of azkaban": "J.K. Rowling",
    "harry potter and the sorcerer s stone": "J.K. Rowling",
    "life of pi": "Yann Martel",
    "the alchemist": "Paulo Coelho",
    "the book thief": "Markus Zusak",
    "the da vinci code": "Dan Brown",
    "the fault in our stars": "John Green",
    "the girl who kicked the hornet s nest": "Stieg Larsson",
    "the girl who played with fire": "Stieg Larsson",
    "the girl with the dragon tattoo": "Stieg Larsson",
    "the giver": "Lois Lowry",
    "the help": "Kathryn Stockett",
    "the hobbit": "J.R.R. Tolkien",
    "the hunger games": "Suzanne Collins",
    "the kite runner": "Khaled Hosseini",
    "the lord of the rings": "J.R.R. Tolkien",
    "the martian": "Andy Weir",
    "the name of the wind": "Patrick Rothfuss",
    "the road": "Cormac McCarthy",
    "twilight": "Stephenie Meyer",
    "unbroken a world war ii story of survival resilience and redemption": "Laura Hillenbrand",
    "water for elephants": "Sara Gruen",
}


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def lookup_author_from_openlibrary(title: str) -> str:
    clean_title = _clean_title_for_lookup(title)
    if not clean_title:
        return ""

    try:
        response = requests.get(
            "https://openlibrary.org/search.json",
            params={"title": clean_title, "limit": 3},
            timeout=1.25,
        )
        response.raise_for_status()
        for doc in response.json().get("docs", []):
            author_names = doc.get("author_name") or []
            if author_names:
                author_name = str(author_names[0]).strip()
                if author_name:
                    return author_name
    except Exception:
        pass
    return ""


def resolve_display_author(title: str, author: str) -> str:
    if not is_missing_author(author):
        return str(author).strip()
    known_author = KNOWN_AUTHOR_BY_TITLE.get(_title_lookup_key(title), "")
    if known_author:
        return known_author
    if os.environ.get("BRS_ENABLE_AUTHOR_LOOKUP", "0") != "1":
        return "Chưa rõ tác giả"
    enriched_author = lookup_author_from_openlibrary(title)
    return enriched_author or "Chưa rõ tác giả"


@st.cache_data(ttl=24 * 60 * 60, show_spinner=False)
def get_cover_image_url(title: str, genre: str, author: str = "", image_url: str = "") -> str:
    if image_url:
        image_url = str(image_url).strip()
        if image_url and image_url.lower() not in {"nan", "none"}:
            return image_url

    if os.environ.get("BRS_ENABLE_COVER_LOOKUP", "0") != "1":
        return build_cover_svg_data_uri(title, genre, author)

    query = {"title": title, "limit": 1}
    if author:
        query["author"] = author

    try:
        response = requests.get("https://openlibrary.org/search.json", params=query, timeout=2)
        response.raise_for_status()
        docs = response.json().get("docs", [])
        if docs:
            cover_id = docs[0].get("cover_i")
            if cover_id:
                return f"https://covers.openlibrary.org/b/id/{cover_id}-M.jpg"
    except Exception:
        pass

    return build_cover_svg_data_uri(title, genre, author)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Source+Serif+4:wght@600;700&display=swap');

    :root {
        --page-bg: #f7f3eb;
        --panel-bg: #ffffff;
        --panel-muted: #faf7f1;
        --text-dark: #1f2430;
        --text-light: #6c7280;
        --primary: {PRIMARY};
        --secondary: {SECONDARY};
        --border: rgba(31, 36, 48, 0.10);
        --shadow: 0 10px 26px rgba(31, 36, 48, 0.05);
        --shadow-soft: 0 6px 16px rgba(31, 36, 48, 0.035);
    }

    .stApp {
        background: var(--page-bg);
        color: var(--text-dark);
    }

    header[data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDeployButton"] {
        display: none !important;
    }

    .block-container {
        max-width: 1400px;
        padding-top: 1rem;
        padding-bottom: 2rem;
        padding-left: 1.25rem;
        padding-right: 1.25rem;
    }

    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #fefcf8 0%, #f7f0e5 100%);
        color: var(--text-dark);
        border-right: 1px solid rgba(31, 36, 48, 0.08);
    }

    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"],
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] div {
        color: var(--text-dark);
    }

    .hero-card {
        background: #ffffff;
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 1.35rem 1.4rem;
        box-shadow: var(--shadow);
        margin-bottom: 1rem;
    }

    .hero-kicker {
        display: inline-flex;
        align-items: center;
        gap: 0.45rem;
        padding: 0.38rem 0.7rem;
        border-radius: 999px;
        background: rgba(233, 69, 96, 0.08);
        color: var(--primary);
        font-size: 0.72rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 0.85rem;
    }

    .hero-title {
        font-family: 'Inter', sans-serif;
        font-size: clamp(2rem, 3vw, 3.1rem);
        line-height: 1.02;
        font-weight: 800;
        letter-spacing: -0.035em;
        color: var(--text-dark);
        margin: 0;
    }

    .hero-subtitle {
        margin-top: 0.8rem;
        max-width: 840px;
        color: var(--text-light);
        font-size: 0.98rem;
        line-height: 1.7;
    }

    .hero-grid {
        display: grid;
        grid-template-columns: minmax(0, 1fr);
        gap: 1.15rem;
    }

    .hero-rail {
        display: grid;
        gap: 0.85rem;
    }

    .hero-rail-label {
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--text-light);
    }

    .hero-featured-book {
        display: grid;
        grid-template-columns: 82px minmax(0, 1fr);
        gap: 0.75rem;
        align-items: center;
        padding: 0.6rem 0.65rem;
        border-radius: 14px;
        background: #ffffff;
        border: 1px solid rgba(31, 36, 48, 0.08);
        box-shadow: var(--shadow-soft);
    }

    .hero-featured-cover {
        width: 82px;
        height: 112px;
        border-radius: 10px;
        display: flex;
        align-items: flex-end;
        justify-content: flex-start;
        padding: 0.45rem;
        box-shadow: 0 8px 14px rgba(31, 36, 48, 0.08);
    }

    .hero-featured-mark {
        font-size: 0.8rem;
        font-weight: 800;
        line-height: 1;
        letter-spacing: -0.03em;
        color: rgba(31, 36, 48, 0.9);
    }

    .hero-featured-title {
        font-family: 'Source Serif 4', serif;
        font-size: 0.95rem;
        line-height: 1.2;
        font-weight: 700;
        color: var(--text-dark);
        margin-bottom: 0.18rem;
    }

    .hero-featured-author {
        font-size: 0.8rem;
        color: var(--text-light);
        margin-bottom: 0.35rem;
    }

    .hero-featured-chip {
        display: inline-flex;
        align-items: center;
        padding: 0.24rem 0.52rem;
        border-radius: 999px;
        background: rgba(31, 36, 48, 0.05);
        color: var(--text-dark);
        font-size: 0.72rem;
        font-weight: 600;
    }

    .hero-panel {
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        padding: 0.25rem 0 0;
        border-radius: 20px;
    }

    .hero-panel-label {
        font-size: 0.7rem;
        font-weight: 800;
        letter-spacing: 0.1em;
        text-transform: uppercase;
        color: var(--text-light);
        margin-bottom: 0.5rem;
    }

    .hero-panel-title {
        font-family: 'Source Serif 4', serif;
        font-size: clamp(1.2rem, 1.7vw, 1.55rem);
        line-height: 1.08;
        font-weight: 700;
        color: var(--text-dark);
        margin-bottom: 0.75rem;
    }

    .hero-panel-title-inline {
        margin-top: 1rem;
        margin-bottom: 0.75rem;
        font-size: 0.84rem;
        font-weight: 800;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        color: var(--text-light);
    }

    .hero-points {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.7rem;
    }

    @media (max-width: 900px) {
        .hero-points {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    .hero-point {
        padding: 0.76rem 0.8rem;
        border-radius: 18px;
        background: #ffffff;
        border: 1px solid rgba(31, 36, 48, 0.08);
    }

    .hero-point-value {
        font-size: 0.95rem;
        font-weight: 800;
        color: var(--text-dark);
        margin-bottom: 0.18rem;
    }

    .hero-point-label {
        font-size: 0.78rem;
        color: var(--text-light);
        line-height: 1.4;
    }

    .hero-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 0.95rem;
    }

    .hero-tag {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.38rem 0.68rem;
        border-radius: 999px;
        font-size: 0.76rem;
        font-weight: 700;
        background: rgba(255, 255, 255, 0.8);
        color: var(--text-dark);
        border: 1px solid rgba(31, 36, 48, 0.08);
    }

    .hero-tag-accent {
        background: linear-gradient(135deg, rgba(233, 69, 96, 0.10), rgba(241, 143, 1, 0.10));
        color: #7a2b22;
    }

    .section-title {
        font-family: 'Inter', sans-serif;
        font-size: 0.92rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.12em;
        color: var(--text-light);
        margin: 1.15rem 0 0.9rem;
        display: flex;
        align-items: center;
        gap: 0.75rem;
    }

    .section-title::before {
        content: "";
        width: 32px;
        height: 2px;
        border-radius: 999px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
        flex: 0 0 auto;
    }

    .section-card {
        padding: 0.95rem 1rem;
        border-radius: 18px;
        background: #ffffff;
        border: 1px solid rgba(31, 36, 48, 0.08);
        box-shadow: var(--shadow-soft);
    }

    .sidebar-card,
    .panel-card,
    .book-card,
    .metric-card,
    .info-box {
        border: 1px solid var(--border);
        border-radius: 20px;
        background: var(--panel-bg);
        box-shadow: var(--shadow-soft);
    }

    .panel-card {
        padding: 1rem 1.05rem;
        position: relative;
        overflow: hidden;
        transition: border-color 0.2s ease, box-shadow 0.2s ease;
    }

    .panel-card::before {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: 18px;
        pointer-events: none;
        background: linear-gradient(135deg, rgba(233, 69, 96, 0.04), rgba(241, 143, 1, 0.03));
        opacity: 0;
        transition: opacity 0.2s ease;
    }

    .panel-card:hover {
        border-color: rgba(233, 69, 96, 0.14);
        box-shadow: 0 12px 24px rgba(31, 36, 48, 0.06);
    }

    .panel-card:hover::before {
        opacity: 1;
    }

    .panel-card h4 {
        position: relative;
        z-index: 1;
        margin-top: 0;
        margin-bottom: 0.45rem;
    }

    .panel-card p {
        position: relative;
        z-index: 1;
        margin: 0;
        color: var(--text-dark);
        line-height: 1.72;
    }

    .model-explain-panel {
        border: 1px solid var(--border);
        border-radius: 8px;
        background: #ffffff;
        box-shadow: var(--shadow-soft);
        padding: 1rem;
        margin: 0.75rem 0 1rem;
    }

    .model-explain-grid {
        display: grid;
        grid-template-columns: minmax(0, 0.9fr) minmax(0, 1.1fr);
        gap: 1rem;
        align-items: start;
    }

    .model-explain-title {
        font-size: 0.82rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: var(--text-light);
        margin-bottom: 0.55rem;
    }

    .explain-chip-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin: 0.45rem 0 0.75rem;
    }

    .explain-chip {
        display: inline-flex;
        align-items: center;
        min-height: 28px;
        padding: 0.28rem 0.58rem;
        border-radius: 999px;
        background: rgba(14, 124, 123, 0.10);
        color: #0e5f5e;
        font-size: 0.74rem;
        font-weight: 700;
    }

    .factor-row {
        display: grid;
        grid-template-columns: 132px minmax(0, 1fr) 48px;
        gap: 0.6rem;
        align-items: center;
        margin-bottom: 0.5rem;
        color: var(--text-dark);
        font-size: 0.82rem;
        font-weight: 700;
    }

    .factor-track {
        height: 8px;
        border-radius: 999px;
        background: rgba(20, 24, 32, 0.08);
        overflow: hidden;
    }

    .factor-fill {
        height: 100%;
        border-radius: inherit;
        background: linear-gradient(90deg, var(--secondary), var(--primary), var(--accent));
    }

    .model-explain-note {
        color: var(--text-light);
        font-size: 0.82rem;
        line-height: 1.58;
        margin-top: 0.45rem;
    }

    @media (max-width: 860px) {
        .model-explain-grid {
            grid-template-columns: 1fr;
        }

        .factor-row {
            grid-template-columns: 112px minmax(0, 1fr) 44px;
        }
    }

    .sidebar-card {
        padding: 0.9rem 0.95rem;
        margin-bottom: 0.85rem;
    }

    .sidebar-card h4 {
        margin: 0 0 0.55rem 0;
        font-size: 0.72rem;
        font-weight: 800;
        color: var(--text-dark);
        text-transform: uppercase;
        letter-spacing: 0.11em;
    }

    .sidebar-card p {
        margin: 0.28rem 0;
        font-size: 0.9rem;
        color: var(--text-dark);
        line-height: 1.55;
    }

    .sidebar-card a {
        color: var(--primary);
        text-decoration: none;
        font-weight: 600;
    }

    .sidebar-card a:hover {
        text-decoration: underline;
    }

    .metric-card {
        padding: 0.95rem 0.95rem 1rem;
        text-align: left;
        position: relative;
        overflow: hidden;
        min-height: 128px;
    }

    .metric-card::before {
        content: "";
        position: absolute;
        left: 0;
        right: 0;
        top: 0;
        height: 6px;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
    }

    .metric-icon {
        width: 2rem;
        height: 2rem;
        display: inline-flex;
        align-items: center;
        justify-content: center;
        border-radius: 999px;
        background: rgba(233, 69, 96, 0.08);
        margin-bottom: 0.7rem;
        font-size: 1rem;
    }

    .metric-icon:empty {
        display: none;
    }

    .metric-value {
        margin-top: 0;
        font-size: 1.55rem;
        line-height: 1.14;
        font-weight: 700;
        color: var(--text-dark);
        min-height: 2.2em;
        overflow-wrap: break-word;
        word-break: normal;
        white-space: normal;
    }

    .metric-label {
        margin-top: 0.35rem;
        font-size: 0.82rem;
        color: var(--text-light);
    }

    .info-box {
        padding: 0.9rem 1rem;
        margin: 0.9rem 0 1rem;
        background: rgba(233, 69, 96, 0.05);
    }

    .book-card {
        padding: 0.95rem;
        margin-bottom: 0.9rem;
        transition: border-color 0.18s ease, box-shadow 0.18s ease;
        position: relative;
        overflow: hidden;
    }

    .book-card:hover {
        box-shadow: 0 12px 24px rgba(31, 36, 48, 0.06);
        border-color: rgba(233, 69, 96, 0.12);
    }

    .book-card::after {
        content: "";
        position: absolute;
        inset: 0 auto 0 0;
        width: 3px;
        background: linear-gradient(180deg, rgba(233, 69, 96, 0.55), rgba(241, 143, 1, 0.55));
    }

    .book-card-header {
        display: flex;
        justify-content: space-between;
        gap: 1rem;
        align-items: flex-start;
        margin-bottom: 0.7rem;
    }

    .book-card-top {
        display: grid;
        grid-template-columns: 96px minmax(0, 1fr);
        gap: 0.85rem;
        align-items: start;
        padding-left: 0.25rem;
    }

    .book-cover {
        width: 96px;
        height: 134px;
        border-radius: 12px;
        border: 1px solid rgba(31, 36, 48, 0.08);
        display: flex;
        align-items: flex-end;
        justify-content: flex-start;
        padding: 0.65rem;
        box-shadow: 0 8px 14px rgba(31, 36, 48, 0.08);
        position: relative;
        overflow: hidden;
        background-size: cover;
        background-position: center;
    }

    .book-cover::before {
        content: "";
        position: absolute;
        inset: 0;
        background: linear-gradient(180deg, rgba(255, 255, 255, 0.06), rgba(31, 36, 48, 0.10) 55%, rgba(31, 36, 48, 0.18));
    }

    .book-cover-index {
        position: relative;
        z-index: 1;
        font-size: 0.82rem;
        line-height: 1;
        font-weight: 800;
        color: white;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        text-shadow: 0 1px 2px rgba(0, 0, 0, 0.18);
    }

    .book-rank {
        min-width: 52px;
        padding: 0.34rem 0.68rem;
        border-radius: 999px;
        background: rgba(31, 36, 48, 0.05);
        color: var(--text-dark);
        text-align: center;
        font-size: 0.8rem;
        font-weight: 700;
    }

    .book-title {
        font-family: 'Source Serif 4', serif;
        font-size: 1.08rem;
        line-height: 1.25;
        font-weight: 700;
        color: var(--text-dark);
        margin-bottom: 0.3rem;
        letter-spacing: -0.01em;
    }

    .book-meta {
        font-size: 0.86rem;
        color: var(--text-light);
    }

    .book-tags {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.68rem;
    }

    .badge {
        display: inline-flex;
        align-items: center;
        border-radius: 999px;
        padding: 0.28rem 0.64rem;
        font-size: 0.74rem;
        font-weight: 600;
        white-space: nowrap;
    }

    .badge-primary {
        background: rgba(233, 69, 96, 0.08);
        color: var(--primary);
    }

    .badge-secondary {
        background: rgba(241, 143, 1, 0.08);
        color: var(--secondary);
    }

    .badge-neutral {
        background: rgba(35, 38, 47, 0.04);
        color: var(--text-dark);
    }

    .badge-success {
        background: rgba(10, 160, 105, 0.11);
        color: #0f766e;
    }

    .book-footer {
        margin-top: 0.75rem;
        display: flex;
        justify-content: space-between;
        align-items: center;
        gap: 1rem;
    }

    .book-reason {
        color: var(--text-light);
        font-size: 0.84rem;
        line-height: 1.5;
    }

    .feedback-panel {
        padding: 0.75rem 0.85rem;
        margin: -0.3rem 0 0.85rem;
        border: 1px solid rgba(31, 36, 48, 0.08);
        border-radius: 12px;
        background: rgba(255, 255, 255, 0.72);
    }

    .feedback-summary {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        align-items: center;
        margin: 0.55rem 0 0.85rem;
    }

    .feedback-status {
        color: var(--text-light);
        font-size: 0.8rem;
        line-height: 1.45;
        margin-top: 0.45rem;
    }

    .recommendation-action-caption {
        color: var(--text-light);
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        padding-top: 0.28rem;
    }

    div[data-testid="stFeedback"] {
        margin-top: -0.12rem;
        margin-bottom: 0.1rem;
    }

    .book-explain-panel {
        padding: 0.78rem 0.85rem;
        border: 1px solid rgba(31, 36, 48, 0.08);
        border-radius: 12px;
        background: #ffffff;
    }

    .book-explain-heading {
        font-weight: 800;
        color: var(--text-dark);
        margin-bottom: 0.55rem;
    }

    .mini-factor-row {
        display: grid;
        grid-template-columns: 128px minmax(0, 1fr) 46px;
        gap: 0.65rem;
        align-items: center;
        margin: 0.42rem 0;
        font-size: 0.82rem;
        color: var(--text-dark);
    }

    .mini-factor-track {
        height: 8px;
        border-radius: 999px;
        overflow: hidden;
        background: rgba(31, 36, 48, 0.08);
    }

    .mini-factor-fill {
        height: 100%;
        border-radius: inherit;
        background: linear-gradient(90deg, var(--primary), var(--secondary));
    }

    .book-explain-detail {
        color: var(--text-light);
        font-size: 0.8rem;
        line-height: 1.52;
        margin-top: 0.55rem;
    }

    .stRadio,
    .stRadio label,
    .stRadio p,
    .stRadio span,
    .stRadio div,
    div[data-testid="stRadio"],
    div[data-testid="stRadio"] label,
    div[data-testid="stRadio"] p,
    div[data-testid="stRadio"] span,
    div[data-testid="stRadio"] div {
        color: #111111 !important;
    }

    .stRadio > label,
    div[data-testid="stRadio"] > label,
    div[data-testid="stRadio"] [data-testid="stMarkdownContainer"] {
        font-weight: 700 !important;
        color: #111111 !important;
    }

    div[data-testid="stRadio"] div[role="radiogroup"] {
        gap: 0.4rem;
    }

    div[data-baseweb="radio"] input {
        accent-color: var(--primary);
    }

    .score-pill {
        flex: 0 0 auto;
        border-radius: 999px;
        padding: 0.38rem 0.72rem;
        background: rgba(233, 69, 96, 0.08);
        color: var(--text-dark);
        font-size: 0.8rem;
        font-weight: 700;
    }

    /* Force Streamlit tables to use the default black text color. */
    table[data-testid="stTableStyledTable"],
    table[data-testid="stTableStyledTable"] * {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }

    table[data-testid="stTableStyledTable"] th,
    table[data-testid="stTableStyledTable"] td,
    table[data-testid="stTableStyledTable"] p,
    table[data-testid="stTableStyledTable"] span,
    table[data-testid="stTableStyledTable"] div {
        color: #000000 !important;
        -webkit-text-fill-color: #000000 !important;
    }

    /* Keep dropdowns/select boxes in a clear light mode. */
    div[data-testid="stSelectbox"],
    div[data-testid="stMultiSelect"],
    div[data-testid="stSelectbox"] *,
    div[data-testid="stMultiSelect"] * {
        color: #111111 !important;
    }

    div[data-baseweb="select"],
    div[data-baseweb="select"] * {
        color: #111111 !important;
    }

    div[data-baseweb="select"] > div,
    div[data-baseweb="select"] input,
    div[data-baseweb="select"] [role="combobox"],
    div[data-baseweb="select"] [aria-haspopup="listbox"] {
        background-color: #ffffff !important;
        color: #111111 !important;
        -webkit-text-fill-color: #111111 !important;
        border-color: rgba(35, 38, 47, 0.14) !important;
    }

    div[data-baseweb="popover"] [role="listbox"],
    div[data-baseweb="menu"],
    div[data-baseweb="menu"] ul,
    div[data-baseweb="menu"] li,
    div[data-baseweb="menu"] [role="option"],
    div[data-baseweb="popover"] [role="option"] {
        background-color: #ffffff !important;
        color: #111111 !important;
    }

    div[data-baseweb="menu"] [aria-selected="true"],
    div[data-baseweb="popover"] [aria-selected="true"] {
        background-color: rgba(233, 69, 96, 0.10) !important;
        color: #111111 !important;
    }

    div[data-baseweb="select"] svg,
    div[data-baseweb="select"] path {
        fill: #111111 !important;
        color: #111111 !important;
    }

    .tabs-shell {
        border-radius: 16px;
    }

    .stTabs [data-baseweb="tab-list"] {
        gap: 0.5rem;
        padding: 0.55rem;
        border-radius: 16px;
        background: #ffffff;
        border: 1px solid var(--border);
        box-shadow: 0 8px 18px rgba(15, 23, 42, 0.04);
    }

    .stTabs [data-baseweb="tab"] {
        border-radius: 999px;
        background: #f3f0ea;
        border: 1px solid rgba(35, 38, 47, 0.08);
        color: var(--text-dark) !important;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        text-align: center;
        padding: 0.68rem 0.95rem !important;
        transition: all 0.18s ease;
    }

    /* On hover, add a subtle glow and lift; do NOT force a white background so
       the active gradient remains visible. This makes the tab slightly glow
       whether it's active or not. */
    .stTabs [data-baseweb="tab"]:hover {
        box-shadow: 0 10px 20px rgba(233, 69, 96, 0.08);
        transform: translateY(-1px);
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }

    .stTabs [data-baseweb="tab"] [data-testid="stMarkdownContainer"],
    .stTabs [data-baseweb="tab"] p,
    .stTabs [data-baseweb="tab"] span {
        font-weight: 700 !important;
        color: inherit !important;
    }

    .stTabs [aria-selected="true"] {
        background: rgba(233, 69, 96, 0.10);
        color: var(--text-dark) !important;
        border-color: transparent;
        box-shadow: 0 10px 18px rgba(233, 69, 96, 0.14);
    }

    .stTabs [aria-selected="true"] * {
        color: var(--text-dark) !important;
        font-weight: 800 !important;
    }

    .stButton>button {
        background: linear-gradient(135deg, var(--primary) 0%, var(--secondary) 100%);
        color: white;
        border: none;
        border-radius: 999px;
        padding: 0.7rem 1.15rem;
        font-family: 'Inter', sans-serif;
        font-weight: 700;
        font-size: 0.94rem;
        transition: all 0.2s ease;
        box-shadow: 0 10px 20px rgba(233, 69, 96, 0.22);
        white-space: nowrap !important;
        width: auto !important;
    }

    .stButton>button:hover {
        transform: translateY(-1px);
        box-shadow: 0 12px 22px rgba(233, 69, 96, 0.20);
    }

    .stButton>button p {
        white-space: nowrap !important;
    }

    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3, .stMarkdown h4 {
        color: var(--text-dark);
    }

    .project-note {
        font-size: 0.76rem;
        font-weight: 800;
        letter-spacing: 0.12em;
        text-transform: uppercase;
        color: var(--primary);
        margin-bottom: 0.55rem;
    }

    .stPlotlyChart text,
    .stPlotlyChart .legendtext,
    .stPlotlyChart .xtick text,
    .stPlotlyChart .ytick text,
    [data-testid="stPlotlyChart"] text,
    [data-testid="stPlotlyChart"] .legendtext,
    [data-testid="stPlotlyChart"] .xtick text,
    [data-testid="stPlotlyChart"] .ytick text {
        fill: #000000 !important;
        color: #000000 !important;
    }

    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""".replace("{PRIMARY}", COLORS["primary"]).replace("{SECONDARY}", COLORS["secondary"]), unsafe_allow_html=True)

st.markdown(
    """
<style>
    * {
        letter-spacing: 0 !important;
    }

    :root {
        --page-bg: #f6f7f4;
        --panel-bg: #ffffff;
        --panel-muted: #f2f5f3;
        --text-dark: #141820;
        --text-light: #5f6b7a;
        --primary: #d84c3f;
        --secondary: #0e7c7b;
        --accent: #d6a11f;
        --border: rgba(20, 24, 32, 0.12);
        --shadow: 0 14px 38px rgba(20, 24, 32, 0.10);
        --shadow-soft: 0 8px 22px rgba(20, 24, 32, 0.07);
    }

    .stApp {
        background: linear-gradient(180deg, #f6f7f4 0%, #eef3f0 100%);
    }

    .block-container {
        max-width: 1320px;
        padding-top: 0.85rem;
        padding-left: 1rem;
        padding-right: 1rem;
    }

    section[data-testid="stSidebar"] {
        background: #f9faf7;
        border-right: 1px solid rgba(20, 24, 32, 0.12);
    }

    .sidebar-card,
    .panel-card,
    .book-card,
    .metric-card,
    .info-box,
    .hero-card,
    .hero-featured-book,
    .hero-featured-cover,
    .book-cover,
    .stTabs [data-baseweb="tab-list"],
    .stTabs [data-baseweb="tab"],
    div[data-baseweb="select"] > div,
    .stButton > button {
        border-radius: 8px !important;
    }

    .sidebar-card {
        padding: 0.88rem 0.95rem;
        margin-bottom: 0.72rem;
        box-shadow: none;
    }

    .sidebar-card h4 {
        font-size: 0.78rem;
        margin-bottom: 0.5rem;
    }

    .sidebar-card p {
        font-size: 0.86rem;
        line-height: 1.5;
    }

    .hero-card {
        display: grid;
        grid-template-columns: minmax(0, 1.08fr) minmax(340px, 0.92fr);
        gap: 1rem;
        align-items: stretch;
        padding: 1.1rem;
        margin-bottom: 0.9rem;
        background: linear-gradient(135deg, #142625 0%, #1f4a49 54%, #f2e5c4 100%);
        border: 1px solid rgba(255, 255, 255, 0.26);
        box-shadow: var(--shadow);
        overflow: hidden;
    }

    .hero-grid {
        display: block;
    }

    .hero-kicker {
        width: fit-content;
        background: rgba(255, 255, 255, 0.16);
        color: #ffffff;
        border: 1px solid rgba(255, 255, 255, 0.22);
        margin-bottom: 0.95rem;
    }

    .hero-title {
        color: #ffffff !important;
        font-size: 3.15rem;
        line-height: 1.02;
        max-width: 780px;
    }

    .hero-title span,
    .hero-card h1,
    .hero-card h1 span {
        color: #ffffff !important;
    }

    .hero-subtitle {
        color: rgba(255, 255, 255, 0.82);
        max-width: 760px;
        font-size: 0.98rem;
        margin-top: 0.9rem;
    }

    .hero-points {
        grid-template-columns: repeat(4, minmax(0, 1fr));
        margin-top: 1.15rem;
    }

    .hero-point {
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.18);
        color: #ffffff;
    }

    .hero-point-value,
    .hero-point-label {
        color: #ffffff;
    }

    .hero-point-label {
        opacity: 0.76;
    }

    .hero-panel {
        padding: 0;
        justify-content: center;
    }

    .hero-panel-title-inline {
        color: rgba(255, 255, 255, 0.82);
        margin-top: 0;
        margin-bottom: 0.7rem;
    }

    .hero-rail {
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.7rem;
    }

    .hero-featured-book {
        grid-template-columns: 64px minmax(0, 1fr);
        gap: 0.7rem;
        min-height: 104px;
        padding: 0.58rem;
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid rgba(255, 255, 255, 0.58);
        box-shadow: 0 10px 24px rgba(20, 24, 32, 0.14);
    }

    .hero-featured-cover {
        width: 64px;
        height: 92px;
        padding: 0;
        overflow: hidden;
        background: #e7ece8;
    }

    .hero-featured-cover img,
    .book-cover img {
        width: 100%;
        height: 100%;
        object-fit: cover;
        display: block;
    }

    .hero-featured-mark {
        display: none;
    }

    .hero-featured-title {
        font-size: 0.93rem;
        color: var(--text-dark);
    }

    .hero-featured-author {
        color: var(--text-light);
    }

    .hero-featured-chip {
        background: rgba(14, 124, 123, 0.10);
        color: #0e5f5e;
    }

    .stTabs [data-baseweb="tab-list"] {
        padding: 0.45rem;
        background: rgba(255, 255, 255, 0.84);
        backdrop-filter: blur(10px);
    }

    .stTabs [data-baseweb="tab"] {
        background: transparent;
        border: 1px solid transparent;
        color: var(--text-dark) !important;
        padding: 0.62rem 0.82rem !important;
    }

    .stTabs [aria-selected="true"] {
        background: #ffffff;
        border-color: rgba(216, 76, 63, 0.24);
        box-shadow: 0 8px 18px rgba(20, 24, 32, 0.08);
    }

    .section-title {
        color: #4a5565;
        font-size: 0.95rem;
        margin-top: 1.05rem;
    }

    .metric-card {
        min-height: 128px;
        padding: 0.92rem;
        box-shadow: var(--shadow-soft);
    }

    .metric-card::before {
        height: 4px;
        background: linear-gradient(90deg, var(--secondary), var(--primary), var(--accent));
    }

    .metric-value {
        font-size: 1.5rem;
        line-height: 1.16;
        min-height: 2.25em;
        white-space: normal;
    }

    .book-card {
        padding: 1rem;
        margin-bottom: 0.86rem;
        border-color: rgba(20, 24, 32, 0.10);
        box-shadow: var(--shadow-soft);
    }

    .book-card::after {
        display: none;
    }

    .book-card:hover {
        transform: translateY(-1px);
        box-shadow: 0 18px 34px rgba(20, 24, 32, 0.12);
        border-color: rgba(14, 124, 123, 0.24);
    }

    .book-card-top {
        grid-template-columns: 112px minmax(0, 1fr);
        gap: 1rem;
        padding-left: 0;
    }

    .book-cover {
        width: 112px;
        height: 168px;
        padding: 0;
        background: #e8ece8;
        box-shadow: 0 16px 28px rgba(20, 24, 32, 0.18);
    }

    .book-cover::before {
        display: none;
    }

    .book-rank-badge {
        position: absolute;
        top: 0.45rem;
        left: 0.45rem;
        z-index: 2;
        min-width: 34px;
        padding: 0.18rem 0.38rem;
        border-radius: 999px;
        background: rgba(20, 24, 32, 0.78);
        color: #ffffff;
        font-size: 0.72rem;
        font-weight: 800;
        text-align: center;
    }

    .book-cover-index {
        display: none;
    }

    .book-card-header {
        align-items: center;
        margin-bottom: 0.55rem;
    }

    .book-rank {
        display: none;
    }

    .book-title {
        font-size: 1.18rem;
        line-height: 1.24;
        margin-bottom: 0.35rem;
    }

    .book-meta {
        color: var(--text-light);
        font-size: 0.86rem;
    }

    .badge,
    .score-pill {
        border-radius: 999px !important;
        font-size: 0.74rem;
    }

    .badge-primary {
        background: rgba(14, 124, 123, 0.10);
        color: #0e5f5e;
    }

    .badge-secondary {
        background: rgba(214, 161, 31, 0.13);
        color: #8a6400;
    }

    .score-pill {
        background: rgba(216, 76, 63, 0.10);
        color: #9d2d24;
    }

    .book-footer {
        margin-top: 0.85rem;
        padding-top: 0.78rem;
        border-top: 1px solid rgba(20, 24, 32, 0.08);
    }

    .stButton > button {
        background: linear-gradient(135deg, var(--secondary), var(--primary));
        box-shadow: 0 12px 22px rgba(14, 124, 123, 0.18);
    }

    @media (max-width: 980px) {
        .hero-card {
            grid-template-columns: 1fr;
        }

        .hero-title {
            font-size: 2.25rem;
        }

        .hero-points {
            grid-template-columns: repeat(2, minmax(0, 1fr));
        }
    }

    @media (max-width: 720px) {
        .hero-rail {
            grid-template-columns: 1fr;
        }

        .book-card-top {
            grid-template-columns: 88px minmax(0, 1fr);
        }

        .book-cover {
            width: 88px;
            height: 132px;
        }
    }
</style>
    """,
    unsafe_allow_html=True,
)


def render_sidebar_card(title: str, body_html: str) -> None:
    st.markdown(
        f"""
        <div class="sidebar-card">
            <h4>{title}</h4>
            {body_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_metric_card(value: str, label: str, icon: str = ""):
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-icon">{icon}</div>
            <div class="metric-value">{value}</div>
            <div class="metric-label">{label}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def display_book_card(rec: BookRecommendation, rank: int):
    bestseller_badge = '<span class="badge badge-success">Phổ biến</span>' if rec.bestseller else ""
    display_author = resolve_display_author(rec.title, rec.author or "")
    cover_url = get_cover_image_url(rec.title, rec.genre, display_author, rec.image_url or "")
    fallback_cover_url = build_cover_svg_data_uri(rec.title, rec.genre, display_author)
    title = escape_html(rec.title)
    author = escape_html(display_author)
    genre = escape_html(translate_genre(rec.genre))
    reason = escape_html(translate_reason(rec.reason))
    cover_url_attr = escape_html(cover_url)
    fallback_cover_attr = escape_html(fallback_cover_url)
    st.markdown(
        f"""
        <div class="book-card">
            <div class="book-card-top">
                <div class="book-cover">
                    <img src="{cover_url_attr}" alt="Bìa sách {title}" loading="lazy" onerror="this.onerror=null;this.src='{fallback_cover_attr}';">
                    <div class="book-rank-badge">#{rank}</div>
                </div>
                <div>
                    <div class="book-card-header">
                        <div class="badge badge-primary">{genre}</div>
                        <div class="score-pill">{rec.score:.0%} phù hợp</div>
                    </div>
                    <div class="book-title">{title} {bestseller_badge}</div>
                    <div class="book-meta">Tác giả: {author} · {int(rec.year)}</div>
                    <div class="book-tags">
                        <span class="badge badge-secondary">{rec.rating:.2f}/5</span>
                        <span class="badge badge-neutral">{format_number(rec.ratings_count)} lượt đánh giá</span>
                    </div>
                    <div class="book-footer">
                        <div class="book-reason">{reason}</div>
                        <div class="score-pill">{rec.score:.0%}</div>
                    </div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def stable_component_key(prefix: str, *parts: object) -> str:
    raw = "|".join(str(part) for part in parts)
    encoded = base64.urlsafe_b64encode(raw.encode("utf-8")).decode("ascii").rstrip("=")
    return f"{prefix}_{encoded[:36]}"


def recommendation_identity(rec: BookRecommendation) -> str:
    if rec.book_id:
        return str(rec.book_id)
    return f"{rec.title}|{rec.author}"


def lookup_book_label(book_key: str, books_df: pd.DataFrame) -> str:
    key = str(book_key)
    if "book_id" in books_df.columns:
        matches = books_df[books_df["book_id"].astype(str) == key]
        if not matches.empty:
            row = matches.iloc[0]
            return f"{row['title']} - {row['author']}"
    if "|" in key:
        title, author = key.split("|", 1)
        return f"{title} - {author}"
    return key


def _book_rows_for_ids(books_df: pd.DataFrame, book_ids: set[str]) -> pd.DataFrame:
    if not book_ids or "book_id" not in books_df.columns:
        return books_df.iloc[0:0].copy()
    catalog = books_df.copy()
    catalog["book_id"] = catalog["book_id"].astype(str)
    return catalog[catalog["book_id"].isin(book_ids)].copy()


def build_feedback_profile(
    feedback_votes: dict | None,
    saved_items: list | None,
    books_df: pd.DataFrame,
) -> dict:
    votes = {str(key): value for key, value in (feedback_votes or {}).items()}
    saved_ids = {str(item) for item in (saved_items or [])}
    liked_ids = {key for key, value in votes.items() if value == "liked"}
    disliked_ids = {key for key, value in votes.items() if value == "disliked"}
    positive_ids = liked_ids | saved_ids

    positive_rows = _book_rows_for_ids(books_df, positive_ids)
    negative_rows = _book_rows_for_ids(books_df, disliked_ids)
    return {
        "liked_ids": liked_ids,
        "disliked_ids": disliked_ids,
        "saved_ids": saved_ids,
        "positive_ids": positive_ids,
        "excluded_ids": positive_ids | disliked_ids,
        "positive_genres": set(positive_rows["genre"].dropna().astype(str)) if not positive_rows.empty else set(),
        "positive_authors": set(positive_rows["author"].dropna().astype(str)) if not positive_rows.empty else set(),
        "negative_genres": set(negative_rows["genre"].dropna().astype(str)) if not negative_rows.empty else set(),
        "negative_authors": set(negative_rows["author"].dropna().astype(str)) if not negative_rows.empty else set(),
    }


def record_recommendation_feedback(rec: BookRecommendation, action: str) -> None:
    book_key = recommendation_identity(rec)
    if action in {"liked", "disliked"}:
        st.session_state.feedback_votes[book_key] = action
    elif action == "saved" and book_key not in st.session_state.reading_shelf:
        st.session_state.reading_shelf.append(book_key)

    st.session_state.feedback_log.insert(
        0,
        {
            "book_id": book_key,
            "title": rec.title,
            "author": rec.author,
            "action": action,
        },
    )
    st.session_state.feedback_log = st.session_state.feedback_log[:30]


def render_feedback_summary(books_df: pd.DataFrame) -> None:
    votes = st.session_state.feedback_votes
    shelf = st.session_state.reading_shelf
    if not votes and not shelf:
        return

    liked_count = sum(1 for value in votes.values() if value == "liked")
    disliked_count = sum(1 for value in votes.values() if value == "disliked")
    saved_count = len(shelf)
    latest = st.session_state.feedback_log[:3]
    latest_html = "".join(
        f"<span class='badge badge-neutral'>{escape_html(item['title'])}</span>"
        for item in latest
    )
    saved_html = "".join(
        f"<span class='badge badge-secondary'>{escape_html(lookup_book_label(book_key, books_df))}</span>"
        for book_key in shelf[:4]
    )

    st.markdown(
        f"""
        <div class="info-box">
            <strong>Phản hồi tức thời đang được dùng để xếp hạng lại danh sách.</strong>
            <div class="feedback-summary">
                <span class="badge badge-success">Đã thích: {liked_count}</span>
                <span class="badge badge-primary">Không hợp: {disliked_count}</span>
                <span class="badge badge-secondary">Đã lưu: {saved_count}</span>
            </div>
            <div class="feedback-status">Tín hiệu gần nhất: {latest_html or "chưa có"}</div>
            <div class="feedback-status">Kệ đọc: {saved_html or "chưa lưu sách nào"}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Xóa phản hồi phiên này", key="clear_feedback_state"):
        st.session_state.feedback_votes = {}
        st.session_state.reading_shelf = []
        st.session_state.feedback_log = []
        st.rerun()


def render_recommendation_actions(rec: BookRecommendation, key_prefix: str) -> None:
    book_key = recommendation_identity(rec)
    vote = st.session_state.feedback_votes.get(book_key)
    saved = book_key in st.session_state.reading_shelf

    action_cols = st.columns([0.86, 0.92, 1.55, 1.6], gap="small", vertical_alignment="center")
    with action_cols[0]:
        st.markdown("<div class='recommendation-action-caption'>Phản hồi</div>", unsafe_allow_html=True)
    with action_cols[1]:
        feedback_default = {"disliked": 0, "liked": 1}.get(vote)
        feedback_value = st.feedback(
            "thumbs",
            key=stable_component_key("feedback", key_prefix, book_key),
            default=feedback_default,
            width="content",
        )
        if feedback_value is None and vote in {"liked", "disliked"}:
            st.session_state.feedback_votes.pop(book_key, None)
            st.rerun()
        if feedback_value is not None:
            next_vote = "liked" if feedback_value == 1 else "disliked"
            if vote != next_vote:
                record_recommendation_feedback(rec, next_vote)
                st.rerun()
    with action_cols[2]:
        saved_value = st.toggle(
            "Lưu vào kệ",
            value=saved,
            key=stable_component_key("save_toggle", key_prefix, book_key),
            width="content",
        )
        if saved_value and not saved:
            record_recommendation_feedback(rec, "saved")
            st.rerun()
        if not saved_value and saved:
            st.session_state.reading_shelf = [
                item for item in st.session_state.reading_shelf if str(item) != book_key
            ]
            st.rerun()

    status_parts = []
    if vote == "liked":
        status_parts.append("Đã ghi nhận: thích sách này")
    elif vote == "disliked":
        status_parts.append("Đã ghi nhận: sách này không hợp")
    if saved:
        status_parts.append("đã lưu vào kệ đọc")
    if status_parts:
        st.markdown(
            f"<div class='feedback-status'>{escape_html('; '.join(status_parts))}.</div>",
            unsafe_allow_html=True,
        )


def build_recommendation_explanation_factors(
    rec: BookRecommendation,
    user_id: str | None,
    ratings_df: pd.DataFrame,
    books_df: pd.DataFrame,
) -> tuple[list[tuple[str, float]], list[str]]:
    max_popularity = float(np.log1p(pd.to_numeric(books_df["ratings_count"], errors="coerce").fillna(0)).max())
    popularity = np.log1p(max(0, rec.ratings_count)) / max_popularity if max_popularity > 0 else 0.0
    novelty = max(0.0, 1.0 - popularity)
    rating_norm = min(1.0, max(0.0, rec.rating / 5.0))
    genre_signal = 0.0
    author_signal = 0.0
    details = [translate_reason(rec.reason)]

    if user_id:
        user_ratings = ratings_df[ratings_df["user_id"].astype(str) == str(user_id)].copy()
        if not user_ratings.empty:
            catalog = books_df[["book_id", "genre", "author", "title"]].copy()
            catalog["book_id"] = catalog["book_id"].astype(str)
            history = user_ratings.merge(catalog, on="book_id", how="left").dropna(subset=["genre"])
            if not history.empty:
                history["preference"] = (pd.to_numeric(history["rating"], errors="coerce").fillna(0) / 5.0).clip(0, 1)
                genre_mean = history.loc[history["genre"] == rec.genre, "preference"].mean()
                author_mean = history.loc[history["author"] == rec.author, "preference"].mean()
                genre_signal = float(genre_mean) if pd.notna(genre_mean) else 0.0
                author_signal = float(author_mean) if pd.notna(author_mean) else 0.0
                if genre_signal > 0:
                    details.append(f"Thể loại {translate_genre(rec.genre)} từng xuất hiện trong lịch sử đánh giá của hồ sơ này.")
                if author_signal > 0:
                    details.append(f"Tác giả {rec.author} có tín hiệu từ lịch sử đọc.")

    feedback_profile = build_feedback_profile(
        st.session_state.get("feedback_votes", {}),
        st.session_state.get("reading_shelf", []),
        books_df,
    )
    feedback_signal = 0.0
    if rec.genre in feedback_profile["positive_genres"] or rec.author in feedback_profile["positive_authors"]:
        feedback_signal = 0.85
        details.append("Danh sách đã dùng phản hồi thích/lưu kệ để ưu tiên sách tương tự.")
    elif rec.genre in feedback_profile["negative_genres"] or rec.author in feedback_profile["negative_authors"]:
        feedback_signal = 0.15
        details.append("Một số tín hiệu phản hồi làm giảm điểm các sách quá giống lựa chọn không hợp.")

    details.append(f"Điểm cộng đồng {rec.rating:.2f}/5 từ {format_number(rec.ratings_count)} lượt đánh giá.")
    if novelty > 0.2:
        details.append("Sách có độ mới lạ tương đối, giúp danh sách không chỉ xoay quanh sách quá phổ biến.")

    factors = [
        ("Thể loại", genre_signal),
        ("Tác giả", author_signal),
        ("Điểm sách", rating_norm),
        ("Phổ biến", popularity),
        ("Mới lạ", novelty),
        ("Phản hồi", feedback_signal),
    ]
    return factors, [detail for detail in details if detail]


def render_book_explanation(
    rec: BookRecommendation,
    user_id: str | None,
    ratings_df: pd.DataFrame,
    books_df: pd.DataFrame,
    key_prefix: str,
) -> None:
    factors, details = build_recommendation_explanation_factors(rec, user_id, ratings_df, books_df)
    factor_html = "".join(
        f"""
        <div class="mini-factor-row">
            <div>{escape_html(label)}</div>
            <div class="mini-factor-track"><div class="mini-factor-fill" style="width: {min(100, max(0, value) * 100):.0f}%;"></div></div>
            <div>{value:.0%}</div>
        </div>
        """
        for label, value in factors
    )
    detail_html = "".join(f"<div>- {escape_html(detail)}</div>" for detail in details[:5])

    with st.expander("Vì sao gợi ý này?", expanded=False):
        st.markdown(
            f"""
            <div class="book-explain-panel">
                <div class="book-explain-heading">Tín hiệu xếp hạng cho sách này</div>
                {factor_html}
                <div class="book-explain-detail">{detail_html}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_recommendation_entry(
    rec: BookRecommendation,
    rank: int,
    key_prefix: str,
    user_id: str | None,
    ratings_df: pd.DataFrame,
    books_df: pd.DataFrame,
) -> None:
    display_book_card(rec, rank)
    render_recommendation_actions(rec, f"{key_prefix}_{rank}")
    render_book_explanation(rec, user_id, ratings_df, books_df, f"{key_prefix}_{rank}")


def initialize_session_state():
    defaults = {
        "show_recommendations": True,
        "selected_user": None,
        "selected_mood": None,
        "user_selectbox_key": 0,
        "mood_selectbox_key": 0,
        "show_similar": False,
        "selected_similar_book": None,
        "similar_book_key": 0,
        "feedback_votes": {},
        "reading_shelf": [],
        "feedback_log": [],
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def clear_recommendations():
    """Clear the recommendations state and reset selections."""
    st.session_state.show_recommendations = False
    st.session_state.selected_user = None
    st.session_state.selected_mood = None
    st.session_state.user_selectbox_key += 1
    st.session_state.mood_selectbox_key += 1


def clear_similar_books():
    """Clear the similar books state and reset selection."""
    st.session_state.show_similar = False
    st.session_state.selected_similar_book = None
    st.session_state.similar_book_key += 1


def main():
    """Main application."""

    initialize_session_state()

    data_audit = inspect_data_files()
    books_df = load_books_data()
    ratings_df = generate_user_ratings(books_df)
    books_catalog_df = books_df.nlargest(5000, "ratings_count").copy()
    active_users = ratings_df["user_id"].value_counts().head(500)

    total_books = len(books_df)
    total_users = ratings_df["user_id"].nunique()
    total_ratings = len(ratings_df)
    total_genres = books_df["genre"].nunique()
    avg_rating = ratings_df["rating"].mean()

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-card">
                <h4>BRS DEMO</h4>
                <p><strong>Hệ gợi ý sách</strong></p>
                <p style="color: var(--text-light); font-size: 0.82rem;">Streamlit · KNN, SVD và mô hình lai</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

        render_sidebar_card(
            "Trạng thái demo",
            (
                f"<p><strong>{format_exact_number(total_books)}</strong> sách đã nạp</p>"
                f"<p><strong>{format_exact_number(total_users)}</strong> hồ sơ người dùng</p>"
                f"<p><strong>{format_exact_number(total_ratings)}</strong> lượt đánh giá</p>"
                f"<p><strong>{format_exact_number(total_genres)}</strong> thể loại</p>"
            ),
        )

        render_sidebar_card(
            "Dữ liệu",
            (
                f"<p><strong>{infer_data_source_label(data_audit)}</strong></p>"
                f"<p><strong>{format_exact_number(data_audit.get('books_records'))}</strong> sách gốc</p>"
                f"<p><strong>{format_exact_number(data_audit.get('users_records') or total_users)}</strong> người dùng gốc</p>"
                f"<p><strong>{format_exact_number(data_audit.get('ratings_records'))}</strong> tương tác gốc</p>"
                f"<p style='color: var(--text-light); font-size: 0.82rem;'>Working set: {format_exact_number(total_books)} sách · {format_exact_number(total_ratings)} đánh giá</p>"
            ),
        )

        render_sidebar_card(
            "Nguồn",
            (
                '<p><a href="https://github.com/OuyangXuelili/BRS" target="_blank">OuyangXuelili/BRS</a></p>'
                '<p><a href="https://sites.google.com/eng.ucsd.edu/ucsdbookgraph/home" target="_blank">UCSD Book Graph</a></p>'
                "<p style='color: var(--text-light); font-size: 0.82rem;'>App tự ưu tiên thư mục có full Goodreads/UCSD; nếu thiếu mới dùng subset hoặc catalog demo.</p>"
            ),
        )

        render_sidebar_card(
            "Điều khiển",
            "<p>Điều chỉnh độ đa dạng của danh sách và số lượng kết quả hiển thị.</p>",
        )

        exploration_level = st.slider(
            "Mức khám phá/đa dạng",
            min_value=0,
            max_value=100,
            value=45,
            step=5,
            help="Tăng để xếp hạng lại theo hướng đa dạng hơn; giảm để ưu tiên kết quả sát hồ sơ nhất.",
        )
        n_recommendations = st.slider("Số gợi ý hiển thị", min_value=5, max_value=20, value=10, step=1)

        featured_books = books_catalog_df.head(4).to_dict("records")
        featured_cards: list[str] = []
        for book in featured_books:
            display_author = resolve_display_author(book["title"], book["author"])
            cover_url = get_cover_image_url(book["title"], book["genre"], display_author, book.get("image_url", ""))
            fallback_cover_url = build_cover_svg_data_uri(book["title"], book["genre"], display_author)
            featured_cards.append(
                f'''<div class="hero-featured-book">
                        <div class="hero-featured-cover">
                            <img src="{escape_html(cover_url)}" alt="Bìa sách {escape_html(book["title"])}" loading="lazy" onerror="this.onerror=null;this.src='{escape_html(fallback_cover_url)}';">
                            <div class="hero-featured-mark">{build_cover_mark(book["title"])}</div>
                        </div>
                        <div>
                            <div class="hero-featured-title">{escape_html(book["title"])}</div>
                            <div class="hero-featured-author">{escape_html(display_author)}</div>
                            <div class="hero-featured-chip">{escape_html(translate_genre(book["genre"]))}</div>
                        </div>
                    </div>'''
            )
        featured_books_html = "".join(featured_cards)

    st.markdown(
        f"""
        <div class="hero-card">
            <div class="hero-grid">
                <div>
                    <div class="hero-kicker">BRS · Hệ gợi ý sách</div>
                    <h1 class="hero-title">Gợi ý sách cá nhân hóa từ lịch sử đánh giá.</h1>
                    <div class="hero-subtitle">
                        Kết hợp dữ liệu sách, hành vi đọc, mô hình lai và thí nghiệm đo lường để tạo danh sách gợi ý
                        có thể giải thích, so sánh và đánh giá rõ ràng.
                    </div>
                    <div class="hero-points">
                        <div class="hero-point">
                            <div class="hero-point-value">{format_number(total_books)}</div>
                            <div class="hero-point-label">Sách</div>
                        </div>
                        <div class="hero-point">
                            <div class="hero-point-value">{format_number(total_users)}</div>
                            <div class="hero-point-label">Hồ sơ người dùng</div>
                        </div>
                        <div class="hero-point">
                            <div class="hero-point-value">{format_number(total_ratings)}</div>
                            <div class="hero-point-label">Lượt đánh giá</div>
                        </div>
                        <div class="hero-point">
                            <div class="hero-point-value">{total_genres}</div>
                            <div class="hero-point-label">Thể loại</div>
                        </div>
                    </div>
                </div>
            </div>
            <div class="hero-panel">
                <div class="hero-panel-title-inline">Sách nổi bật</div>
                <div class="hero-rail">
                    {featured_books_html}
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs(TAB_LABELS)

    with tab1:
        st.markdown('<div class="section-title">Gợi ý cá nhân hóa</div>', unsafe_allow_html=True)
        render_feedback_summary(books_df)

        rec_method = st.radio(
            "Chọn chế độ gợi ý",
            ["Theo lịch sử đánh giá", "Theo tâm trạng đọc", "Người dùng mới"],
            horizontal=True,
        )

        if rec_method == "Theo lịch sử đánh giá":
            users = active_users.index.tolist()
            selected_user = st.selectbox(
                "Chọn hồ sơ người dùng",
                users,
                format_func=lambda user_id: format_user_profile_label(user_id, active_users.get(user_id, 0)),
                help="Chỉ hiển thị những người dùng hoạt động nhiều nhất để chọn nhanh",
                key=f"user_select_{st.session_state.user_selectbox_key}",
            )
            if st.session_state.selected_user is None:
                st.session_state.selected_user = selected_user

            btn_cols = st.columns([1.1, 0.8, 3.1])
            with btn_cols[0]:
                get_recs = st.button("Tạo gợi ý", type="primary")

            if get_recs:
                st.session_state.show_recommendations = True
                st.session_state.selected_user = selected_user

            if st.session_state.show_recommendations and st.session_state.selected_user:
                with st.spinner("Đang phân tích thói quen đọc..."):
                    user_ratings = ratings_df[ratings_df["user_id"] == st.session_state.selected_user]

                    metrics = st.columns(4)
                    with metrics[0]:
                        display_metric_card(str(len(user_ratings)), "Sách đã đánh giá")
                    with metrics[1]:
                        display_metric_card(f"{user_ratings['rating'].mean():.1f}", "Điểm đánh giá TB")
                    with metrics[2]:
                        fav_genre = (
                            books_df[books_df["book_id"].isin(user_ratings.nlargest(5, "rating")["book_id"])]
                            ["genre"]
                            .mode()
                            .iloc[0]
                            if len(user_ratings) > 0
                            else "N/A"
                        )
                        display_metric_card(translate_genre(fav_genre), "Thể loại yêu thích")
                    with metrics[3]:
                        display_metric_card(str(n_recommendations), "Số gợi ý")

                    with st.expander(f"Xem lịch sử đọc của {format_user_profile_label(st.session_state.selected_user, len(user_ratings))} ({len(user_ratings)} cuốn)"):
                        user_books = (
                            user_ratings.merge(
                                books_df[["book_id", "title", "author", "genre"]],
                                on="book_id",
                            ).sort_values("rating", ascending=False)
                        )

                        for _, row in user_books.iterrows():
                            st.markdown(
                                f"""
                                <div class="book-card" style="padding: 0.85rem 1rem; margin-bottom: 0.65rem;">
                                    <div class="book-card-header" style="margin-bottom: 0.4rem;">
                                        <div>
                                            <div class="book-title" style="font-size: 1rem; margin-bottom: 0.18rem;">{row['title']}</div>
                                            <div class="book-meta">Tác giả: {row['author']}</div>
                                        </div>
                                        <div class="score-pill">Điểm: {row['rating']:.1f}/5</div>
                                    </div>
                                    <div class="book-tags">
                                        <span class="badge badge-primary">{translate_genre(row['genre'])}</span>
                                    </div>
                                </div>
                                """,
                                unsafe_allow_html=True,
                            )

                    recommendations = get_user_recommendations(
                        st.session_state.selected_user,
                        ratings_df,
                        books_catalog_df,
                        n_recommendations,
                        exploration_level=exploration_level,
                        feedback_votes=st.session_state.feedback_votes,
                        saved_items=st.session_state.reading_shelf,
                    )

                    st.markdown(
                        f"<div class='info-box'><strong>Đã tìm thấy {len(recommendations)} gợi ý cá nhân cho {format_user_profile_label(st.session_state.selected_user)}.</strong></div>",
                        unsafe_allow_html=True,
                    )
                    render_recommendation_quality(recommendations, books_catalog_df)
                    render_user_model_explainability(
                        st.session_state.selected_user,
                        ratings_df,
                        books_df,
                        recommendations,
                        exploration_level,
                    )

                    for start in range(0, len(recommendations), 2):
                        row = recommendations[start : start + 2]
                        cols = st.columns(len(row))
                        for col, rec_idx in zip(cols, range(start, start + len(row))):
                            with col:
                                render_recommendation_entry(
                                    recommendations[rec_idx],
                                    rec_idx + 1,
                                    "profile",
                                    st.session_state.selected_user,
                                    ratings_df,
                                    books_df,
                                )

        elif rec_method == "Theo tâm trạng đọc":
            selected_mood = st.selectbox(
                "Hôm nay bạn muốn đọc theo tâm trạng nào?",
                list(READING_MOODS.keys()),
                format_func=translate_mood,
                help="Hệ thống sẽ tìm sách phù hợp với cảm xúc đọc hiện tại.",
                key=f"mood_select_{st.session_state.mood_selectbox_key}",
            )

            btn_cols = st.columns([1.1, 0.8, 3.1])
            with btn_cols[0]:
                get_mood_recs = st.button("Khám phá sách", type="primary")

            if get_mood_recs:
                st.session_state.show_recommendations = True
                st.session_state.selected_mood = selected_mood

            if st.session_state.show_recommendations and st.session_state.selected_mood:
                with st.spinner(f"Đang tìm sách theo tâm trạng {translate_mood(st.session_state.selected_mood)}..."):
                    recommendations = get_recommendations_by_mood(
                        st.session_state.selected_mood,
                        books_catalog_df,
                        n_recommendations + 8,
                        exploration_level=exploration_level,
                    )
                    recommendations = apply_feedback_to_recommendations(
                        recommendations,
                        books_catalog_df,
                        st.session_state.feedback_votes,
                        st.session_state.reading_shelf,
                        n_recommendations,
                    )

                    st.markdown(
                        f"<div class='info-box'><strong>Đã chọn tâm trạng {translate_mood(st.session_state.selected_mood)}.</strong><br><span style='color: {COLORS['text_light']};'>Đang tìm trong: {', '.join(translate_genre(g) for g in READING_MOODS[st.session_state.selected_mood])}</span></div>",
                        unsafe_allow_html=True,
                    )
                    render_recommendation_quality(recommendations, books_catalog_df)

                    for start in range(0, len(recommendations), 2):
                        row = recommendations[start : start + 2]
                        cols = st.columns(len(row))
                        for col, rec_idx in zip(cols, range(start, start + len(row))):
                            with col:
                                render_recommendation_entry(
                                    recommendations[rec_idx],
                                    rec_idx + 1,
                                    "mood",
                                    None,
                                    ratings_df,
                                    books_df,
                                )

        else:
            all_genres = sorted(books_catalog_df["genre"].dropna().unique().tolist())
            default_genres = [genre for genre in ["Fantasy", "Science Fiction", "Classic Fiction"] if genre in all_genres]
            preferred_genres = st.multiselect(
                "Chọn thể loại yêu thích ban đầu",
                all_genres,
                default=default_genres[:2] or all_genres[:2],
                format_func=translate_genre,
                help="Mô phỏng tình huống người dùng mới khi hệ thống chưa có lịch sử đánh giá.",
            )
            popularity_weight = st.slider(
                "Ưu tiên sách phổ biến",
                min_value=0.15,
                max_value=0.75,
                value=0.45,
                step=0.05,
                help="Giảm thanh này nếu muốn hệ thống thử nhiều sách mới lạ hơn.",
            )

            if preferred_genres:
                recommendations = get_cold_start_recommendations(
                    preferred_genres,
                    books_catalog_df,
                    n_recommendations + 8,
                    popularity_weight=popularity_weight,
                    exploration_level=exploration_level,
                )
                recommendations = apply_feedback_to_recommendations(
                    recommendations,
                    books_catalog_df,
                    st.session_state.feedback_votes,
                    st.session_state.reading_shelf,
                    n_recommendations,
                )
                st.markdown(
                    "<div class='info-box'><strong>Chế độ người dùng mới:</strong> hệ thống dùng sở thích thể loại ban đầu, điểm sách, độ phổ biến và độ mới lạ để tạo danh sách đầu tiên.</div>",
                    unsafe_allow_html=True,
                )
                render_recommendation_quality(recommendations, books_catalog_df)

                for start in range(0, len(recommendations), 2):
                    row = recommendations[start : start + 2]
                    cols = st.columns(len(row))
                    for col, rec_idx in zip(cols, range(start, start + len(row))):
                        with col:
                            render_recommendation_entry(
                                recommendations[rec_idx],
                                rec_idx + 1,
                                "cold",
                                None,
                                ratings_df,
                                books_df,
                            )
            else:
                st.info("Chọn ít nhất một thể loại để khởi tạo gợi ý cho người dùng mới.")

    with tab2:
        st.markdown('<div class="section-title">Sách phổ biến</div>', unsafe_allow_html=True)

        st.markdown(
            "<div class='info-box'><strong>Các sách được quan tâm nhiều</strong> dựa trên số lượt đánh giá trong bộ dữ liệu Goodreads.</div>",
            unsafe_allow_html=True,
        )

        filter_cols = st.columns(2)
        with filter_cols[0]:
            genre_filter = st.selectbox(
                "Lọc theo thể loại",
                ["Tất cả thể loại"] + sorted(books_df["genre"].unique().tolist()),
                format_func=lambda value: value if value == "Tất cả thể loại" else translate_genre(value),
            )
        with filter_cols[1]:
            sort_by = st.selectbox(
                "Sắp xếp theo",
                ["Phổ biến nhất", "Điểm cao nhất", "Mới nhất", "Cũ nhất"],
            )

        filtered_df = books_catalog_df.copy()
        if genre_filter != "Tất cả thể loại":
            filtered_df = filtered_df[filtered_df["genre"] == genre_filter]

        if sort_by == "Phổ biến nhất":
            filtered_df = filtered_df.sort_values("ratings_count", ascending=False)
        elif sort_by == "Điểm cao nhất":
            filtered_df = filtered_df.sort_values("rating", ascending=False)
        elif sort_by == "Mới nhất":
            filtered_df = filtered_df.sort_values("year", ascending=False)
        else:
            filtered_df = filtered_df.sort_values("year", ascending=True)

        popular_recommendations = []
        for _, book in filtered_df.head(n_recommendations + 8).iterrows():
            popular_recommendations.append(
                BookRecommendation(
                    title=book["title"],
                    author=book["author"],
                    genre=book["genre"],
                    image_url=book.get("image_url", ""),
                    score=min(0.99, book["rating"] / 5),
                    rating=book["rating"],
                    ratings_count=book["ratings_count"],
                    year=book["year"],
                    bestseller=book["bestseller"],
                    reason=f"Xếp hạng trong {translate_genre(genre_filter) if genre_filter != 'Tất cả thể loại' else 'tất cả sách'}",
                    book_id=str(book["book_id"]),
                )
            )
        popular_recommendations = apply_feedback_to_recommendations(
            popular_recommendations,
            books_catalog_df,
            st.session_state.feedback_votes,
            st.session_state.reading_shelf,
            n_recommendations,
        )
        for start in range(0, len(popular_recommendations), 2):
            row = popular_recommendations[start : start + 2]
            cols = st.columns(len(row))
            for col, rec_idx in zip(cols, range(start, start + len(row))):
                with col:
                    render_recommendation_entry(
                        popular_recommendations[rec_idx],
                        rec_idx + 1,
                        "popular",
                        None,
                        ratings_df,
                        books_df,
                    )

    with tab3:
        st.markdown('<div class="section-title">Tìm sách tương tự</div>', unsafe_allow_html=True)

        book_options = books_catalog_df["book_id"].tolist()
        book_labels = {
            row.book_id: f"{row.title} — {row.author if not is_missing_author(row.author) else 'Chưa rõ tác giả'}"
            for row in books_catalog_df.itertuples()
        }
        selected_book_id = st.selectbox(
            "Chọn một cuốn bạn thích",
            book_options,
            format_func=lambda book_id: book_labels.get(book_id, str(book_id)),
            help="Hệ thống sẽ tìm những cuốn sách tương tự trong nhóm sách phổ biến.",
            key=f"book_select_{st.session_state.similar_book_key}",
        )

        btn_cols = st.columns([1.1, 0.8, 3.1])
        with btn_cols[0]:
            find_similar = st.button("Tìm sách tương tự", type="primary")

        if find_similar:
            st.session_state.show_similar = True
            st.session_state.selected_similar_book = selected_book_id

        if st.session_state.show_similar and st.session_state.selected_similar_book:
            selected_book_matches = books_catalog_df[books_catalog_df["book_id"].astype(str) == str(st.session_state.selected_similar_book)]
            if selected_book_matches.empty:
                st.warning("Không tìm thấy sách đã chọn trong bộ dữ liệu hiện tại.")
            else:
                book_row = selected_book_matches.iloc[0]

                st.markdown(
                    f"<div class='info-box'><strong>Đã chọn: {book_row['title']}</strong><br><span style='color: {COLORS['text_light']};'>Tác giả: {resolve_display_author(book_row['title'], book_row['author'])} · {translate_genre(book_row['genre'])} · {get_star_rating(book_row['rating'])}</span></div>",
                    unsafe_allow_html=True,
                )

                with st.spinner("Đang tìm sách tương tự..."):
                    similar_books = get_similar_books(book_row["book_id"], books_catalog_df, n_recommendations + 8)
                    similar_books = apply_feedback_to_recommendations(
                        similar_books,
                        books_catalog_df,
                        st.session_state.feedback_votes,
                        st.session_state.reading_shelf,
                        n_recommendations,
                    )

                    for start in range(0, len(similar_books), 2):
                        row = similar_books[start : start + 2]
                        cols = st.columns(len(row))
                        for col, rec_idx in zip(cols, range(start, start + len(row))):
                            with col:
                                render_recommendation_entry(
                                    similar_books[rec_idx],
                                    rec_idx + 1,
                                    "similar",
                                    None,
                                    ratings_df,
                                    books_df,
                                )

    with tab4:
        st.markdown('<div class="section-title">Khám phá dữ liệu & working set</div>', unsafe_allow_html=True)
        render_data_audit_panel(books_df, ratings_df, data_audit)

        working_set_note = (
            "Nguồn đang trỏ tới full UCSD Book Graph. Các số bên dưới là working set đang chạy mô hình: "
            f"{format_exact_number(total_books)} sách nổi bật và {format_exact_number(total_ratings)} tương tác liên quan được lọc từ file gốc để giao diện phản hồi nhanh."
            if data_audit.get("likely_full_ucsd")
            else "Các số bên dưới là bộ dữ liệu hiện app đang dùng. Khi có full UCSD Book Graph, app sẽ tự nhận thư mục dữ liệu lớn nhất hợp lệ."
        )
        st.markdown(
            f"<div class='info-box'><strong>Lưu ý:</strong> {working_set_note}</div>",
            unsafe_allow_html=True,
        )

        stat_cols = st.columns(4)
        with stat_cols[0]:
            display_metric_card(str(total_books), "Sách trong dữ liệu")
        with stat_cols[1]:
            display_metric_card(str(total_users), "Hồ sơ người dùng")
        with stat_cols[2]:
            display_metric_card(format_number(total_ratings), "Lượt đánh giá")
        with stat_cols[3]:
            display_metric_card(f"{avg_rating:.2f}", "Điểm trung bình")

        st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

        chart_cols = st.columns(2)

        with chart_cols[0]:
            st.markdown('<div class="panel-card"><h4>Phân bố điểm đánh giá</h4></div>', unsafe_allow_html=True)
            fig_rating = px.histogram(
                ratings_df,
                x="rating",
                nbins=5,
                title="Phân bố điểm đánh giá",
                color_discrete_sequence=[COLORS["primary"]],
            )
            fig_rating.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis_title="Điểm đánh giá",
                yaxis_title="Số lượng",
                title_font=dict(size=18, family="Inter"),
            )
            st.plotly_chart(fig_rating, width="stretch")

        with chart_cols[1]:
            st.markdown('<div class="panel-card"><h4>Thể loại hàng đầu</h4></div>', unsafe_allow_html=True)
            genre_counts = books_df["genre"].value_counts().head(10)
            genre_counts.index = genre_counts.index.map(translate_genre)
            fig_genre = px.bar(
                x=genre_counts.values,
                y=genre_counts.index,
                orientation="h",
                title="Thể loại hàng đầu",
                color=genre_counts.values,
                color_continuous_scale=[[0, COLORS["secondary"]], [1, COLORS["primary"]]],
            )
            fig_genre.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                xaxis_title="Số lượng sách",
                yaxis_title="",
                showlegend=False,
                coloraxis_showscale=False,
                title_font=dict(size=18, family="Inter"),
            )
            st.plotly_chart(fig_genre, width="stretch")

        st.markdown('<div class="section-title">Mẫu sách</div>', unsafe_allow_html=True)
        display_df = books_df[["title", "author", "genre", "year", "rating", "ratings_count", "bestseller"]].head(15).copy()
        display_df["author"] = display_df.apply(
            lambda row: resolve_display_author(row["title"], row["author"]),
            axis=1,
        )
        display_df["genre"] = display_df["genre"].apply(translate_genre)
        display_df.columns = ["Tiêu đề", "Tác giả", "Thể loại", "Năm", "Điểm", "Lượt đánh giá", "Phổ biến"]
        display_df["Lượt đánh giá"] = display_df["Lượt đánh giá"].apply(format_number)
        st.dataframe(display_df, width="stretch", hide_index=True)

    with tab5:
        st.markdown('<div class="section-title">Đánh giá mô hình</div>', unsafe_allow_html=True)

        benchmark_ratings_df, benchmark_books_df, benchmark_summary = build_benchmark_sample(
            ratings_df,
            books_catalog_df,
        )
        eval_df, eval_summary = evaluate_working_recommenders(benchmark_ratings_df, benchmark_books_df, k=10)
        eval_summary.update(benchmark_summary)
        st.markdown(
            f"""
            <div class='info-box'>
                <strong>Đánh giá trên benchmark sample được rút từ working set hiện tại.</strong>
                Mỗi người dùng được giữ lại 20% lượt đánh giá làm tập kiểm tra; sách được xem là phù hợp khi điểm >= {eval_summary['relevance_threshold']:.1f}.
                Sample gồm {format_number(eval_summary['benchmark_books'])} sách,
                {format_number(eval_summary['benchmark_users'])} người dùng và
                {format_number(eval_summary['benchmark_ratings'])} lượt đánh giá, lấy từ working set
                {format_number(eval_summary['source_books'])} sách / {format_number(eval_summary['source_ratings'])} đánh giá.
                Đang đánh giá {eval_summary['evaluated_users']} người dùng với {format_number(eval_summary['train_ratings'])} lượt đánh giá huấn luyện
                và {format_number(eval_summary['test_ratings'])} lượt đánh giá kiểm tra; mỗi danh sách lấy top-{eval_summary['k']} gợi ý.
                Mốc so sánh SVD dùng {eval_summary['svd_components']} yếu tố ẩn
                (phương sai giải thích {eval_summary['svd_explained_variance']:.1%}).
            </div>
            """,
            unsafe_allow_html=True,
        )

        if eval_df.empty:
            st.warning("Chưa đủ dữ liệu kiểm tra để tính các chỉ số xếp hạng.")
        else:
            ablation_df = evaluate_ablation_study(benchmark_ratings_df, benchmark_books_df, k=10)
            rerank_df = evaluate_reranking_tradeoff(benchmark_ratings_df, benchmark_books_df, k=10)
            segment_df = evaluate_user_segment_performance(benchmark_ratings_df, benchmark_books_df, k=10)
            svd_sweep_df = evaluate_svd_factor_sweep(benchmark_ratings_df, benchmark_books_df, k=10)
            render_executive_summary(
                eval_df,
                eval_summary,
                ablation_df,
                rerank_df,
                benchmark_books_df,
                benchmark_ratings_df,
                data_audit,
            )

            best_row = eval_df.sort_values("NDCG@10", ascending=False).iloc[0]
            summary_cols = st.columns(5)
            with summary_cols[0]:
                display_metric_card(f"{best_row['Precision@10']:.1%}", "Precision@10 tốt nhất")
            with summary_cols[1]:
                display_metric_card(f"{best_row['Recall@10']:.1%}", "Recall@10 tốt nhất")
            with summary_cols[2]:
                display_metric_card(f"{best_row['NDCG@10']:.3f}", "NDCG@10 tốt nhất")
            with summary_cols[3]:
                display_metric_card(f"{best_row['Coverage']:.1%}", "Độ phủ dữ liệu sách")
            with summary_cols[4]:
                display_metric_card(str(best_row["Mô hình"]), "Mô hình dẫn đầu")

            st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)

            ranking_metrics = ["Precision@10", "Recall@10", "NDCG@10", "MAP@10", "MRR@10", "Hit Rate@10"]
            ranking_long = eval_df.melt(
                id_vars="Mô hình",
                value_vars=ranking_metrics,
                var_name="Metric",
                value_name="Điểm",
            )
            fig_ranking = px.bar(
                ranking_long,
                x="Mô hình",
                y="Điểm",
                color="Metric",
                barmode="group",
                color_discrete_sequence=[
                    COLORS["secondary"],
                    COLORS["primary"],
                    COLORS["highlight"],
                    "#64748b",
                    COLORS["accent"],
                    "#ef476f",
                ],
                title="Chất lượng xếp hạng trên tập kiểm tra",
            )
            fig_ranking.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                yaxis_title="Điểm",
                xaxis_title="",
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
            )
            st.plotly_chart(fig_ranking, width="stretch")

            quality_cols = st.columns(2)
            with quality_cols[0]:
                quality_long = eval_df.melt(
                    id_vars="Mô hình",
                    value_vars=["Coverage", "Diversity", "Novelty", "Personalization", "Long-tail Share"],
                    var_name="Metric",
                    value_name="Điểm",
                )
                quality_long["Metric"] = quality_long["Metric"].replace(
                    {
                        "Coverage": "Độ phủ",
                        "Diversity": "Đa dạng",
                        "Novelty": "Mới lạ",
                        "Personalization": "Cá nhân hóa",
                        "Long-tail Share": "Tỷ lệ sách ngách",
                    }
                )
                fig_quality = px.bar(
                    quality_long,
                    x="Mô hình",
                    y="Điểm",
                    color="Metric",
                    barmode="group",
                    color_discrete_sequence=[
                        COLORS["secondary"],
                        COLORS["highlight"],
                        "#6b7280",
                        COLORS["primary"],
                        COLORS["accent"],
                    ],
                    title="Chỉ số ngoài độ chính xác",
                )
                fig_quality.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    yaxis_title="Điểm",
                    xaxis_title="",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_quality, width="stretch")

            with quality_cols[1]:
                display_eval_df = eval_df.copy()
                percentage_cols = [
                    "Precision@10",
                    "Recall@10",
                    "Hit Rate@10",
                    "Coverage",
                    "Diversity",
                    "Novelty",
                    "Personalization",
                    "Long-tail Share",
                ]
                decimal_cols = ["NDCG@10", "MAP@10", "MRR@10", "Popularity Gini", "Calibration Error"]
                for col in percentage_cols:
                    display_eval_df[col] = display_eval_df[col].map(lambda value: f"{value:.1%}")
                for col in decimal_cols:
                    display_eval_df[col] = display_eval_df[col].map(lambda value: f"{value:.3f}")
                display_eval_df = display_eval_df.rename(
                    columns={
                        "Coverage": "Độ phủ",
                        "Diversity": "Đa dạng",
                        "Novelty": "Mới lạ",
                        "Personalization": "Cá nhân hóa",
                        "Long-tail Share": "Tỷ lệ sách ngách",
                        "Popularity Gini": "Gini phổ biến",
                        "Calibration Error": "Sai lệch hiệu chỉnh",
                    }
                )
                st.dataframe(display_eval_df, width="stretch", hide_index=True)
                st.markdown(
                    """
                    <div class="info-box" style="margin-top: 0.65rem;">
                        <strong>Cách đọc nhanh:</strong> MAP/MRR đo chất lượng thứ hạng;
                        độ cá nhân hóa và tỷ lệ sách ngách càng cao càng tốt;
                        Gini phổ biến và sai lệch hiệu chỉnh càng thấp càng tốt.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

            bias_long = eval_df.melt(
                id_vars="Mô hình",
                value_vars=["Popularity Gini", "Calibration Error"],
                var_name="Metric",
                value_name="Điểm",
            )
            bias_long["Metric"] = bias_long["Metric"].replace(
                {
                    "Popularity Gini": "Gini phổ biến",
                    "Calibration Error": "Sai lệch hiệu chỉnh",
                }
            )
            fig_bias = px.bar(
                bias_long,
                x="Mô hình",
                y="Điểm",
                color="Metric",
                barmode="group",
                color_discrete_sequence=[COLORS["accent"], "#64748b"],
                title="Thiên lệch phổ biến và sai lệch hiệu chỉnh (càng thấp càng tốt)",
            )
            fig_bias.update_layout(
                plot_bgcolor="white",
                paper_bgcolor="white",
                yaxis_title="Điểm",
                xaxis_title="",
                legend=dict(orientation="h", yanchor="bottom", y=1.02),
            )
            st.plotly_chart(fig_bias, width="stretch")

            st.markdown("### Ablation study")
            ablation_df = evaluate_ablation_study(benchmark_ratings_df, benchmark_books_df, k=10)
            if ablation_df.empty:
                st.info("Chưa đủ dữ liệu để chạy ablation study.")
            else:
                strongest_drop = ablation_df[ablation_df["Biến thể"] != "Đầy đủ"].sort_values("Δ NDCG@10").iloc[0]
                st.markdown(
                    f"""
                    <div class="info-box">
                        <strong>Ablation study đo tác động của từng tín hiệu trong mô hình lai.</strong>
                        Biến thể làm giảm NDCG@10 mạnh nhất hiện tại là
                        <strong>{strongest_drop['Biến thể']}</strong>
                        ({strongest_drop['Δ NDCG@10']:.3f} so với mô hình đầy đủ).
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                fig_ablation = px.bar(
                    ablation_df,
                    x="Biến thể",
                    y="Δ NDCG@10",
                    color="Tín hiệu bị bỏ",
                    color_discrete_sequence=[
                        COLORS["primary"],
                        COLORS["secondary"],
                        COLORS["highlight"],
                        COLORS["accent"],
                        "#64748b",
                        "#0f766e",
                    ],
                    title="Tác động khi loại từng tín hiệu khỏi mô hình lai",
                )
                fig_ablation.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    yaxis_title="Chênh lệch NDCG@10",
                    xaxis_title="",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_ablation, width="stretch")

                display_ablation_df = ablation_df.copy()
                percentage_cols = ["Coverage", "Diversity", "Novelty", "Personalization", "Long-tail Share"]
                decimal_cols = [
                    "Precision@10",
                    "Recall@10",
                    "NDCG@10",
                    "MAP@10",
                    "MRR@10",
                    "Popularity Gini",
                    "Calibration Error",
                    "Δ NDCG@10",
                    "Δ Đa dạng",
                    "Δ Độ phủ",
                ]
                for col in percentage_cols:
                    display_ablation_df[col] = display_ablation_df[col].map(lambda value: f"{value:.1%}")
                for col in decimal_cols:
                    display_ablation_df[col] = display_ablation_df[col].map(lambda value: f"{value:.3f}")
                display_ablation_df = display_ablation_df.rename(
                    columns={
                        "Coverage": "Độ phủ",
                        "Diversity": "Đa dạng",
                        "Novelty": "Mới lạ",
                        "Personalization": "Cá nhân hóa",
                        "Long-tail Share": "Tỷ lệ sách ngách",
                        "Popularity Gini": "Gini phổ biến",
                        "Calibration Error": "Sai lệch hiệu chỉnh",
                    }
                )
                st.dataframe(display_ablation_df, width="stretch", hide_index=True)

            st.markdown("### Đánh đổi khi xếp hạng lại")
            rerank_df = evaluate_reranking_tradeoff(benchmark_ratings_df, benchmark_books_df, k=10)
            if rerank_df.empty:
                st.info("Chưa đủ dữ liệu để đánh giá các mức xếp hạng lại.")
            else:
                best_tradeoff = rerank_df.sort_values("Balanced Score", ascending=False).iloc[0]
                st.markdown(
                    f"""
                    <div class='info-box'>
                        <strong>Xếp hạng lại là bước sắp xếp lại danh sách sau khi tạo tập ứng viên.</strong>
                        Ở đây hệ thống bắt đầu từ mô hình hồ sơ lai, sau đó giảm nhẹ điểm những ứng viên thuộc thể loại
                        đã xuất hiện nhiều trong top list. Mức cân bằng tốt nhất hiện tại là
                        <strong>{best_tradeoff['Reranking']}</strong> với điểm cân bằng
                        <strong>{best_tradeoff['Balanced Score']:.3f}</strong>.
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                tradeoff_long = rerank_df.melt(
                    id_vars=["Reranking"],
                    value_vars=[
                        "NDCG@10",
                        "Diversity",
                        "Coverage",
                        "Personalization",
                        "Long-tail Share",
                        "Balanced Score",
                    ],
                    var_name="Metric",
                    value_name="Điểm",
                )
                tradeoff_long["Metric"] = tradeoff_long["Metric"].replace(
                    {
                        "Diversity": "Đa dạng",
                        "Coverage": "Độ phủ",
                        "Personalization": "Cá nhân hóa",
                        "Long-tail Share": "Tỷ lệ sách ngách",
                        "Balanced Score": "Điểm cân bằng",
                    }
                )
                fig_tradeoff = px.line(
                    tradeoff_long,
                    x="Reranking",
                    y="Điểm",
                    color="Metric",
                    markers=True,
                    color_discrete_sequence=[
                        COLORS["primary"],
                        COLORS["secondary"],
                        COLORS["accent"],
                        "#64748b",
                        COLORS["highlight"],
                        "#475569",
                    ],
                    title="Ảnh hưởng của xếp hạng lại theo độ đa dạng",
                )
                fig_tradeoff.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    yaxis_title="Điểm",
                    xaxis_title="Mức xếp hạng lại",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_tradeoff, width="stretch")

                display_rerank_df = rerank_df.copy()
                for col in [
                    "Diversity weight",
                    "Precision@10",
                    "Recall@10",
                    "NDCG@10",
                    "MAP@10",
                    "MRR@10",
                    "Hit Rate@10",
                    "Coverage",
                    "Diversity",
                    "Novelty",
                    "Personalization",
                    "Long-tail Share",
                    "Popularity Gini",
                    "Calibration Error",
                    "Balanced Score",
                ]:
                    display_rerank_df[col] = display_rerank_df[col].map(lambda value: f"{value:.3f}")
                display_rerank_df = display_rerank_df.rename(
                    columns={
                        "Reranking": "Mức xếp hạng lại",
                        "Diversity weight": "Trọng số đa dạng",
                        "Coverage": "Độ phủ",
                        "Diversity": "Đa dạng",
                        "Novelty": "Mới lạ",
                        "Personalization": "Cá nhân hóa",
                        "Long-tail Share": "Tỷ lệ sách ngách",
                        "Popularity Gini": "Gini phổ biến",
                        "Calibration Error": "Sai lệch hiệu chỉnh",
                        "Balanced Score": "Điểm cân bằng",
                    }
                )
                st.dataframe(display_rerank_df, width="stretch", hide_index=True)

            st.markdown("### Độ ổn định theo nhóm người dùng")
            segment_df = evaluate_user_segment_performance(benchmark_ratings_df, benchmark_books_df, k=10)
            if segment_df.empty:
                st.info("Chưa đủ dữ liệu để phân tích theo nhóm người dùng.")
            else:
                fig_segment = px.bar(
                    segment_df,
                    x="Nhóm người dùng",
                    y="NDCG@10",
                    color="Mô hình",
                    barmode="group",
                    color_discrete_sequence=[COLORS["primary"], COLORS["secondary"], COLORS["accent"], COLORS["highlight"]],
                    title="NDCG@10 theo mức độ lịch sử của người dùng",
                )
                fig_segment.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    yaxis_title="NDCG@10",
                    xaxis_title="",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_segment, width="stretch")

                display_segment_df = segment_df.copy()
                for col in ["Precision@10", "Recall@10"]:
                    display_segment_df[col] = display_segment_df[col].map(lambda value: f"{value:.1%}")
                for col in ["NDCG@10", "MAP@10", "MRR@10", "Calibration Error"]:
                    display_segment_df[col] = display_segment_df[col].map(lambda value: f"{value:.3f}")
                display_segment_df = display_segment_df.rename(
                    columns={
                        "Calibration Error": "Sai lệch hiệu chỉnh",
                    }
                )
                st.dataframe(display_segment_df, width="stretch", hide_index=True)

            st.markdown("### Tối ưu số yếu tố ẩn")
            svd_sweep_df = evaluate_svd_factor_sweep(benchmark_ratings_df, benchmark_books_df, k=10)
            if svd_sweep_df.empty:
                st.info("Chưa đủ dữ liệu để thử nhiều số yếu tố ẩn.")
            else:
                best_svd = svd_sweep_df.sort_values("NDCG@10", ascending=False).iloc[0]
                st.markdown(
                    f"""
                    <div class="info-box">
                        <strong>Thử nghiệm số yếu tố ẩn của SVD:</strong> thử nhiều kích thước biểu diễn ẩn để tránh chọn tham số cảm tính.
                        Trên dữ liệu demo hiện tại, <strong>{int(best_svd['Factors'])} yếu tố ẩn</strong>
                        đang cho NDCG@10 tốt nhất ({best_svd['NDCG@10']:.3f}).
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
                svd_sweep_long = svd_sweep_df.melt(
                    id_vars="Factors",
                    value_vars=["NDCG@10", "MAP@10", "MRR@10", "Personalization", "Calibration Error"],
                    var_name="Metric",
                    value_name="Điểm",
                )
                svd_sweep_long["Metric"] = svd_sweep_long["Metric"].replace(
                    {
                        "Personalization": "Cá nhân hóa",
                        "Calibration Error": "Sai lệch hiệu chỉnh",
                    }
                )
                fig_svd_sweep = px.line(
                    svd_sweep_long,
                    x="Factors",
                    y="Điểm",
                    color="Metric",
                    markers=True,
                    color_discrete_sequence=[
                        COLORS["primary"],
                        COLORS["secondary"],
                        COLORS["highlight"],
                        COLORS["accent"],
                        "#64748b",
                    ],
                    title="Thử nghiệm siêu tham số cho SVD",
                )
                fig_svd_sweep.update_layout(
                    plot_bgcolor="white",
                    paper_bgcolor="white",
                    yaxis_title="Điểm",
                    xaxis_title="Số yếu tố ẩn",
                    legend=dict(orientation="h", yanchor="bottom", y=1.02),
                )
                st.plotly_chart(fig_svd_sweep, width="stretch")

                display_svd_sweep_df = svd_sweep_df.copy()
                for col in [
                    "Explained variance",
                    "Precision@10",
                    "Recall@10",
                    "NDCG@10",
                    "MAP@10",
                    "MRR@10",
                    "Coverage",
                    "Personalization",
                    "Calibration Error",
                ]:
                    display_svd_sweep_df[col] = display_svd_sweep_df[col].map(lambda value: f"{value:.3f}")
                display_svd_sweep_df = display_svd_sweep_df.rename(
                    columns={
                        "Factors": "Số yếu tố ẩn",
                        "Explained variance": "Phương sai giải thích",
                        "Coverage": "Độ phủ",
                        "Personalization": "Cá nhân hóa",
                        "Calibration Error": "Sai lệch hiệu chỉnh",
                    }
                )
                st.dataframe(display_svd_sweep_df, width="stretch", hide_index=True)

            st.markdown("### Xuất báo cáo")
            report_markdown = build_evaluation_report(
                eval_df,
                eval_summary,
                ablation_df,
                rerank_df,
                segment_df,
                svd_sweep_df,
            )
            metrics_csv = build_metrics_csv_export(
                [
                    ("So sánh mô hình", eval_df),
                    ("Ablation study", ablation_df),
                    ("Xếp hạng lại", rerank_df),
                    ("Nhóm người dùng", segment_df),
                    ("SVD sweep", svd_sweep_df),
                ]
            )
            export_cols = st.columns(2)
            with export_cols[0]:
                st.download_button(
                    "Tải báo cáo Markdown",
                    data=report_markdown.encode("utf-8"),
                    file_name="brs_evaluation_report.md",
                    mime="text/markdown",
                    use_container_width=True,
                )
            with export_cols[1]:
                st.download_button(
                    "Tải bảng metric CSV",
                    data=metrics_csv.encode("utf-8-sig"),
                    file_name="brs_metrics_export.csv",
                    mime="text/csv",
                    use_container_width=True,
                )

            st.markdown("### Thiết kế thí nghiệm")
            insight_cols = st.columns(4)
            with insight_cols[0]:
                st.markdown(
                    """
                    <div class="panel-card" style="min-height: 190px;">
                        <h4>Mốc so sánh rõ ràng</h4>
                        <p>Mô hình phổ biến được giữ lại để chứng minh gợi ý cá nhân hóa có tạo thêm giá trị so với việc chỉ đề xuất sách nổi tiếng.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with insight_cols[1]:
                st.markdown(
                    """
                    <div class="panel-card" style="min-height: 190px;">
                        <h4>Lọc cộng tác</h4>
                        <p>Item-KNN và SVD cùng được đưa vào để so sánh hai hướng collaborative filtering: láng giềng item-item và biểu diễn ẩn.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with insight_cols[2]:
                st.markdown(
                    """
                    <div class="panel-card" style="min-height: 190px;">
                        <h4>Ngoài độ chính xác</h4>
                        <p>Độ phủ, đa dạng, mới lạ, cá nhân hóa, sách ngách và hiệu chỉnh giúp chứng minh hệ thống không chỉ tối ưu tỷ lệ trúng.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )
            with insight_cols[3]:
                st.markdown(
                    """
                    <div class="panel-card" style="min-height: 190px;">
                        <h4>Người dùng mới</h4>
                        <p>Chế độ này cho thấy hệ thống có chiến lược khi chưa có lịch sử đánh giá; phản hồi trực tiếp tiếp tục tinh chỉnh danh sách trong phiên demo.</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )


FAMOUS_BOOKS = [

    # Classic Fiction
    {"title": "To Kill a Mockingbird", "author": "Harper Lee", "genre": "Classic Fiction", "year": 1960, "rating": 4.27, "ratings_count": 5012983, "bestseller": True},
    {"title": "1984", "author": "George Orwell", "genre": "Dystopian Fiction", "year": 1949, "rating": 4.19, "ratings_count": 4012832, "bestseller": True},
    {"title": "Pride and Prejudice", "author": "Jane Austen", "genre": "Romance", "year": 1813, "rating": 4.28, "ratings_count": 3654821, "bestseller": True},
    {"title": "The Great Gatsby", "author": "F. Scott Fitzgerald", "genre": "Classic Fiction", "year": 1925, "rating": 3.93, "ratings_count": 4821093, "bestseller": True},
    {"title": "One Hundred Years of Solitude", "author": "Gabriel García Márquez", "genre": "Magical Realism", "year": 1967, "rating": 4.11, "ratings_count": 873291, "bestseller": True},
    {"title": "Jane Eyre", "author": "Charlotte Brontë", "genre": "Classic Fiction", "year": 1847, "rating": 4.14, "ratings_count": 1876543, "bestseller": False},
    {"title": "Wuthering Heights", "author": "Emily Brontë", "genre": "Classic Fiction", "year": 1847, "rating": 3.88, "ratings_count": 1432198, "bestseller": False},
    {"title": "The Catcher in the Rye", "author": "J.D. Salinger", "genre": "Classic Fiction", "year": 1951, "rating": 3.81, "ratings_count": 3210987, "bestseller": True},
    {"title": "Crime and Punishment", "author": "Fyodor Dostoevsky", "genre": "Classic Fiction", "year": 1866, "rating": 4.27, "ratings_count": 765432, "bestseller": False},
    {"title": "The Count of Monte Cristo", "author": "Alexandre Dumas", "genre": "Classic Fiction", "year": 1844, "rating": 4.29, "ratings_count": 876543, "bestseller": False},
    {"title": "Moby Dick", "author": "Herman Melville", "genre": "Classic Fiction", "year": 1851, "rating": 3.53, "ratings_count": 654321, "bestseller": False},
    {"title": "War and Peace", "author": "Leo Tolstoy", "genre": "Classic Fiction", "year": 1869, "rating": 4.18, "ratings_count": 432198, "bestseller": False},
    {"title": "Anna Karenina", "author": "Leo Tolstoy", "genre": "Classic Fiction", "year": 1877, "rating": 4.09, "ratings_count": 765432, "bestseller": False},
    {"title": "The Brothers Karamazov", "author": "Fyodor Dostoevsky", "genre": "Classic Fiction", "year": 1880, "rating": 4.36, "ratings_count": 321098, "bestseller": False},
    {"title": "Les Misérables", "author": "Victor Hugo", "genre": "Classic Fiction", "year": 1862, "rating": 4.20, "ratings_count": 876543, "bestseller": True},

    # Fantasy
    {"title": "Harry Potter and the Sorcerer's Stone", "author": "J.K. Rowling", "genre": "Fantasy", "year": 1997, "rating": 4.47, "ratings_count": 8923014, "bestseller": True},
    {"title": "The Hobbit", "author": "J.R.R. Tolkien", "genre": "Fantasy", "year": 1937, "rating": 4.28, "ratings_count": 3421098, "bestseller": True},
    {"title": "A Game of Thrones", "author": "George R.R. Martin", "genre": "Fantasy", "year": 1996, "rating": 4.44, "ratings_count": 2198432, "bestseller": True},
    {"title": "The Name of the Wind", "author": "Patrick Rothfuss", "genre": "Fantasy", "year": 2007, "rating": 4.52, "ratings_count": 987234, "bestseller": False},
    {"title": "Mistborn: The Final Empire", "author": "Brandon Sanderson", "genre": "Fantasy", "year": 2006, "rating": 4.46, "ratings_count": 654321, "bestseller": False},
    {"title": "The Way of Kings", "author": "Brandon Sanderson", "genre": "Fantasy", "year": 2010, "rating": 4.64, "ratings_count": 432198, "bestseller": True},
    {"title": "The Lord of the Rings", "author": "J.R.R. Tolkien", "genre": "Fantasy", "year": 1954, "rating": 4.53, "ratings_count": 6543210, "bestseller": True},
    {"title": "A Wizard of Earthsea", "author": "Ursula K. Le Guin", "genre": "Fantasy", "year": 1968, "rating": 4.01, "ratings_count": 321098, "bestseller": False},

    # Science Fiction
    {"title": "Dune", "author": "Frank Herbert", "genre": "Science Fiction", "year": 1965, "rating": 4.26, "ratings_count": 1234567, "bestseller": True},
    {"title": "Ender's Game", "author": "Orson Scott Card", "genre": "Science Fiction", "year": 1985, "rating": 4.30, "ratings_count": 1432198, "bestseller": True},
    {"title": "The Hitchhiker's Guide to the Galaxy", "author": "Douglas Adams", "genre": "Science Fiction", "year": 1979, "rating": 4.23, "ratings_count": 1821093, "bestseller": True},
    {"title": "Foundation", "author": "Isaac Asimov", "genre": "Science Fiction", "year": 1951, "rating": 4.17, "ratings_count": 432198, "bestseller": False},
    {"title": "Brave New World", "author": "Aldous Huxley", "genre": "Dystopian Fiction", "year": 1932, "rating": 3.99, "ratings_count": 1654321, "bestseller": True},
    {"title": "Project Hail Mary", "author": "Andy Weir", "genre": "Science Fiction", "year": 2021, "rating": 4.52, "ratings_count": 876543, "bestseller": True},
    {"title": "The Martian", "author": "Andy Weir", "genre": "Science Fiction", "year": 2011, "rating": 4.41, "ratings_count": 987654, "bestseller": True},
    {"title": "Neuromancer", "author": "William Gibson", "genre": "Science Fiction", "year": 1984, "rating": 3.89, "ratings_count": 321098, "bestseller": False},
    {"title": "Snow Crash", "author": "Neal Stephenson", "genre": "Science Fiction", "year": 1992, "rating": 4.03, "ratings_count": 234567, "bestseller": False},
    {"title": "Ready Player One", "author": "Ernest Cline", "genre": "Science Fiction", "year": 2011, "rating": 4.25, "ratings_count": 876543, "bestseller": True},

    # Mystery & Thriller
    {"title": "The Girl with the Dragon Tattoo", "author": "Stieg Larsson", "genre": "Mystery", "year": 2005, "rating": 4.14, "ratings_count": 2876543, "bestseller": True},
    {"title": "Gone Girl", "author": "Gillian Flynn", "genre": "Thriller", "year": 2012, "rating": 4.12, "ratings_count": 2543210, "bestseller": True},
    {"title": "The Da Vinci Code", "author": "Dan Brown", "genre": "Thriller", "year": 2003, "rating": 3.91, "ratings_count": 3210987, "bestseller": True},
    {"title": "And Then There Were None", "author": "Agatha Christie", "genre": "Mystery", "year": 1939, "rating": 4.27, "ratings_count": 987654, "bestseller": True},
    {"title": "The Silent Patient", "author": "Alex Michaelides", "genre": "Thriller", "year": 2019, "rating": 4.08, "ratings_count": 876543, "bestseller": True},
    {"title": "In Cold Blood", "author": "Truman Capote", "genre": "True Crime", "year": 1966, "rating": 4.08, "ratings_count": 432198, "bestseller": True},
    {"title": "The Girl on the Train", "author": "Paula Hawkins", "genre": "Thriller", "year": 2015, "rating": 3.94, "ratings_count": 2109876, "bestseller": True},
    {"title": "Big Little Lies", "author": "Liane Moriarty", "genre": "Mystery", "year": 2014, "rating": 4.07, "ratings_count": 765432, "bestseller": True},

    # Non-Fiction & Self-Help
    {"title": "Sapiens: A Brief History of Humankind", "author": "Yuval Noah Harari", "genre": "Non-Fiction", "year": 2011, "rating": 4.39, "ratings_count": 1765432, "bestseller": True},
    {"title": "Atomic Habits", "author": "James Clear", "genre": "Self-Help", "year": 2018, "rating": 4.37, "ratings_count": 987654, "bestseller": True},
    {"title": "Thinking, Fast and Slow", "author": "Daniel Kahneman", "genre": "Psychology", "year": 2011, "rating": 4.18, "ratings_count": 654321, "bestseller": True},
    {"title": "The Power of Habit", "author": "Charles Duhigg", "genre": "Self-Help", "year": 2012, "rating": 4.13, "ratings_count": 543210, "bestseller": False},
    {"title": "Educated", "author": "Tara Westover", "genre": "Memoir", "year": 2018, "rating": 4.47, "ratings_count": 1234567, "bestseller": True},
    {"title": "Becoming", "author": "Michelle Obama", "genre": "Memoir", "year": 2018, "rating": 4.53, "ratings_count": 1543210, "bestseller": True},
    {"title": "Steve Jobs", "author": "Walter Isaacson", "genre": "Biography", "year": 2011, "rating": 4.18, "ratings_count": 987654, "bestseller": True},
    {"title": "The Subtle Art of Not Giving a F*ck", "author": "Mark Manson", "genre": "Self-Help", "year": 2016, "rating": 3.93, "ratings_count": 1234567, "bestseller": True},
    {"title": "Zero to One", "author": "Peter Thiel", "genre": "Business", "year": 2014, "rating": 4.18, "ratings_count": 543210, "bestseller": True},
    {"title": "The Lean Startup", "author": "Eric Ries", "genre": "Business", "year": 2011, "rating": 4.11, "ratings_count": 432198, "bestseller": True},
    {"title": "Deep Work", "author": "Cal Newport", "genre": "Self-Help", "year": 2016, "rating": 4.18, "ratings_count": 321098, "bestseller": True},
    {"title": "Meditations", "author": "Marcus Aurelius", "genre": "Philosophy", "year": 180, "rating": 4.26, "ratings_count": 432198, "bestseller": False},

    # Romance
    {"title": "The Notebook", "author": "Nicholas Sparks", "genre": "Romance", "year": 1996, "rating": 4.10, "ratings_count": 1432198, "bestseller": True},
    {"title": "Outlander", "author": "Diana Gabaldon", "genre": "Romance", "year": 1991, "rating": 4.25, "ratings_count": 987654, "bestseller": True},
    {"title": "Me Before You", "author": "Jojo Moyes", "genre": "Romance", "year": 2012, "rating": 4.27, "ratings_count": 876543, "bestseller": True},
    {"title": "The Fault in Our Stars", "author": "John Green", "genre": "Romance", "year": 2012, "rating": 4.14, "ratings_count": 3654821, "bestseller": True},
    {"title": "Beach Read", "author": "Emily Henry", "genre": "Romance", "year": 2020, "rating": 3.95, "ratings_count": 543210, "bestseller": False},
    {"title": "It Ends with Us", "author": "Colleen Hoover", "genre": "Romance", "year": 2016, "rating": 4.38, "ratings_count": 2109876, "bestseller": True},
    {"title": "The Seven Husbands of Evelyn Hugo", "author": "Taylor Jenkins Reid", "genre": "Romance", "year": 2017, "rating": 4.46, "ratings_count": 1543210, "bestseller": True},
    {"title": "People We Meet on Vacation", "author": "Emily Henry", "genre": "Romance", "year": 2021, "rating": 4.08, "ratings_count": 654321, "bestseller": True},

    # Horror
    {"title": "It", "author": "Stephen King", "genre": "Horror", "year": 1986, "rating": 4.25, "ratings_count": 876543, "bestseller": True},
    {"title": "The Shining", "author": "Stephen King", "genre": "Horror", "year": 1977, "rating": 4.26, "ratings_count": 765432, "bestseller": True},
    {"title": "Dracula", "author": "Bram Stoker", "genre": "Horror", "year": 1897, "rating": 4.01, "ratings_count": 1098765, "bestseller": False},
    {"title": "Mexican Gothic", "author": "Silvia Moreno-Garcia", "genre": "Horror", "year": 2020, "rating": 3.69, "ratings_count": 321098, "bestseller": False},
    {"title": "House of Leaves", "author": "Mark Z. Danielewski", "genre": "Horror", "year": 2000, "rating": 4.12, "ratings_count": 210987, "bestseller": False},
    {"title": "Pet Sematary", "author": "Stephen King", "genre": "Horror", "year": 1983, "rating": 4.05, "ratings_count": 543210, "bestseller": True},
    {"title": "The Haunting of Hill House", "author": "Shirley Jackson", "genre": "Horror", "year": 1959, "rating": 4.02, "ratings_count": 321098, "bestseller": False},

    # Historical Fiction
    {"title": "The Book Thief", "author": "Markus Zusak", "genre": "Historical Fiction", "year": 2005, "rating": 4.39, "ratings_count": 2109876, "bestseller": True},
    {"title": "All the Light We Cannot See", "author": "Anthony Doerr", "genre": "Historical Fiction", "year": 2014, "rating": 4.34, "ratings_count": 1098765, "bestseller": True},
    {"title": "The Pillars of the Earth", "author": "Ken Follett", "genre": "Historical Fiction", "year": 1989, "rating": 4.34, "ratings_count": 654321, "bestseller": True},
    {"title": "Circe", "author": "Madeline Miller", "genre": "Historical Fiction", "year": 2018, "rating": 4.28, "ratings_count": 765432, "bestseller": True},
    {"title": "The Kite Runner", "author": "Khaled Hosseini", "genre": "Historical Fiction", "year": 2003, "rating": 4.34, "ratings_count": 2876543, "bestseller": True},
    {"title": "A Thousand Splendid Suns", "author": "Khaled Hosseini", "genre": "Historical Fiction", "year": 2007, "rating": 4.42, "ratings_count": 1234567, "bestseller": True},
    {"title": "The Tattooist of Auschwitz", "author": "Heather Morris", "genre": "Historical Fiction", "year": 2018, "rating": 4.29, "ratings_count": 654321, "bestseller": True},
    {"title": "The Song of Achilles", "author": "Madeline Miller", "genre": "Historical Fiction", "year": 2011, "rating": 4.38, "ratings_count": 876543, "bestseller": True},

    # Contemporary Fiction
    {"title": "Where the Crawdads Sing", "author": "Delia Owens", "genre": "Contemporary Fiction", "year": 2018, "rating": 4.46, "ratings_count": 2543210, "bestseller": True},
    {"title": "The Midnight Library", "author": "Matt Haig", "genre": "Contemporary Fiction", "year": 2020, "rating": 4.02, "ratings_count": 876543, "bestseller": True},
    {"title": "A Man Called Ove", "author": "Fredrik Backman", "genre": "Contemporary Fiction", "year": 2012, "rating": 4.38, "ratings_count": 987654, "bestseller": True},
    {"title": "Little Fires Everywhere", "author": "Celeste Ng", "genre": "Contemporary Fiction", "year": 2017, "rating": 4.12, "ratings_count": 654321, "bestseller": True},
    {"title": "Normal People", "author": "Sally Rooney", "genre": "Contemporary Fiction", "year": 2018, "rating": 3.87, "ratings_count": 543210, "bestseller": True},
    {"title": "The Alchemist", "author": "Paulo Coelho", "genre": "Philosophy", "year": 1988, "rating": 3.92, "ratings_count": 2876543, "bestseller": True},
    {"title": "Life of Pi", "author": "Yann Martel", "genre": "Contemporary Fiction", "year": 2001, "rating": 3.94, "ratings_count": 1543210, "bestseller": True},
    {"title": "Tomorrow and Tomorrow and Tomorrow", "author": "Gabrielle Zevin", "genre": "Contemporary Fiction", "year": 2022, "rating": 4.21, "ratings_count": 543210, "bestseller": True},

    # Young Adult
    {"title": "The Hunger Games", "author": "Suzanne Collins", "genre": "Young Adult", "year": 2008, "rating": 4.32, "ratings_count": 7654321, "bestseller": True},
    {"title": "Divergent", "author": "Veronica Roth", "genre": "Young Adult", "year": 2011, "rating": 4.15, "ratings_count": 3456789, "bestseller": True},
    {"title": "Percy Jackson: The Lightning Thief", "author": "Rick Riordan", "genre": "Young Adult", "year": 2005, "rating": 4.29, "ratings_count": 2345678, "bestseller": True},
    {"title": "Twilight", "author": "Stephenie Meyer", "genre": "Young Adult", "year": 2005, "rating": 3.64, "ratings_count": 5678901, "bestseller": True},
    {"title": "The Maze Runner", "author": "James Dashner", "genre": "Young Adult", "year": 2009, "rating": 4.03, "ratings_count": 1234567, "bestseller": True},
    {"title": "The Giver", "author": "Lois Lowry", "genre": "Young Adult", "year": 1993, "rating": 4.13, "ratings_count": 2109876, "bestseller": True},
    {"title": "Six of Crows", "author": "Leigh Bardugo", "genre": "Young Adult", "year": 2015, "rating": 4.49, "ratings_count": 654321, "bestseller": True},
    {"title": "Children of Blood and Bone", "author": "Tomi Adeyemi", "genre": "Young Adult", "year": 2018, "rating": 4.09, "ratings_count": 321098, "bestseller": True},
    {"title": "The Perks of Being a Wallflower", "author": "Stephen Chbosky", "genre": "Young Adult", "year": 1999, "rating": 4.22, "ratings_count": 1765432, "bestseller": True},
    {"title": "Simon vs. the Homo Sapiens Agenda", "author": "Becky Albertalli", "genre": "Young Adult", "year": 2015, "rating": 4.27, "ratings_count": 543210, "bestseller": True},
    {"title": "Fahrenheit 451", "author": "Ray Bradbury", "genre": "Dystopian Fiction", "year": 1953, "rating": 3.97, "ratings_count": 1987654, "bestseller": True},
    {"title": "Slaughterhouse-Five", "author": "Kurt Vonnegut", "genre": "Science Fiction", "year": 1969, "rating": 4.09, "ratings_count": 876543, "bestseller": True},
    {"title": "The Road", "author": "Cormac McCarthy", "genre": "Dystopian Fiction", "year": 2006, "rating": 3.98, "ratings_count": 765432, "bestseller": True},
    {"title": "Frankenstein", "author": "Mary Shelley", "genre": "Horror", "year": 1818, "rating": 3.84, "ratings_count": 1432198, "bestseller": False},
    {"title": "Rebecca", "author": "Daphne du Maurier", "genre": "Mystery", "year": 1938, "rating": 4.24, "ratings_count": 543210, "bestseller": True},
    {"title": "The Picture of Dorian Gray", "author": "Oscar Wilde", "genre": "Classic Fiction", "year": 1890, "rating": 4.12, "ratings_count": 1234567, "bestseller": False},
]

# ============================================================================
# USER NAMES - Realistic names for demo
# ============================================================================
USER_NAMES = [
    "Emma Thompson", "Liam Anderson", "Olivia Martinez", "Noah Williams", "Ava Johnson",
    "Ethan Brown", "Sophia Davis", "Mason Garcia", "Isabella Miller", "James Wilson",
    "Mia Moore", "Benjamin Taylor", "Charlotte Thomas", "Lucas Jackson", "Amelia White",
    "Henry Harris", "Harper Martin", "Alexander Lee", "Evelyn Clark", "Sebastian Lewis",
    "Abigail Walker", "Jack Robinson", "Emily Hall", "Daniel Allen", "Elizabeth Young",
    "Michael King", "Sofia Wright", "David Scott", "Avery Green", "Joseph Baker",
    "Scarlett Adams", "Samuel Nelson", "Victoria Hill", "Owen Campbell", "Grace Mitchell",
    "Gabriel Roberts", "Chloe Carter", "Carter Phillips", "Lily Evans", "Jayden Turner",
    "Zoey Collins", "Dylan Edwards", "Penelope Stewart", "Luke Morris", "Layla Murphy",
    "Anthony Rivera", "Riley Cook", "Isaac Rogers", "Nora Morgan", "Christopher Cooper",
    "Hannah Peterson", "Andrew Reed", "Aria Bailey", "Joshua Howard", "Ellie Ward",
    "Nathan Foster", "Audrey Sanders", "Ryan Price", "Leah Bennett", "Brandon Wood",
    "Savannah Brooks", "Kevin Kelly", "Brooklyn Hughes", "Justin Long", "Stella Ross",
    "Austin Powell", "Claire Jenkins", "Evan Perry", "Violet Butler", "Aaron Russell",
    "Lucy Griffin", "Adam Hayes", "Anna Simmons", "Tyler Patterson", "Maya Henderson",
    "Zachary Coleman", "Autumn Richardson", "Hunter Cox", "Bella Howard", "Jordan Ward",
    "Katherine Gonzalez", "Jason Bryant", "Natalie Alexander", "Caleb Russell", "Sarah Torres",
    "Christian Gray", "Aaliyah Ramirez", "Jonathan Watson", "Madison Brooks", "Nicholas Flores",
    "Taylor Washington", "Adrian Butler", "Samantha Barnes", "Thomas Fisher", "Alexandra Rivera",
    "Patrick Sullivan", "Morgan Price", "Marcus Chen", "Rachel Kim", "Vincent Lee"
]

READING_MOODS = {
    "Adventurous": ["Fantasy", "Science Fiction", "Thriller", "Young Adult"],
    "Romantic": ["Romance", "Contemporary Fiction", "Memoir"],
    "Intellectual": ["Non-Fiction", "Psychology", "Self-Help", "Business", "Philosophy", "Biography"],
    "Thrilling": ["Horror", "Thriller", "Mystery", "True Crime", "Dystopian Fiction"],
    "Classic Vibes": ["Classic Fiction", "Historical Fiction", "Magical Realism"],
    "Emotional": ["Contemporary Fiction", "Memoir", "Romance", "Young Adult"],
    "Escapist": ["Fantasy", "Magical Realism", "Science Fiction", "Young Adult"],
}


@dataclass
class BookRecommendation:
    title: str
    author: str
    genre: str
    score: float
    rating: float
    ratings_count: int
    year: int
    bestseller: bool
    reason: str = ""
    image_url: str = ""
    book_id: str = ""


def get_star_rating(rating: float) -> str:
    return f"{rating:.2f}/5"


def format_number(num: int) -> str:
    if num >= 1_000_000:
        return f"{num/1_000_000:.1f}M"
    elif num >= 1_000:
        return f"{num/1_000:.0f}K"
    return str(num)


def format_exact_number(num: int | None) -> str:
    if num is None:
        return "N/A"
    return f"{int(num):,}".replace(",", ".")


def format_bytes(num_bytes: int) -> str:
    value = float(num_bytes)
    for unit in ["B", "KB", "MB", "GB"]:
        if value < 1024 or unit == "GB":
            return f"{value:.1f} {unit}" if unit != "B" else f"{int(value)} B"
        value /= 1024
    return f"{value:.1f} GB"


APP_ROOT = Path(__file__).resolve().parents[1]

FULL_UCSD_SOURCE_COUNTS = {
    "books_records": 2_360_655,
    "ratings_records": 228_648_342,
    "users_records": 876_145,
}


def _candidate_data_dirs() -> list[Path]:
    candidates = []
    env_dir = os.environ.get("BRS_DATA_DIR")
    if env_dir:
        candidates.append(Path(env_dir))

    candidates.extend(
        [
            APP_ROOT / "data",
            APP_ROOT.parent / "book-recommendation-system" / "data",
            APP_ROOT.parent / "BRS+" / "data",
        ]
    )
    for sibling in APP_ROOT.parent.iterdir():
        if sibling.is_dir():
            candidates.append(sibling / "data")

    deduped = []
    seen = set()
    for candidate in candidates:
        try:
            resolved = candidate.resolve()
        except OSError:
            resolved = candidate
        if resolved not in seen:
            seen.add(resolved)
            deduped.append(resolved)
    return deduped


def _score_data_dir(data_dir: Path) -> tuple[int, int]:
    books_path = data_dir / "goodreads_books.json.gz"
    ratings_path = data_dir / "goodreads_interactions.csv"
    if not books_path.exists() or not ratings_path.exists():
        return (0, 0)
    total_size = books_path.stat().st_size + ratings_path.stat().st_size
    return (1, total_size)


def discover_data_dir() -> Path:
    candidates = _candidate_data_dirs()
    best = max(candidates, key=_score_data_dir, default=APP_ROOT / "data")
    return best if _score_data_dir(best)[0] else APP_ROOT / "data"


DATA_DIR = discover_data_dir()


@st.cache_data(show_spinner=False)
def inspect_data_files() -> dict:
    books_path = DATA_DIR / "goodreads_books.json.gz"
    ratings_path = DATA_DIR / "goodreads_interactions.csv"
    small_file_limit = 50 * 1024 * 1024
    expected_books_size = 500 * 1024 * 1024
    expected_ratings_size = 500 * 1024 * 1024

    def line_count(path: Path, compressed: bool = False) -> int | None:
        if not path.exists() or path.stat().st_size > small_file_limit:
            return None
        opener = gzip.open if compressed else open
        with opener(path, "rt", encoding="utf-8", errors="ignore") as handle:
            return sum(1 for _ in handle)

    books_size = books_path.stat().st_size if books_path.exists() else 0
    ratings_size = ratings_path.stat().st_size if ratings_path.exists() else 0
    likely_full_ucsd = books_size >= expected_books_size and ratings_size >= expected_ratings_size
    books_records = line_count(books_path, compressed=True)
    rating_lines = line_count(ratings_path, compressed=False)
    ratings_records = max(0, rating_lines - 1) if rating_lines is not None else None
    users_records = None

    if likely_full_ucsd:
        books_records = books_records or FULL_UCSD_SOURCE_COUNTS["books_records"]
        ratings_records = ratings_records or FULL_UCSD_SOURCE_COUNTS["ratings_records"]
        users_records = FULL_UCSD_SOURCE_COUNTS["users_records"]

    return {
        "data_dir": str(DATA_DIR),
        "books_path": str(books_path),
        "ratings_path": str(ratings_path),
        "books_exists": books_path.exists(),
        "ratings_exists": ratings_path.exists(),
        "books_size": books_size,
        "ratings_size": ratings_size,
        "books_records": books_records,
        "ratings_records": ratings_records,
        "users_records": users_records,
        "likely_full_ucsd": likely_full_ucsd,
    }


def infer_data_source_label(audit: dict) -> str:
    if audit.get("likely_full_ucsd"):
        return "UCSD Book Graph đầy đủ"
    if audit.get("books_exists") and audit.get("ratings_exists"):
        return "Goodreads subset trong thư mục data"
    return "Catalog demo dự phòng"


def compact_data_source_label(audit: dict) -> str:
    if audit.get("likely_full_ucsd"):
        return "Full UCSD"
    if audit.get("books_exists") and audit.get("ratings_exists"):
        return "Goodreads subset"
    return "Demo fallback"


def render_data_audit_panel(books_df: pd.DataFrame, ratings_df: pd.DataFrame, audit: dict) -> None:
    source_label = infer_data_source_label(audit)
    density = len(ratings_df) / max(1, books_df["book_id"].nunique() * ratings_df["user_id"].nunique())
    books_records = audit.get("books_records")
    ratings_records = audit.get("ratings_records")
    users_records = audit.get("users_records")
    source_note = (
        f"Nguồn gốc: {format_exact_number(books_records)} sách, "
        f"{format_exact_number(users_records)} người dùng, "
        f"{format_exact_number(ratings_records)} tương tác."
        if audit.get("likely_full_ucsd")
        else (
            f"Nguồn hiện tại: {format_exact_number(books_records)} sách, "
            f"{format_exact_number(ratings_records)} tương tác."
            if books_records is not None and ratings_records is not None
            else "Nguồn hiện tại là file lớn/subset chưa đếm toàn bộ khi render."
        )
    )
    status_note = (
        "App tạo working set có kiểm soát từ nguồn gốc để mô hình phản hồi nhanh khi demo."
        if audit.get("likely_full_ucsd")
        else "Đây là subset/working set định dạng Goodreads; nếu có file UCSD đầy đủ, app sẽ tự ưu tiên thư mục dữ liệu lớn hơn."
    )

    st.markdown(
        f"""
        <div class="info-box">
            <strong>Data audit:</strong> {source_label}. App đang nạp
            <strong>{format_exact_number(len(books_df))}</strong> sách,
            <strong>{format_exact_number(ratings_df['user_id'].nunique())}</strong> người dùng và
            <strong>{format_exact_number(len(ratings_df))}</strong> lượt đánh giá.
            <br>
            <span style="color: {COLORS['text_light']};">
                Thư mục dữ liệu: {escape_html(audit.get('data_dir', str(DATA_DIR)))}.
                {source_note} Mật độ ma trận working set: {density:.2%}. {status_note}
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _extract_text_value(value, default: str = "") -> str:
    if value is None:
        return default

    if isinstance(value, dict):
        for key in ("name", "genre", "tag", "label"):
            candidate = value.get(key)
            if candidate:
                return str(candidate).strip() or default
        return default

    if isinstance(value, list):
        for item in value:
            if isinstance(item, dict):
                candidate = item.get("name") or item.get("genre") or item.get("tag") or item.get("label")
            else:
                candidate = item

            candidate_text = str(candidate).strip() if candidate is not None else ""
            if candidate_text and candidate_text.lower() not in {"fiction", "books", "to-read", "currently-reading"}:
                return candidate_text
        return default

    text = str(value).strip()
    if not text or text.lower() in {"nan", "none"}:
        return default
    return text.split(",")[0].strip() or default


def _normalize_books_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    if "book_id" in df.columns:
        df["book_id"] = df["book_id"].astype(str)

    if "author" not in df.columns:
        if "authors" in df.columns:
            df["author"] = df["authors"].apply(lambda value: _extract_text_value(value, "Unknown"))
        else:
            df["author"] = "Unknown"
    else:
        df["author"] = df["author"].apply(lambda value: _extract_text_value(value, "Unknown"))

    if "rating" not in df.columns:
        if "avg_rating" in df.columns:
            df["rating"] = pd.to_numeric(df["avg_rating"], errors="coerce")
        elif "average_rating" in df.columns:
            df["rating"] = pd.to_numeric(df["average_rating"], errors="coerce")
        else:
            df["rating"] = np.nan
    else:
        df["rating"] = pd.to_numeric(df["rating"], errors="coerce")

    if "ratings_count" not in df.columns:
        if "n_ratings" in df.columns:
            df["ratings_count"] = pd.to_numeric(df["n_ratings"], errors="coerce")
        elif "ratingsCount" in df.columns:
            df["ratings_count"] = pd.to_numeric(df["ratingsCount"], errors="coerce")
        else:
            df["ratings_count"] = 0
    else:
        df["ratings_count"] = pd.to_numeric(df["ratings_count"], errors="coerce")

    if "year" not in df.columns:
        for source in ("publication_year", "pub_year", "published", "year"):
            if source in df.columns:
                df["year"] = pd.to_numeric(df[source], errors="coerce")
                break
        else:
            df["year"] = np.nan
    else:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")

    if "genre" not in df.columns:
        for source in ("genres", "popular_shelves", "shelves", "categories"):
            if source in df.columns:
                df["genre"] = df[source].apply(lambda value: _extract_genre_value(value, "General Fiction"))
                break
        else:
            df["genre"] = "General Fiction"
    else:
        df["genre"] = df["genre"].apply(lambda value: _extract_genre_value(value, "General Fiction"))

    if "bestseller" not in df.columns:
        ratings_count = pd.to_numeric(df["ratings_count"], errors="coerce").fillna(0)
        bestseller_cutoff = max(ratings_count.quantile(0.85), 10_000) if len(ratings_count) else 0
        df["bestseller"] = ratings_count >= bestseller_cutoff
    else:
        df["bestseller"] = df["bestseller"].fillna(False).astype(bool)

    df["title"] = df["title"].fillna("Unknown Title").astype(str).str.strip()
    df["author"] = df["author"].fillna("Unknown").astype(str).str.strip()
    df["genre"] = df["genre"].fillna("General Fiction").astype(str).str.strip().replace("", "General Fiction")
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce").fillna(0).clip(0, 5)
    df["ratings_count"] = pd.to_numeric(df["ratings_count"], errors="coerce").fillna(0).astype(int)
    df["year"] = pd.to_numeric(df["year"], errors="coerce").fillna(0).astype(int)
    if "image_url" not in df.columns:
        df["image_url"] = ""
    df["image_url"] = df["image_url"].fillna("").astype(str)

    df = df.dropna(subset=["book_id", "title"])
    df = df[df["title"].str.len() > 0].copy()
    display_columns = [
        "book_id",
        "title",
        "author",
        "genre",
        "rating",
        "ratings_count",
        "year",
        "bestseller",
        "image_url",
    ]
    return df[display_columns].reset_index(drop=True)


def _normalize_ratings_df(df: pd.DataFrame, books_df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["user_id"] = df["user_id"].astype(str)
    df["book_id"] = df["book_id"].astype(str)
    df["rating"] = pd.to_numeric(df["rating"], errors="coerce")
    df = df.dropna(subset=["rating"])
    df["rating"] = df["rating"].clip(1, 5).astype(float)

    if not books_df.empty and "book_id" in books_df.columns:
        valid_book_ids = set(books_df["book_id"].astype(str))
        df = df[df["book_id"].isin(valid_book_ids)].copy()

    return df.reset_index(drop=True)


def _build_demo_books_df() -> pd.DataFrame:
    demo_books = []
    for index, book in enumerate(FAMOUS_BOOKS):
        row = dict(book)
        row["book_id"] = f"demo_{index:04d}"
        row.setdefault("image_url", "")
        demo_books.append(row)

    return _normalize_books_df(pd.DataFrame(demo_books))


def _build_demo_ratings_df(books_df: pd.DataFrame, seed: int = 42) -> pd.DataFrame:
    rng = random.Random(seed)
    if books_df.empty:
        return pd.DataFrame(columns=["user_id", "book_id", "rating"])

    genres = sorted(books_df["genre"].dropna().unique().tolist())
    rows = []
    for user_index in range(140):
        primary_genre = genres[user_index % len(genres)]
        secondary_genre = genres[(user_index * 3 + 5) % len(genres)]
        preferred_genres = {primary_genre, secondary_genre}
        candidate_books = books_df[
            books_df["genre"].isin(preferred_genres)
            | (books_df["rating"] >= books_df["rating"].quantile(0.72))
        ]
        sample_size = min(28, len(candidate_books))
        sampled_books = candidate_books.sample(n=sample_size, random_state=seed + user_index)

        for _, book in sampled_books.iterrows():
            rating_floor = 4 if book["genre"] in preferred_genres else 3
            rating = min(5, max(1, rating_floor + rng.choice([-1, 0, 0, 1])))
            rows.append(
                {
                    "user_id": f"user_{user_index:03d}",
                    "book_id": str(book["book_id"]),
                    "rating": float(rating),
                }
            )

    return _normalize_ratings_df(pd.DataFrame(rows), books_df)


@st.cache_data
def load_books_data():
    books_path = DATA_DIR / "goodreads_books.json.gz"
    max_books = 5000
    scan_limit = 100_000

    if books_path.exists():
        top_books: list[dict] = []
        try:
            record_index = 0
            with gzip.open(books_path, "rt", encoding="utf-8") as file_handle:
                for line in file_handle:
                    try:
                        record = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    ratings_count = record.get("ratings_count") or record.get("n_ratings") or 0
                    try:
                        ratings_count_value = int(ratings_count)
                    except (TypeError, ValueError):
                        ratings_count_value = 0

                    book_row = {
                        "book_id": str(
                            record.get("book_id")
                            or record.get("bookId")
                            or record.get("id")
                            or record.get("isbn")
                            or record.get("ISBN")
                            or len(top_books)
                        ),
                        "title": record.get("title") or record.get("name") or record.get("book_title") or "Unknown Title",
                        "authors": record.get("authors") or record.get("author") or record.get("writers"),
                        "image_url": record.get("image_url") or record.get("cover_url") or record.get("cover_image"),
                        "avg_rating": record.get("average_rating") or record.get("avg_rating") or record.get("rating"),
                        "n_ratings": ratings_count_value,
                        "year": record.get("publication_year") or record.get("pub_year") or record.get("published") or record.get("year"),
                        "genres": record.get("genres") or record.get("genre") or record.get("popular_shelves") or record.get("shelves"),
                    }

                    entry = (ratings_count_value, record_index, book_row)
                    if len(top_books) < 5000:
                        heapq.heappush(top_books, entry)
                    elif ratings_count_value > top_books[0][0]:
                        heapq.heapreplace(top_books, entry)

                    record_index += 1
                    if record_index >= scan_limit:
                        break
        except Exception:
            raise RuntimeError(f"Không thể đọc dữ liệu sách thật từ {books_path}")

        if top_books:
            df = pd.DataFrame([row for _, _, row in sorted(top_books, key=lambda item: item[0], reverse=True)])[:max_books]
            return _normalize_books_df(df)

    return _build_demo_books_df()


@st.cache_data
def generate_user_ratings(books_df, seed=42):
    """Load real Goodreads user ratings for the current books dataset."""
    ratings_path = DATA_DIR / "goodreads_interactions.csv"
    max_ratings = 30_000
    max_scan_rows = 5_000_000

    if ratings_path.exists() and not books_df.empty:
        try:
            book_ids = set(books_df["book_id"].astype(str))
            filtered_chunks = []
            total_rows = 0
            scanned_rows = 0
            reader = pd.read_csv(
                ratings_path,
                usecols=["user_id", "book_id", "rating"],
                chunksize=500_000,
            )
            for chunk in reader:
                scanned_rows += len(chunk)
                chunk["user_id"] = chunk["user_id"].astype(str)
                chunk["book_id"] = chunk["book_id"].astype(str)
                chunk = chunk[chunk["book_id"].isin(book_ids)]
                if chunk.empty:
                    if scanned_rows >= max_scan_rows:
                        break
                    continue

                filtered_chunks.append(chunk)
                total_rows += len(chunk)
                if total_rows >= max_ratings or scanned_rows >= max_scan_rows:
                    break

            if filtered_chunks:
                ratings_df = pd.concat(filtered_chunks, ignore_index=True)
                if len(ratings_df) > max_ratings:
                    ratings_df = ratings_df.sample(n=max_ratings, random_state=seed)
                return _normalize_ratings_df(ratings_df, books_df)
        except Exception:
            raise RuntimeError(f"Không thể đọc dữ liệu tương tác Goodreads thật từ {ratings_path}")

    return _build_demo_ratings_df(books_df, seed)


def _popularity_norm(books_df: pd.DataFrame) -> pd.Series:
    values = np.log1p(pd.to_numeric(books_df["ratings_count"], errors="coerce").fillna(0))
    max_value = float(values.max()) if len(values) else 0.0
    if max_value <= 0:
        return pd.Series(0.0, index=books_df.index)
    return values / max_value


def _apply_feedback_to_candidates(
    candidates: pd.DataFrame,
    books_df: pd.DataFrame,
    feedback_votes: dict | None = None,
    saved_items: list | None = None,
) -> pd.DataFrame:
    if candidates.empty:
        return candidates

    feedback_profile = build_feedback_profile(feedback_votes, saved_items, books_df)
    if not feedback_profile["excluded_ids"] and not feedback_profile["positive_genres"] and not feedback_profile["negative_genres"]:
        return candidates

    candidates = candidates.copy()
    candidates["book_id"] = candidates["book_id"].astype(str)
    if feedback_profile["excluded_ids"]:
        candidates = candidates[~candidates["book_id"].isin(feedback_profile["excluded_ids"])].copy()
    if candidates.empty:
        return candidates

    boost = pd.Series(0.0, index=candidates.index)
    if feedback_profile["positive_genres"]:
        boost += candidates["genre"].isin(feedback_profile["positive_genres"]).astype(float) * 0.12
    if feedback_profile["positive_authors"]:
        boost += candidates["author"].isin(feedback_profile["positive_authors"]).astype(float) * 0.10
    if feedback_profile["negative_genres"]:
        boost -= candidates["genre"].isin(feedback_profile["negative_genres"]).astype(float) * 0.10
    if feedback_profile["negative_authors"]:
        boost -= candidates["author"].isin(feedback_profile["negative_authors"]).astype(float) * 0.08

    positive_mask = boost > 0
    negative_mask = boost < 0
    candidates["_score"] = (candidates["_score"] + boost).clip(0.01, 0.99)
    candidates.loc[positive_mask, "_reason"] = "Tương tự những sách bạn vừa thích hoặc lưu kệ"
    candidates.loc[negative_mask, "_reason"] = "Đã giảm điểm do phản hồi không hợp gần đây"
    return candidates


def _score_user_candidates(
    user_id: str,
    ratings_df: pd.DataFrame,
    books_df: pd.DataFrame,
    exclude_rated: bool = True,
    weights: dict[str, float] | None = None,
    feedback_votes: dict | None = None,
    saved_items: list | None = None,
) -> pd.DataFrame:
    weights = weights or DEFAULT_HYBRID_WEIGHTS
    user_id = str(user_id)
    user_ratings = ratings_df[ratings_df["user_id"].astype(str) == user_id].copy()
    user_ratings["book_id"] = user_ratings["book_id"].astype(str)
    rated_books = set(user_ratings["book_id"].astype(str).tolist())
    history_catalog = books_df[["book_id", "genre", "author", "rating"]].copy()
    history_catalog["book_id"] = history_catalog["book_id"].astype(str)
    history = user_ratings.merge(
        history_catalog,
        on="book_id",
        how="left",
        suffixes=("_user", "_book"),
    )

    if history.empty:
        genre_weights = {}
        author_weights = {}
    else:
        history["preference"] = ((history["rating_user"] - 3.0) / 2.0).clip(0, 1)
        genre_weights = history.groupby("genre")["preference"].mean().to_dict()
        author_weights = history.groupby("author")["preference"].mean().to_dict()
        if not any(value > 0 for value in genre_weights.values()):
            genre_weights = (history.groupby("genre")["rating_user"].mean() / 5.0).to_dict()

    candidates = books_df.copy()
    candidates["book_id"] = candidates["book_id"].astype(str)
    if exclude_rated and rated_books:
        candidates = candidates[~candidates["book_id"].isin(rated_books)].copy()

    if candidates.empty:
        return candidates.assign(_score=[], _reason=[], _novelty=[])

    candidates["_popularity"] = _popularity_norm(candidates)
    candidates["_rating_norm"] = pd.to_numeric(candidates["rating"], errors="coerce").fillna(0) / 5.0
    candidates["_novelty"] = (1.0 - candidates["_popularity"]).clip(0, 1)
    candidates["_genre_affinity"] = candidates["genre"].map(genre_weights).fillna(0.0)
    candidates["_author_affinity"] = candidates["author"].map(author_weights).fillna(0.0)

    candidates["_raw_score"] = (
        weights.get("genre", 0.0) * candidates["_genre_affinity"]
        + weights.get("rating", 0.0) * candidates["_rating_norm"]
        + weights.get("popularity", 0.0) * candidates["_popularity"]
        + weights.get("author", 0.0) * candidates["_author_affinity"]
        + weights.get("novelty", 0.0) * candidates["_novelty"]
    )

    min_score = float(candidates["_raw_score"].min())
    max_score = float(candidates["_raw_score"].max())
    if max_score > min_score:
        candidates["_score"] = 0.45 + 0.54 * ((candidates["_raw_score"] - min_score) / (max_score - min_score))
    else:
        candidates["_score"] = 0.62

    def build_reason(row: pd.Series) -> str:
        if row["_genre_affinity"] > 0:
            return f"Based on your interest in {row['genre']}"
        if row["_author_affinity"] > 0:
            return f"Same author: {row['author']}"
        if row["bestseller"]:
            return "Sách phổ biến có điểm đánh giá cao"
        return "Cân bằng giữa điểm đánh giá, độ mới lạ và mức độ phổ biến"

    candidates["_reason"] = candidates.apply(build_reason, axis=1)
    candidates = _apply_feedback_to_candidates(candidates, books_df, feedback_votes, saved_items)
    return candidates.sort_values(["_score", "rating", "ratings_count"], ascending=False)


def _rank_scored_candidates(
    scored_df: pd.DataFrame,
    n: int,
    diversity_weight: float = 0.0,
) -> list[dict]:
    candidates = scored_df.to_dict("records")
    selected: list[dict] = []
    genre_counts: Counter = Counter()

    while candidates and len(selected) < n:
        best_index = 0
        best_score = -float("inf")
        for index, candidate in enumerate(candidates):
            genre = candidate.get("genre", "")
            genre_penalty = genre_counts[genre] / max(1, len(selected))
            adjusted_score = float(candidate.get("_score", 0)) - diversity_weight * genre_penalty
            if adjusted_score > best_score:
                best_index = index
                best_score = adjusted_score

        selected_candidate = candidates.pop(best_index)
        genre_counts[selected_candidate.get("genre", "")] += 1
        selected.append(selected_candidate)

    return selected


def _rank_recommendations_for_diversity(
    recommendations: List[BookRecommendation],
    n: int,
    diversity_weight: float,
) -> List[BookRecommendation]:
    candidates = list(recommendations)
    selected: List[BookRecommendation] = []
    genre_counts: Counter = Counter()

    while candidates and len(selected) < n:
        best_index = 0
        best_score = -float("inf")
        for index, rec in enumerate(candidates):
            genre_penalty = genre_counts[rec.genre] / max(1, len(selected))
            adjusted_score = rec.score - diversity_weight * genre_penalty
            if adjusted_score > best_score:
                best_index = index
                best_score = adjusted_score

        selected_rec = candidates.pop(best_index)
        genre_counts[selected_rec.genre] += 1
        selected.append(selected_rec)

    return selected


def apply_feedback_to_recommendations(
    recommendations: List[BookRecommendation],
    books_df: pd.DataFrame,
    feedback_votes: dict | None,
    saved_items: list | None,
    n: int,
) -> List[BookRecommendation]:
    if not recommendations:
        return []

    feedback_profile = build_feedback_profile(feedback_votes, saved_items, books_df)
    updated: List[BookRecommendation] = []
    for rec in recommendations:
        book_key = recommendation_identity(rec)
        if book_key in feedback_profile["excluded_ids"]:
            continue

        adjustment = 0.0
        reason = rec.reason
        if rec.genre in feedback_profile["positive_genres"]:
            adjustment += 0.08
            reason = "Tương tự thể loại bạn vừa thích hoặc lưu kệ"
        if rec.author in feedback_profile["positive_authors"]:
            adjustment += 0.08
            reason = "Tương tự tác giả bạn vừa thích hoặc lưu kệ"
        if rec.genre in feedback_profile["negative_genres"]:
            adjustment -= 0.07
        if rec.author in feedback_profile["negative_authors"]:
            adjustment -= 0.07

        updated.append(replace(rec, score=round(min(0.99, max(0.01, rec.score + adjustment)), 3), reason=reason))

    updated.sort(key=lambda item: (item.score, item.rating, item.ratings_count), reverse=True)
    return updated[:n]


def _candidate_to_recommendation(candidate: dict) -> BookRecommendation:
    return BookRecommendation(
        title=candidate["title"],
        author=candidate["author"],
        genre=candidate["genre"],
        image_url=candidate.get("image_url", ""),
        score=round(float(candidate.get("_score", 0.0)), 3),
        rating=float(candidate["rating"]),
        ratings_count=int(candidate["ratings_count"]),
        year=int(candidate["year"]),
        bestseller=bool(candidate["bestseller"]),
        reason=candidate.get("_reason", ""),
        book_id=str(candidate.get("book_id", "")),
    )


def recommendation_list_metrics(recommendations: List[BookRecommendation], books_df: pd.DataFrame) -> dict:
    if not recommendations:
        return {"diversity": 0.0, "avg_rating": 0.0, "novelty": 0.0, "bestseller_share": 0.0}

    max_popularity = float(np.log1p(pd.to_numeric(books_df["ratings_count"], errors="coerce").fillna(0)).max())
    novelty_values = []
    for rec in recommendations:
        if max_popularity > 0:
            novelty_values.append(1.0 - (np.log1p(max(0, rec.ratings_count)) / max_popularity))
        else:
            novelty_values.append(0.0)

    return {
        "diversity": len({rec.genre for rec in recommendations}) / len(recommendations),
        "avg_rating": float(np.mean([rec.rating for rec in recommendations])),
        "novelty": float(np.mean(novelty_values)),
        "bestseller_share": sum(1 for rec in recommendations if rec.bestseller) / len(recommendations),
    }


def render_recommendation_quality(recommendations: List[BookRecommendation], books_df: pd.DataFrame) -> None:
    metrics = recommendation_list_metrics(recommendations, books_df)
    cols = st.columns(4)
    with cols[0]:
        display_metric_card(f"{metrics['diversity']:.0%}", "Độ đa dạng thể loại")
    with cols[1]:
        display_metric_card(f"{metrics['avg_rating']:.2f}", "Điểm sách trung bình")
    with cols[2]:
        display_metric_card(f"{metrics['novelty']:.0%}", "Độ mới lạ")
    with cols[3]:
        display_metric_card(f"{metrics['bestseller_share']:.0%}", "Tỷ lệ sách phổ biến")


def exploration_to_diversity_weight(
    exploration_level: float,
    base: float = 0.06,
    max_add: float = 0.18,
) -> float:
    normalized = min(1.0, max(0.0, float(exploration_level) / 100.0))
    return base + normalized * max_add


def render_user_model_explainability(
    user_id: str,
    ratings_df: pd.DataFrame,
    books_df: pd.DataFrame,
    recommendations: List[BookRecommendation],
    exploration_level: float,
) -> None:
    if not user_id or not recommendations:
        return

    user_ratings = ratings_df[ratings_df["user_id"].astype(str) == str(user_id)].copy()
    user_ratings["book_id"] = user_ratings["book_id"].astype(str)
    catalog = books_df[["book_id", "title", "author", "genre"]].copy()
    catalog["book_id"] = catalog["book_id"].astype(str)
    history = user_ratings.merge(
        catalog,
        on="book_id",
        how="left",
    ).dropna(subset=["genre"])
    if history.empty:
        return

    positive_history = history[history["rating"] >= max(4.0, float(history["rating"].mean()))]
    if positive_history.empty:
        positive_history = history.nlargest(min(5, len(history)), "rating")

    top_genres = (
        positive_history.groupby("genre")
        .agg(count=("book_id", "count"), mean_rating=("rating", "mean"))
        .assign(strength=lambda df: df["count"] * df["mean_rating"])
        .sort_values(["strength", "count", "mean_rating"], ascending=False)
        .head(4)
    )
    top_authors = (
        positive_history.groupby("author")
        .agg(count=("book_id", "count"), mean_rating=("rating", "mean"))
        .sort_values(["count", "mean_rating"], ascending=False)
        .head(3)
    )

    genre_chips = "".join(
        f'<span class="explain-chip">{escape_html(translate_genre(str(genre)))} · {int(row["count"])} đánh giá cao</span>'
        for genre, row in top_genres.iterrows()
    )
    author_chips = "".join(
        f'<span class="explain-chip">{escape_html(str(author))}</span>'
        for author in top_authors.index
    )
    if not genre_chips:
        genre_chips = '<span class="explain-chip">Chưa đủ tín hiệu thể loại</span>'
    if not author_chips:
        author_chips = '<span class="explain-chip">Tác giả phân tán</span>'

    list_metrics = recommendation_list_metrics(recommendations, books_df)
    diversity_weight = exploration_to_diversity_weight(exploration_level)
    factors = [
        ("Thể loại", 0.46),
        ("Điểm sách", 0.20),
        ("Phổ biến", 0.16),
        ("Tác giả", 0.12),
        ("Độ mới lạ", 0.06),
        ("Xếp hạng lại", diversity_weight),
    ]
    factor_rows = "".join(
        f'<div class="factor-row">'
        f'<div>{escape_html(label)}</div>'
        f'<div class="factor-track"><div class="factor-fill" style="width: {min(100, value * 100):.0f}%;"></div></div>'
        f'<div>{value:.0%}</div>'
        f'</div>'
        for label, value in factors
    )

    model_html = (
        '<div class="model-explain-panel">'
        '<div class="model-explain-grid">'
        '<div>'
        '<div class="model-explain-title">Hồ sơ sở thích đã học</div>'
        '<div class="model-explain-note">'
        'Hệ thống ưu tiên tín hiệu từ những cuốn người dùng đánh giá cao, sau đó loại các sách đã đọc khỏi tập ứng viên.'
        '</div>'
        f'<div class="explain-chip-row">{genre_chips}</div>'
        '<div class="model-explain-title">Tác giả có tín hiệu tốt</div>'
        f'<div class="explain-chip-row">{author_chips}</div>'
        '</div>'
        '<div>'
        '<div class="model-explain-title">Công thức xếp hạng đang dùng</div>'
        f'{factor_rows}'
        '<div class="model-explain-note">'
        f'Mức khám phá hiện tại: <strong>{float(exploration_level):.0f}/100</strong>. '
        f'Danh sách kết quả có <strong>{list_metrics["diversity"]:.0%}</strong> độ đa dạng thể loại '
        f'và <strong>{list_metrics["novelty"]:.0%}</strong> độ mới lạ.'
        '</div>'
        '</div>'
        '</div>'
        '</div>'
    )
    st.markdown(model_html, unsafe_allow_html=True)


def get_cold_start_recommendations(
    preferred_genres: list[str],
    books_df: pd.DataFrame,
    n: int = 10,
    popularity_weight: float = 0.45,
    exploration_level: float = 45.0,
) -> List[BookRecommendation]:
    candidates = books_df.copy()
    candidates["_popularity"] = _popularity_norm(candidates)
    candidates["_rating_norm"] = pd.to_numeric(candidates["rating"], errors="coerce").fillna(0) / 5.0
    candidates["_genre_match"] = candidates["genre"].isin(preferred_genres).astype(float)
    candidates["_novelty"] = 1.0 - candidates["_popularity"]
    candidates["_score"] = (
        0.48 * candidates["_genre_match"]
        + popularity_weight * candidates["_popularity"]
        + 0.22 * candidates["_rating_norm"]
        + max(0.0, 0.30 - popularity_weight / 2) * candidates["_novelty"]
    ).clip(0, 1)
    candidates["_reason"] = candidates["genre"].apply(
        lambda genre: f"Phù hợp với sở thích ban đầu: {translate_genre(genre)}"
    )
    ranked = _rank_scored_candidates(
        candidates.sort_values(["_score", "rating", "ratings_count"], ascending=False),
        n,
        diversity_weight=exploration_to_diversity_weight(exploration_level, base=0.05, max_add=0.18),
    )
    return [_candidate_to_recommendation(candidate) for candidate in ranked]


def _split_ratings_for_evaluation(ratings_df: pd.DataFrame, test_size: float = 0.2, seed: int = 42):
    train_parts = []
    test_parts = []
    for _, group in ratings_df.groupby("user_id"):
        if len(group) < 5:
            train_parts.append(group)
            continue
        shuffled = group.sample(frac=1, random_state=seed)
        n_test = max(1, int(round(len(shuffled) * test_size)))
        test_parts.append(shuffled.head(n_test))
        train_parts.append(shuffled.tail(len(shuffled) - n_test))

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else ratings_df.iloc[0:0].copy()
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else ratings_df.iloc[0:0].copy()
    return train_df, test_df


@st.cache_data(show_spinner=False)
def build_benchmark_sample(
    ratings_df: pd.DataFrame,
    books_df: pd.DataFrame,
    max_users: int = 360,
    max_books: int = 900,
    max_ratings: int = 12_000,
    min_user_ratings: int = 5,
    seed: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    ratings = ratings_df[["user_id", "book_id", "rating"]].copy()
    books = books_df.copy()
    ratings["user_id"] = ratings["user_id"].astype(str)
    ratings["book_id"] = ratings["book_id"].astype(str)
    books["book_id"] = books["book_id"].astype(str)

    valid_book_ids = set(books["book_id"])
    ratings = ratings[ratings["book_id"].isin(valid_book_ids)].copy()
    if ratings.empty:
        return ratings, books.iloc[0:0].copy(), {
            "source_books": len(books_df),
            "source_ratings": len(ratings_df),
            "benchmark_books": 0,
            "benchmark_ratings": 0,
            "benchmark_users": 0,
            "strategy": "Không đủ dữ liệu hợp lệ để tạo benchmark sample",
        }

    user_counts = ratings.groupby("user_id").size().sort_values(ascending=False)
    eligible_users = user_counts[user_counts >= min_user_ratings]
    if eligible_users.empty:
        eligible_users = user_counts

    if len(eligible_users) > max_users:
        top_user_count = min(max(80, max_users // 3), len(eligible_users))
        top_users = eligible_users.head(top_user_count).index.astype(str).tolist()
        remaining_users = eligible_users.drop(index=top_users, errors="ignore").index.astype(str).tolist()
        random_user_count = max_users - len(top_users)
        if random_user_count > 0 and remaining_users:
            sampled_users = (
                pd.Series(remaining_users)
                .sample(n=min(random_user_count, len(remaining_users)), random_state=seed)
                .astype(str)
                .tolist()
            )
        else:
            sampled_users = []
        selected_users = set(top_users + sampled_users)
    else:
        selected_users = set(eligible_users.index.astype(str))

    sampled_ratings = ratings[ratings["user_id"].isin(selected_users)].copy()
    sampled_ratings = (
        sampled_ratings.sample(frac=1, random_state=seed)
        .groupby("user_id", group_keys=False)
        .head(35)
        .reset_index(drop=True)
    )

    book_counts = sampled_ratings.groupby("book_id").size().sort_values(ascending=False)
    selected_book_ids = set(book_counts.head(max_books).index.astype(str))
    sampled_ratings = sampled_ratings[sampled_ratings["book_id"].isin(selected_book_ids)].copy()
    per_user_counts = sampled_ratings.groupby("user_id").size()
    keep_users = set(per_user_counts[per_user_counts >= min_user_ratings].index.astype(str))
    sampled_ratings = sampled_ratings[sampled_ratings["user_id"].isin(keep_users)].copy()

    if len(sampled_ratings) > max_ratings:
        sampled_ratings = sampled_ratings.sample(n=max_ratings, random_state=seed).copy()
        per_user_counts = sampled_ratings.groupby("user_id").size()
        keep_users = set(per_user_counts[per_user_counts >= min_user_ratings].index.astype(str))
        sampled_ratings = sampled_ratings[sampled_ratings["user_id"].isin(keep_users)].copy()

    selected_book_ids = set(sampled_ratings["book_id"].astype(str).unique())
    sampled_books = books[books["book_id"].isin(selected_book_ids)].copy()
    sampled_books = sampled_books.sort_values("ratings_count", ascending=False).reset_index(drop=True)
    sampled_ratings = _normalize_ratings_df(sampled_ratings, sampled_books)

    summary = {
        "source_books": len(books_df),
        "source_ratings": len(ratings_df),
        "source_users": ratings_df["user_id"].nunique() if "user_id" in ratings_df.columns else 0,
        "benchmark_books": sampled_books["book_id"].nunique(),
        "benchmark_ratings": len(sampled_ratings),
        "benchmark_users": sampled_ratings["user_id"].nunique(),
        "max_users": max_users,
        "max_books": max_books,
        "max_ratings": max_ratings,
        "strategy": "Chọn user đủ lịch sử, giữ sách có nhiều tương tác trong working set, rồi giới hạn rating để benchmark phản hồi nhanh",
    }
    return sampled_ratings.reset_index(drop=True), sampled_books.reset_index(drop=True), summary


def _ranking_metrics(recommended: list[str], relevant: set[str], k: int) -> dict:
    top_k = recommended[:k]
    hits = [item_id for item_id in top_k if item_id in relevant]
    precision = len(hits) / k if k else 0.0
    recall = len(hits) / len(relevant) if relevant else 0.0
    hit_rate = 1.0 if hits else 0.0

    running_hits = 0
    precision_at_hits = []
    reciprocal_rank = 0.0
    for index, item_id in enumerate(top_k, start=1):
        if item_id in relevant:
            running_hits += 1
            precision_at_hits.append(running_hits / index)
            if reciprocal_rank == 0.0:
                reciprocal_rank = 1.0 / index

    average_precision = (
        sum(precision_at_hits) / min(len(relevant), k)
        if relevant and precision_at_hits
        else 0.0
    )
    dcg = sum(1.0 / np.log2(index + 2) for index, item_id in enumerate(top_k) if item_id in relevant)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / np.log2(index + 2) for index in range(ideal_hits))
    ndcg = dcg / idcg if idcg else 0.0

    return {
        "Precision@K": precision,
        "Recall@K": recall,
        "NDCG@K": ndcg,
        "Hit Rate@K": hit_rate,
        "MAP@K": average_precision,
        "MRR@K": reciprocal_rank,
    }


def _gini_coefficient(values: list[float]) -> float:
    array = np.asarray(values, dtype=float)
    if array.size == 0:
        return 0.0
    array = np.clip(array, 0, None)
    total = float(array.sum())
    if total <= 0:
        return 0.0
    array = np.sort(array)
    n = array.size
    index = np.arange(1, n + 1)
    return float((2 * np.sum(index * array) / (n * total)) - ((n + 1) / n))


def _inter_list_diversity(recommendation_lists: list[list[str]]) -> float:
    rec_sets = [set(rec_ids) for rec_ids in recommendation_lists if rec_ids]
    if len(rec_sets) < 2:
        return 0.0

    total_distance = 0.0
    pair_count = 0
    for left_index in range(len(rec_sets)):
        for right_index in range(left_index + 1, len(rec_sets)):
            union = rec_sets[left_index] | rec_sets[right_index]
            if not union:
                continue
            similarity = len(rec_sets[left_index] & rec_sets[right_index]) / len(union)
            total_distance += 1.0 - similarity
            pair_count += 1
    return total_distance / pair_count if pair_count else 0.0


def _genre_distribution(item_ids: list[str], book_lookup: pd.DataFrame) -> dict[str, int]:
    counts: Counter = Counter()
    for item_id in item_ids:
        if item_id in book_lookup.index:
            counts[str(book_lookup.loc[item_id, "genre"])] += 1
    return dict(counts)


def _jensen_shannon_divergence(left_counts: dict[str, int], right_counts: dict[str, int]) -> float:
    keys = sorted(set(left_counts) | set(right_counts))
    if not keys:
        return 0.0

    left = np.asarray([left_counts.get(key, 0) for key in keys], dtype=float)
    right = np.asarray([right_counts.get(key, 0) for key in keys], dtype=float)
    if left.sum() <= 0 or right.sum() <= 0:
        return 0.0

    left = left / left.sum()
    right = right / right.sum()
    middle = 0.5 * (left + right)

    def kl_divergence(source: np.ndarray, target: np.ndarray) -> float:
        mask = source > 0
        return float(np.sum(source[mask] * np.log2(source[mask] / target[mask])))

    return 0.5 * kl_divergence(left, middle) + 0.5 * kl_divergence(right, middle)


def _user_history_item_ids(
    user_id: str,
    train_df: pd.DataFrame,
    relevance_threshold: float,
) -> list[str]:
    user_history = train_df[train_df["user_id"].astype(str) == str(user_id)]
    positive_history = user_history[user_history["rating"] >= relevance_threshold]
    if positive_history.empty:
        positive_history = user_history
    return positive_history["book_id"].astype(str).tolist()


def _recommend_popular_ids(user_id: str, train_df: pd.DataFrame, books_df: pd.DataFrame, n: int) -> list[str]:
    rated = set(train_df[train_df["user_id"].astype(str) == str(user_id)]["book_id"].astype(str))
    popularity = (
        train_df.groupby("book_id")
        .agg(mean_rating=("rating", "mean"), rating_count=("rating", "size"))
        .reset_index()
    )
    catalog = books_df.merge(popularity, on="book_id", how="left").fillna({"mean_rating": 0, "rating_count": 0})
    catalog["_popularity"] = np.log1p(catalog["rating_count"])
    max_popularity = float(catalog["_popularity"].max())
    if max_popularity > 0:
        catalog["_popularity"] = catalog["_popularity"] / max_popularity
    catalog["_score"] = 0.68 * catalog["_popularity"] + 0.32 * (catalog["rating"] / 5.0)
    catalog = catalog[~catalog["book_id"].astype(str).isin(rated)]
    return catalog.sort_values(["_score", "rating", "ratings_count"], ascending=False)["book_id"].astype(str).head(n).tolist()


def _recommend_profile_ids(
    user_id: str,
    train_df: pd.DataFrame,
    books_df: pd.DataFrame,
    n: int,
    diversity_weight: float = 0.0,
    weights: dict[str, float] | None = None,
    feedback_votes: dict | None = None,
    saved_items: list | None = None,
) -> list[str]:
    scored = _score_user_candidates(
        user_id,
        train_df,
        books_df,
        exclude_rated=True,
        weights=weights,
        feedback_votes=feedback_votes,
        saved_items=saved_items,
    )
    ranked = _rank_scored_candidates(scored, n, diversity_weight=diversity_weight)
    return [str(candidate["book_id"]) for candidate in ranked]


def _build_svd_model(
    train_df: pd.DataFrame,
    books_df: pd.DataFrame,
    n_components: int = 24,
    random_state: int = 42,
) -> dict:
    catalog = books_df.copy()
    catalog["book_id"] = catalog["book_id"].astype(str)
    item_ids = catalog["book_id"].tolist()
    item_index = {item_id: index for index, item_id in enumerate(item_ids)}

    interactions = train_df.copy()
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["book_id"] = interactions["book_id"].astype(str)
    interactions = interactions[interactions["book_id"].isin(item_index)]
    user_ids = sorted(interactions["user_id"].unique().tolist())
    user_index = {user_id: index for index, user_id in enumerate(user_ids)}

    if not user_ids or not item_ids:
        return {"ready": False}

    row_indices = interactions["user_id"].map(user_index).to_numpy()
    col_indices = interactions["book_id"].map(item_index).to_numpy()
    values = (pd.to_numeric(interactions["rating"], errors="coerce").fillna(0).to_numpy() / 5.0).clip(0, 1)
    matrix = sparse.csr_matrix((values, (row_indices, col_indices)), shape=(len(user_ids), len(item_ids)))
    max_components = min(matrix.shape) - 1
    if matrix.nnz == 0 or max_components < 2:
        return {"ready": False}

    components = min(n_components, max_components)
    svd = TruncatedSVD(n_components=components, random_state=random_state)
    user_factors = svd.fit_transform(matrix)
    item_factors = svd.components_.T

    return {
        "ready": True,
        "user_index": user_index,
        "item_ids": item_ids,
        "user_factors": user_factors,
        "item_factors": item_factors,
        "explained_variance": float(svd.explained_variance_ratio_.sum()),
        "n_components": components,
    }


def _prune_sparse_topk(matrix: sparse.csr_matrix, top_k: int = 60) -> sparse.csr_matrix:
    matrix = matrix.tocsr()
    data = []
    indices = []
    indptr = [0]
    for row_index in range(matrix.shape[0]):
        start, end = matrix.indptr[row_index], matrix.indptr[row_index + 1]
        row_data = matrix.data[start:end]
        row_indices = matrix.indices[start:end]
        if len(row_data) > top_k:
            keep = np.argpartition(row_data, -top_k)[-top_k:]
            order = keep[np.argsort(row_data[keep])[::-1]]
            row_data = row_data[order]
            row_indices = row_indices[order]
        data.extend(row_data.tolist())
        indices.extend(row_indices.tolist())
        indptr.append(len(data))
    return sparse.csr_matrix((data, indices, indptr), shape=matrix.shape)


def _build_item_knn_model(train_df: pd.DataFrame, books_df: pd.DataFrame, n_neighbors: int = 60) -> dict:
    catalog = books_df.copy()
    catalog["book_id"] = catalog["book_id"].astype(str)
    item_ids = catalog["book_id"].tolist()
    item_index = {item_id: index for index, item_id in enumerate(item_ids)}

    interactions = train_df.copy()
    interactions["user_id"] = interactions["user_id"].astype(str)
    interactions["book_id"] = interactions["book_id"].astype(str)
    interactions = interactions[interactions["book_id"].isin(item_index)]
    user_ids = sorted(interactions["user_id"].unique().tolist())
    user_index = {user_id: index for index, user_id in enumerate(user_ids)}
    if not user_ids or not item_ids or interactions.empty:
        return {"ready": False, "item_ids": item_ids, "item_index": item_index}

    row_indices = interactions["user_id"].map(user_index).to_numpy()
    col_indices = interactions["book_id"].map(item_index).to_numpy()
    values = (pd.to_numeric(interactions["rating"], errors="coerce").fillna(0).to_numpy(dtype=float) / 5.0).clip(0, 1)
    user_item = sparse.csr_matrix((values, (row_indices, col_indices)), shape=(len(user_ids), len(item_ids)))
    item_user = user_item.T.tocsr()
    norms = np.sqrt(item_user.multiply(item_user).sum(axis=1)).A1
    norms[norms == 0] = 1.0
    normalized_item_user = item_user.multiply(1.0 / norms[:, None]).tocsr()
    similarity = (normalized_item_user @ normalized_item_user.T).tocsr()
    similarity.setdiag(0)
    similarity.eliminate_zeros()
    similarity = _prune_sparse_topk(similarity, top_k=n_neighbors)
    return {
        "ready": True,
        "item_ids": item_ids,
        "item_index": item_index,
        "similarity": similarity,
        "n_neighbors": n_neighbors,
    }


def _recommend_item_knn_ids(
    user_id: str,
    train_df: pd.DataFrame,
    books_df: pd.DataFrame,
    item_knn_model: dict,
    n: int,
    diversity_weight: float = 0.06,
) -> list[str]:
    user_id = str(user_id)
    if not item_knn_model.get("ready"):
        return _recommend_popular_ids(user_id, train_df, books_df, n)

    item_index = item_knn_model["item_index"]
    item_ids = item_knn_model["item_ids"]
    user_history = train_df[train_df["user_id"].astype(str) == user_id].copy()
    user_history["book_id"] = user_history["book_id"].astype(str)
    user_history = user_history[user_history["book_id"].isin(item_index)]
    if user_history.empty:
        return _recommend_popular_ids(user_id, train_df, books_df, n)

    cols = user_history["book_id"].map(item_index).to_numpy()
    values = (pd.to_numeric(user_history["rating"], errors="coerce").fillna(0).to_numpy(dtype=float) / 5.0).clip(0, 1)
    user_vector = sparse.csr_matrix((values, ([0] * len(cols), cols)), shape=(1, len(item_ids)))
    scores = (user_vector @ item_knn_model["similarity"]).toarray().ravel()
    rated = set(user_history["book_id"].astype(str))
    candidates = books_df.copy()
    candidates["book_id"] = candidates["book_id"].astype(str)
    candidates = candidates[~candidates["book_id"].isin(rated)].copy()
    if candidates.empty:
        return []

    score_lookup = dict(zip(item_ids, scores))
    candidates["_cf_score"] = candidates["book_id"].map(score_lookup).fillna(0.0)
    max_cf = float(candidates["_cf_score"].max())
    if max_cf > 0:
        candidates["_cf_score"] = candidates["_cf_score"] / max_cf
    candidates["_rating_norm"] = pd.to_numeric(candidates["rating"], errors="coerce").fillna(0) / 5.0
    candidates["_popularity"] = _popularity_norm(candidates)
    candidates["_score"] = (
        0.78 * candidates["_cf_score"]
        + 0.12 * candidates["_rating_norm"]
        + 0.10 * candidates["_popularity"]
    ).clip(0, 1)
    candidates["_reason"] = "Dựa trên các sách tương tự trong lịch sử đánh giá"
    ranked = _rank_scored_candidates(
        candidates.sort_values(["_score", "rating", "ratings_count"], ascending=False),
        n,
        diversity_weight=diversity_weight,
    )
    return [str(candidate["book_id"]) for candidate in ranked]


def _recommend_svd_ids(
    user_id: str,
    train_df: pd.DataFrame,
    books_df: pd.DataFrame,
    svd_model: dict,
    n: int,
    diversity_weight: float = 0.08,
) -> list[str]:
    user_id = str(user_id)
    if not svd_model.get("ready") or user_id not in svd_model.get("user_index", {}):
        return _recommend_profile_ids(user_id, train_df, books_df, n, diversity_weight=diversity_weight)

    user_vector = svd_model["user_factors"][svd_model["user_index"][user_id]]
    latent_scores = user_vector @ svd_model["item_factors"].T
    latent_min = float(np.min(latent_scores))
    latent_max = float(np.max(latent_scores))
    if latent_max > latent_min:
        latent_scores = (latent_scores - latent_min) / (latent_max - latent_min)
    else:
        latent_scores = np.full_like(latent_scores, 0.5, dtype=float)

    score_lookup = dict(zip(svd_model["item_ids"], latent_scores))
    rated = set(train_df[train_df["user_id"].astype(str) == user_id]["book_id"].astype(str))
    candidates = books_df.copy()
    candidates["book_id"] = candidates["book_id"].astype(str)
    candidates = candidates[~candidates["book_id"].isin(rated)].copy()
    if candidates.empty:
        return []

    candidates["_latent"] = candidates["book_id"].map(score_lookup).fillna(0.0)
    candidates["_rating_norm"] = pd.to_numeric(candidates["rating"], errors="coerce").fillna(0) / 5.0
    candidates["_popularity"] = _popularity_norm(candidates)
    candidates["_score"] = (
        0.78 * candidates["_latent"]
        + 0.14 * candidates["_rating_norm"]
        + 0.08 * candidates["_popularity"]
    ).clip(0, 1)
    candidates["_reason"] = "Dựa trên biểu diễn ẩn học từ ma trận đánh giá"
    ranked = _rank_scored_candidates(
        candidates.sort_values(["_score", "rating", "ratings_count"], ascending=False),
        n,
        diversity_weight=diversity_weight,
    )
    return [str(candidate["book_id"]) for candidate in ranked]


@st.cache_data(show_spinner=False)
def evaluate_working_recommenders(
    ratings_df: pd.DataFrame,
    books_df: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 4.0,
    max_users: int = 35,
) -> tuple[pd.DataFrame, dict]:
    train_df, test_df = _split_ratings_for_evaluation(ratings_df, test_size=0.2, seed=42)
    relevant_by_user = {
        str(user_id): set(group[group["rating"] >= relevance_threshold]["book_id"].astype(str))
        for user_id, group in test_df.groupby("user_id")
    }
    relevant_by_user = {user_id: items for user_id, items in relevant_by_user.items() if items}
    users = list(relevant_by_user.keys())[:max_users]

    book_lookup = books_df.set_index(books_df["book_id"].astype(str))
    catalog_ids = books_df["book_id"].astype(str).tolist()
    ratings_count_lookup = pd.to_numeric(book_lookup["ratings_count"], errors="coerce").fillna(0)
    long_tail_cutoff = float(ratings_count_lookup.quantile(0.80)) if len(ratings_count_lookup) else 0.0
    max_popularity = float(np.log1p(pd.to_numeric(books_df["ratings_count"], errors="coerce").fillna(0)).max())
    svd_model = _build_svd_model(train_df, books_df, n_components=24)
    item_knn_model = _build_item_knn_model(train_df, books_df, n_neighbors=60)
    methods = {
        "Mô hình phổ biến": lambda user_id: _recommend_popular_ids(user_id, train_df, books_df, k),
        "Item-KNN CF": lambda user_id: _recommend_item_knn_ids(user_id, train_df, books_df, item_knn_model, k),
        "SVD yếu tố ẩn": lambda user_id: _recommend_svd_ids(user_id, train_df, books_df, svd_model, k),
        "Hồ sơ lai": lambda user_id: _recommend_profile_ids(user_id, train_df, books_df, k, diversity_weight=0.0),
        "Lai + đa dạng": lambda user_id: _recommend_profile_ids(user_id, train_df, books_df, k, diversity_weight=0.16),
    }

    rows = []
    for method_name, recommender_fn in methods.items():
        metric_rows = []
        covered_items = set()
        recommendation_lists = []
        recommendation_counter: Counter = Counter()
        diversity_values = []
        novelty_values = []
        calibration_values = []
        long_tail_hits = 0
        total_recommendations = 0
        for user_id in users:
            rec_ids = recommender_fn(user_id)
            relevant = relevant_by_user[user_id]
            covered_items.update(rec_ids)
            recommendation_lists.append(rec_ids)
            recommendation_counter.update(rec_ids)
            total_recommendations += len(rec_ids)
            metric_rows.append(_ranking_metrics(rec_ids, relevant, k))
            genres = [book_lookup.loc[item_id, "genre"] for item_id in rec_ids if item_id in book_lookup.index]
            if genres:
                diversity_values.append(len(set(genres)) / len(genres))
            if max_popularity > 0:
                for item_id in rec_ids:
                    if item_id in book_lookup.index:
                        novelty_values.append(
                            1.0 - np.log1p(float(book_lookup.loc[item_id, "ratings_count"])) / max_popularity
                        )
            long_tail_hits += sum(
                1
                for item_id in rec_ids
                if item_id in book_lookup.index and float(ratings_count_lookup.loc[item_id]) <= long_tail_cutoff
            )

            history_dist = _genre_distribution(
                _user_history_item_ids(user_id, train_df, relevance_threshold),
                book_lookup,
            )
            rec_dist = _genre_distribution(rec_ids, book_lookup)
            if history_dist and rec_dist:
                calibration_values.append(_jensen_shannon_divergence(history_dist, rec_dist))

        if metric_rows:
            metric_df = pd.DataFrame(metric_rows)
            rows.append(
                {
                    "Mô hình": method_name,
                    "Precision@10": float(metric_df["Precision@K"].mean()),
                    "Recall@10": float(metric_df["Recall@K"].mean()),
                    "NDCG@10": float(metric_df["NDCG@K"].mean()),
                    "MAP@10": float(metric_df["MAP@K"].mean()),
                    "MRR@10": float(metric_df["MRR@K"].mean()),
                    "Hit Rate@10": float(metric_df["Hit Rate@K"].mean()),
                    "Coverage": len(covered_items) / max(1, books_df["book_id"].nunique()),
                    "Diversity": float(np.mean(diversity_values)) if diversity_values else 0.0,
                    "Novelty": float(np.mean(novelty_values)) if novelty_values else 0.0,
                    "Personalization": _inter_list_diversity(recommendation_lists),
                    "Long-tail Share": long_tail_hits / total_recommendations if total_recommendations else 0.0,
                    "Popularity Gini": _gini_coefficient(
                        [recommendation_counter.get(item_id, 0) for item_id in catalog_ids]
                    ),
                    "Calibration Error": float(np.mean(calibration_values)) if calibration_values else 0.0,
                }
            )

    summary = {
        "train_ratings": len(train_df),
        "test_ratings": len(test_df),
        "evaluated_users": len(users),
        "catalog_size": books_df["book_id"].nunique(),
        "k": k,
        "relevance_threshold": relevance_threshold,
        "svd_components": svd_model.get("n_components", 0),
        "svd_explained_variance": svd_model.get("explained_variance", 0.0),
        "item_knn_neighbors": item_knn_model.get("n_neighbors", 0),
    }
    return pd.DataFrame(rows), summary


@st.cache_data(show_spinner=False)
def evaluate_ablation_study(
    ratings_df: pd.DataFrame,
    books_df: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 4.0,
    max_users: int = 22,
) -> pd.DataFrame:
    train_df, test_df = _split_ratings_for_evaluation(ratings_df, test_size=0.2, seed=42)
    relevant_by_user = {
        str(user_id): set(group[group["rating"] >= relevance_threshold]["book_id"].astype(str))
        for user_id, group in test_df.groupby("user_id")
    }
    relevant_by_user = {user_id: items for user_id, items in relevant_by_user.items() if items}
    users = list(relevant_by_user.keys())[:max_users]
    if not users:
        return pd.DataFrame()

    def without_signal(signal_name: str) -> dict[str, float]:
        weights = DEFAULT_HYBRID_WEIGHTS.copy()
        weights[signal_name] = 0.0
        return weights

    variants = [
        ("Đầy đủ", "Không", DEFAULT_HYBRID_WEIGHTS.copy(), 0.16),
        ("Không dùng thể loại", "Thể loại", without_signal("genre"), 0.16),
        ("Không dùng tác giả", "Tác giả", without_signal("author"), 0.16),
        ("Không dùng độ phổ biến", "Độ phổ biến", without_signal("popularity"), 0.16),
        ("Không dùng độ mới lạ", "Độ mới lạ", without_signal("novelty"), 0.16),
        ("Không xếp hạng lại", "Đa dạng", DEFAULT_HYBRID_WEIGHTS.copy(), 0.0),
    ]

    book_lookup = books_df.set_index(books_df["book_id"].astype(str))
    catalog_ids = books_df["book_id"].astype(str).tolist()
    ratings_count_lookup = pd.to_numeric(book_lookup["ratings_count"], errors="coerce").fillna(0)
    long_tail_cutoff = float(ratings_count_lookup.quantile(0.80)) if len(ratings_count_lookup) else 0.0
    max_popularity = float(np.log1p(pd.to_numeric(books_df["ratings_count"], errors="coerce").fillna(0)).max())
    rows = []

    for variant_name, removed_signal, weights, diversity_weight in variants:
        metric_rows = []
        covered_items = set()
        recommendation_lists = []
        recommendation_counter: Counter = Counter()
        diversity_values = []
        novelty_values = []
        long_tail_hits = 0
        total_recommendations = 0
        calibration_values = []

        for user_id in users:
            rec_ids = _recommend_profile_ids(
                user_id,
                train_df,
                books_df,
                k,
                diversity_weight=diversity_weight,
                weights=weights,
            )
            covered_items.update(rec_ids)
            recommendation_lists.append(rec_ids)
            recommendation_counter.update(rec_ids)
            total_recommendations += len(rec_ids)
            metric_rows.append(_ranking_metrics(rec_ids, relevant_by_user[user_id], k))

            genres = [book_lookup.loc[item_id, "genre"] for item_id in rec_ids if item_id in book_lookup.index]
            if genres:
                diversity_values.append(len(set(genres)) / len(genres))
            if max_popularity > 0:
                for item_id in rec_ids:
                    if item_id in book_lookup.index:
                        novelty_values.append(
                            1.0 - np.log1p(float(book_lookup.loc[item_id, "ratings_count"])) / max_popularity
                        )
            long_tail_hits += sum(
                1
                for item_id in rec_ids
                if item_id in book_lookup.index and float(ratings_count_lookup.loc[item_id]) <= long_tail_cutoff
            )
            history_dist = _genre_distribution(
                _user_history_item_ids(user_id, train_df, relevance_threshold),
                book_lookup,
            )
            rec_dist = _genre_distribution(rec_ids, book_lookup)
            if history_dist and rec_dist:
                calibration_values.append(_jensen_shannon_divergence(history_dist, rec_dist))

        if metric_rows:
            metric_df = pd.DataFrame(metric_rows)
            rows.append(
                {
                    "Biến thể": variant_name,
                    "Tín hiệu bị bỏ": removed_signal,
                    "Precision@10": float(metric_df["Precision@K"].mean()),
                    "Recall@10": float(metric_df["Recall@K"].mean()),
                    "NDCG@10": float(metric_df["NDCG@K"].mean()),
                    "MAP@10": float(metric_df["MAP@K"].mean()),
                    "MRR@10": float(metric_df["MRR@K"].mean()),
                    "Coverage": len(covered_items) / max(1, books_df["book_id"].nunique()),
                    "Diversity": float(np.mean(diversity_values)) if diversity_values else 0.0,
                    "Novelty": float(np.mean(novelty_values)) if novelty_values else 0.0,
                    "Personalization": _inter_list_diversity(recommendation_lists),
                    "Long-tail Share": long_tail_hits / total_recommendations if total_recommendations else 0.0,
                    "Popularity Gini": _gini_coefficient(
                        [recommendation_counter.get(item_id, 0) for item_id in catalog_ids]
                    ),
                    "Calibration Error": float(np.mean(calibration_values)) if calibration_values else 0.0,
                }
            )

    ablation_df = pd.DataFrame(rows)
    if ablation_df.empty:
        return ablation_df
    full_rows = ablation_df[ablation_df["Biến thể"] == "Đầy đủ"]
    if not full_rows.empty:
        baseline_ndcg = float(full_rows.iloc[0]["NDCG@10"])
        baseline_diversity = float(full_rows.iloc[0]["Diversity"])
        baseline_coverage = float(full_rows.iloc[0]["Coverage"])
        ablation_df["Δ NDCG@10"] = ablation_df["NDCG@10"] - baseline_ndcg
        ablation_df["Δ Đa dạng"] = ablation_df["Diversity"] - baseline_diversity
        ablation_df["Δ Độ phủ"] = ablation_df["Coverage"] - baseline_coverage
    return ablation_df


@st.cache_data(show_spinner=False)
def evaluate_reranking_tradeoff(
    ratings_df: pd.DataFrame,
    books_df: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 4.0,
    max_users: int = 22,
) -> pd.DataFrame:
    train_df, test_df = _split_ratings_for_evaluation(ratings_df, test_size=0.2, seed=42)
    relevant_by_user = {
        str(user_id): set(group[group["rating"] >= relevance_threshold]["book_id"].astype(str))
        for user_id, group in test_df.groupby("user_id")
    }
    relevant_by_user = {user_id: items for user_id, items in relevant_by_user.items() if items}
    users = list(relevant_by_user.keys())[:max_users]
    if not users:
        return pd.DataFrame()

    book_lookup = books_df.set_index(books_df["book_id"].astype(str))
    catalog_ids = books_df["book_id"].astype(str).tolist()
    ratings_count_lookup = pd.to_numeric(book_lookup["ratings_count"], errors="coerce").fillna(0)
    long_tail_cutoff = float(ratings_count_lookup.quantile(0.80)) if len(ratings_count_lookup) else 0.0
    max_popularity = float(np.log1p(pd.to_numeric(books_df["ratings_count"], errors="coerce").fillna(0)).max())
    sweep = [
        ("Không xếp lại", 0.00),
        ("Nhẹ", 0.06),
        ("Cân bằng", 0.12),
        ("Đa dạng", 0.18),
        ("Khám phá", 0.24),
    ]

    rows = []
    for label, diversity_weight in sweep:
        metric_rows = []
        covered_items = set()
        recommendation_lists = []
        recommendation_counter: Counter = Counter()
        diversity_values = []
        novelty_values = []
        calibration_values = []
        long_tail_hits = 0
        total_recommendations = 0

        for user_id in users:
            rec_ids = _recommend_profile_ids(
                user_id,
                train_df,
                books_df,
                k,
                diversity_weight=diversity_weight,
            )
            covered_items.update(rec_ids)
            recommendation_lists.append(rec_ids)
            recommendation_counter.update(rec_ids)
            total_recommendations += len(rec_ids)
            metric_rows.append(_ranking_metrics(rec_ids, relevant_by_user[user_id], k))

            genres = [book_lookup.loc[item_id, "genre"] for item_id in rec_ids if item_id in book_lookup.index]
            if genres:
                diversity_values.append(len(set(genres)) / len(genres))
            if max_popularity > 0:
                for item_id in rec_ids:
                    if item_id in book_lookup.index:
                        novelty_values.append(
                            1.0 - np.log1p(float(book_lookup.loc[item_id, "ratings_count"])) / max_popularity
                        )
            long_tail_hits += sum(
                1
                for item_id in rec_ids
                if item_id in book_lookup.index and float(ratings_count_lookup.loc[item_id]) <= long_tail_cutoff
            )

            history_dist = _genre_distribution(
                _user_history_item_ids(user_id, train_df, relevance_threshold),
                book_lookup,
            )
            rec_dist = _genre_distribution(rec_ids, book_lookup)
            if history_dist and rec_dist:
                calibration_values.append(_jensen_shannon_divergence(history_dist, rec_dist))

        metric_df = pd.DataFrame(metric_rows)
        ndcg = float(metric_df["NDCG@K"].mean())
        diversity = float(np.mean(diversity_values)) if diversity_values else 0.0
        coverage = len(covered_items) / max(1, books_df["book_id"].nunique())
        novelty = float(np.mean(novelty_values)) if novelty_values else 0.0
        personalization = _inter_list_diversity(recommendation_lists)
        long_tail_share = long_tail_hits / total_recommendations if total_recommendations else 0.0
        calibration_error = float(np.mean(calibration_values)) if calibration_values else 0.0
        popularity_gini = _gini_coefficient([recommendation_counter.get(item_id, 0) for item_id in catalog_ids])
        rows.append(
            {
                "Reranking": label,
                "Diversity weight": diversity_weight,
                "Precision@10": float(metric_df["Precision@K"].mean()),
                "Recall@10": float(metric_df["Recall@K"].mean()),
                "NDCG@10": ndcg,
                "MAP@10": float(metric_df["MAP@K"].mean()),
                "MRR@10": float(metric_df["MRR@K"].mean()),
                "Hit Rate@10": float(metric_df["Hit Rate@K"].mean()),
                "Coverage": coverage,
                "Diversity": diversity,
                "Novelty": novelty,
                "Personalization": personalization,
                "Long-tail Share": long_tail_share,
                "Popularity Gini": popularity_gini,
                "Calibration Error": calibration_error,
                "Balanced Score": (
                    0.46 * ndcg
                    + 0.16 * diversity
                    + 0.12 * coverage
                    + 0.10 * personalization
                    + 0.08 * long_tail_share
                    + 0.05 * novelty
                    - 0.03 * calibration_error
                ),
            }
        )

    return pd.DataFrame(rows)


@st.cache_data(show_spinner=False)
def evaluate_user_segment_performance(
    ratings_df: pd.DataFrame,
    books_df: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 4.0,
    max_users: int = 30,
) -> pd.DataFrame:
    train_df, test_df = _split_ratings_for_evaluation(ratings_df, test_size=0.2, seed=42)
    relevant_by_user = {
        str(user_id): set(group[group["rating"] >= relevance_threshold]["book_id"].astype(str))
        for user_id, group in test_df.groupby("user_id")
    }
    relevant_by_user = {user_id: items for user_id, items in relevant_by_user.items() if items}
    users = list(relevant_by_user.keys())[:max_users]
    if not users:
        return pd.DataFrame()

    history_counts = train_df.groupby(train_df["user_id"].astype(str)).size()
    sorted_users = sorted(users, key=lambda user_id: (history_counts.get(str(user_id), 0), str(user_id)))
    segment_lookup = {}
    for index, user_id in enumerate(sorted_users):
        percentile = index / max(1, len(sorted_users) - 1)
        if percentile < 1 / 3:
            segment_lookup[user_id] = "Ít lịch sử"
        elif percentile < 2 / 3:
            segment_lookup[user_id] = "Trung bình"
        else:
            segment_lookup[user_id] = "Nhiều lịch sử"

    def segment_for_user(user_id: str) -> str:
        return segment_lookup.get(str(user_id), "Trung bình")

    svd_model = _build_svd_model(train_df, books_df, n_components=24)
    item_knn_model = _build_item_knn_model(train_df, books_df, n_neighbors=60)
    methods = {
        "Item-KNN CF": lambda user_id: _recommend_item_knn_ids(user_id, train_df, books_df, item_knn_model, k),
        "SVD yếu tố ẩn": lambda user_id: _recommend_svd_ids(user_id, train_df, books_df, svd_model, k),
        "Hồ sơ lai": lambda user_id: _recommend_profile_ids(user_id, train_df, books_df, k, diversity_weight=0.0),
        "Lai + đa dạng": lambda user_id: _recommend_profile_ids(user_id, train_df, books_df, k, diversity_weight=0.16),
    }
    book_lookup = books_df.set_index(books_df["book_id"].astype(str))
    grouped_rows: dict[tuple[str, str], list[dict]] = defaultdict(list)

    for method_name, recommender_fn in methods.items():
        for user_id in users:
            rec_ids = recommender_fn(user_id)
            metrics = _ranking_metrics(rec_ids, relevant_by_user[user_id], k)
            history_dist = _genre_distribution(
                _user_history_item_ids(user_id, train_df, relevance_threshold),
                book_lookup,
            )
            rec_dist = _genre_distribution(rec_ids, book_lookup)
            metrics["Calibration Error"] = (
                _jensen_shannon_divergence(history_dist, rec_dist)
                if history_dist and rec_dist
                else 0.0
            )
            grouped_rows[(method_name, segment_for_user(user_id))].append(metrics)

    rows = []
    segment_order = {"Ít lịch sử": 0, "Trung bình": 1, "Nhiều lịch sử": 2}
    for (method_name, segment), metric_rows in grouped_rows.items():
        metric_df = pd.DataFrame(metric_rows)
        rows.append(
            {
                "Mô hình": method_name,
                "Nhóm người dùng": segment,
                "Số người dùng": len(metric_rows),
                "Precision@10": float(metric_df["Precision@K"].mean()),
                "Recall@10": float(metric_df["Recall@K"].mean()),
                "NDCG@10": float(metric_df["NDCG@K"].mean()),
                "MAP@10": float(metric_df["MAP@K"].mean()),
                "MRR@10": float(metric_df["MRR@K"].mean()),
                "Calibration Error": float(metric_df["Calibration Error"].mean()),
                "_segment_order": segment_order.get(segment, 99),
            }
        )

    return pd.DataFrame(rows).sort_values(["_segment_order", "Mô hình"]).drop(columns=["_segment_order"])


@st.cache_data(show_spinner=False)
def evaluate_svd_factor_sweep(
    ratings_df: pd.DataFrame,
    books_df: pd.DataFrame,
    k: int = 10,
    relevance_threshold: float = 4.0,
    max_users: int = 15,
) -> pd.DataFrame:
    train_df, test_df = _split_ratings_for_evaluation(ratings_df, test_size=0.2, seed=42)
    relevant_by_user = {
        str(user_id): set(group[group["rating"] >= relevance_threshold]["book_id"].astype(str))
        for user_id, group in test_df.groupby("user_id")
    }
    relevant_by_user = {user_id: items for user_id, items in relevant_by_user.items() if items}
    users = list(relevant_by_user.keys())[:max_users]
    if not users:
        return pd.DataFrame()

    book_lookup = books_df.set_index(books_df["book_id"].astype(str))
    requested_components = [4, 8, 12, 16, 24, 32]
    seen_components = set()
    rows = []
    for requested in requested_components:
        svd_model = _build_svd_model(train_df, books_df, n_components=requested)
        actual_components = int(svd_model.get("n_components", 0))
        if not svd_model.get("ready") or actual_components in seen_components:
            continue
        seen_components.add(actual_components)

        metric_rows = []
        covered_items = set()
        recommendation_lists = []
        calibration_values = []
        for user_id in users:
            rec_ids = _recommend_svd_ids(user_id, train_df, books_df, svd_model, k)
            covered_items.update(rec_ids)
            recommendation_lists.append(rec_ids)
            metric_rows.append(_ranking_metrics(rec_ids, relevant_by_user[user_id], k))
            history_dist = _genre_distribution(
                _user_history_item_ids(user_id, train_df, relevance_threshold),
                book_lookup,
            )
            rec_dist = _genre_distribution(rec_ids, book_lookup)
            if history_dist and rec_dist:
                calibration_values.append(_jensen_shannon_divergence(history_dist, rec_dist))

        metric_df = pd.DataFrame(metric_rows)
        rows.append(
            {
                "Factors": actual_components,
                "Explained variance": svd_model.get("explained_variance", 0.0),
                "Precision@10": float(metric_df["Precision@K"].mean()),
                "Recall@10": float(metric_df["Recall@K"].mean()),
                "NDCG@10": float(metric_df["NDCG@K"].mean()),
                "MAP@10": float(metric_df["MAP@K"].mean()),
                "MRR@10": float(metric_df["MRR@K"].mean()),
                "Coverage": len(covered_items) / max(1, books_df["book_id"].nunique()),
                "Personalization": _inter_list_diversity(recommendation_lists),
                "Calibration Error": float(np.mean(calibration_values)) if calibration_values else 0.0,
            }
        )

    return pd.DataFrame(rows)


def dataframe_to_markdown_table(df: pd.DataFrame, max_rows: int = 12) -> str:
    if df is None or df.empty:
        return "_Không có dữ liệu._"

    view = df.head(max_rows).copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda value: f"{value:.4f}")
    headers = [str(col) for col in view.columns]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in view.iterrows():
        values = [str(row[col]).replace("|", "/") for col in view.columns]
        lines.append("| " + " | ".join(values) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Hiển thị {max_rows}/{len(df)} dòng._")
    return "\n".join(lines)


def build_evaluation_report(
    eval_df: pd.DataFrame,
    eval_summary: dict,
    ablation_df: pd.DataFrame,
    rerank_df: pd.DataFrame,
    segment_df: pd.DataFrame,
    svd_sweep_df: pd.DataFrame,
) -> str:
    best_model = eval_df.sort_values("NDCG@10", ascending=False).iloc[0] if not eval_df.empty else None
    best_rerank = rerank_df.sort_values("Balanced Score", ascending=False).iloc[0] if not rerank_df.empty else None
    strongest_drop = None
    if ablation_df is not None and not ablation_df.empty and "Δ NDCG@10" in ablation_df.columns:
        ablation_without_full = ablation_df[ablation_df["Biến thể"] != "Đầy đủ"].copy()
        if not ablation_without_full.empty:
            strongest_drop = ablation_without_full.sort_values("Δ NDCG@10").iloc[0]

    conclusion_lines = []
    if best_model is not None:
        conclusion_lines.append(
            f"- Mô hình dẫn đầu theo NDCG@10: **{best_model['Mô hình']}** ({best_model['NDCG@10']:.4f})."
        )
    if best_rerank is not None:
        conclusion_lines.append(
            f"- Mức xếp hạng lại cân bằng nhất: **{best_rerank['Reranking']}** "
            f"(điểm cân bằng {best_rerank['Balanced Score']:.4f})."
        )
    if strongest_drop is not None:
        conclusion_lines.append(
            f"- Ablation nhạy nhất: **{strongest_drop['Biến thể']}** "
            f"(Δ NDCG@10 {strongest_drop['Δ NDCG@10']:.4f})."
        )
    if not conclusion_lines:
        conclusion_lines.append("- Chưa đủ dữ liệu để sinh kết luận tự động.")

    return f"""# Báo cáo đánh giá hệ gợi ý sách BRS

## Cấu hình thí nghiệm

- Người dùng đánh giá: {eval_summary.get('evaluated_users', 0)}
- Lượt đánh giá huấn luyện: {format_number(eval_summary.get('train_ratings', 0))}
- Lượt đánh giá kiểm tra: {format_number(eval_summary.get('test_ratings', 0))}
- K: {eval_summary.get('k', 10)}
- Ngưỡng phù hợp: {eval_summary.get('relevance_threshold', 4.0):.1f}/5
- SVD: {eval_summary.get('svd_components', 0)} yếu tố ẩn, phương sai giải thích {eval_summary.get('svd_explained_variance', 0.0):.2%}
- Item-KNN: top-{eval_summary.get('item_knn_neighbors', 0)} láng giềng item-item

## Kết luận nhanh

{chr(10).join(conclusion_lines)}

## So sánh mô hình

{dataframe_to_markdown_table(eval_df)}

## Ablation Study

{dataframe_to_markdown_table(ablation_df)}

## Xếp hạng lại theo độ đa dạng

{dataframe_to_markdown_table(rerank_df)}

## Độ ổn định theo nhóm người dùng

{dataframe_to_markdown_table(segment_df)}

## Tối ưu số yếu tố ẩn SVD

{dataframe_to_markdown_table(svd_sweep_df)}
"""


def build_metrics_csv_export(tables: list[tuple[str, pd.DataFrame]]) -> str:
    sections = []
    for title, df in tables:
        sections.append(title)
        if df is None or df.empty:
            sections.append("Không có dữ liệu")
        else:
            sections.append(df.to_csv(index=False))
        sections.append("")
    return "\n".join(sections)


def render_executive_summary(
    eval_df: pd.DataFrame,
    eval_summary: dict,
    ablation_df: pd.DataFrame,
    rerank_df: pd.DataFrame,
    books_df: pd.DataFrame,
    ratings_df: pd.DataFrame,
    data_audit: dict,
) -> None:
    if eval_df.empty:
        return

    best_ndcg = eval_df.sort_values("NDCG@10", ascending=False).iloc[0]
    best_diversity = eval_df.sort_values("Diversity", ascending=False).iloc[0]
    best_rerank = rerank_df.sort_values("Balanced Score", ascending=False).iloc[0] if not rerank_df.empty else None
    strongest_ablation = None
    if not ablation_df.empty and "Δ NDCG@10" in ablation_df.columns:
        ablation_without_full = ablation_df[ablation_df["Biến thể"] != "Đầy đủ"]
        if not ablation_without_full.empty:
            strongest_ablation = ablation_without_full.sort_values("Δ NDCG@10").iloc[0]

    st.markdown("### Tóm tắt điều hành")
    summary_cols = st.columns(4)
    with summary_cols[0]:
        display_metric_card(f"{best_ndcg['NDCG@10']:.3f}", f"Dẫn đầu: {best_ndcg['Mô hình']}")
    with summary_cols[1]:
        display_metric_card(f"{best_diversity['Diversity']:.1%}", f"Đa dạng: {best_diversity['Mô hình']}")
    with summary_cols[2]:
        if strongest_ablation is not None:
            display_metric_card(f"{strongest_ablation['Δ NDCG@10']:.3f}", strongest_ablation["Biến thể"])
        else:
            display_metric_card("N/A", "Ablation")
    with summary_cols[3]:
        display_metric_card(compact_data_source_label(data_audit), "Nguồn dữ liệu")

    rerank_sentence = (
        f"Mức xếp hạng lại cân bằng nhất là {best_rerank['Reranking']} "
        f"(điểm cân bằng {best_rerank['Balanced Score']:.3f})."
        if best_rerank is not None
        else "Chưa đủ dữ liệu để kết luận mức xếp hạng lại."
    )
    ablation_sentence = (
        f"Ablation nhạy nhất là {strongest_ablation['Biến thể']}, cho thấy tín hiệu này có ảnh hưởng rõ đến NDCG@10."
        if strongest_ablation is not None
        else "Ablation chưa đủ dữ liệu để kết luận tín hiệu nhạy nhất."
    )
    st.markdown(
        f"""
        <div class="info-box">
            <strong>Kết luận tự động:</strong> trên {format_number(len(books_df))} sách,
            {format_number(ratings_df['user_id'].nunique())} người dùng và
            {format_number(len(ratings_df))} lượt đánh giá, mô hình
            <strong>{best_ndcg['Mô hình']}</strong> đang có NDCG@10 tốt nhất.
            {rerank_sentence} {ablation_sentence}
            <br>
            <span style="color: {COLORS['text_light']};">
                So sánh hiện gồm Popular baseline, Item-KNN CF, SVD yếu tố ẩn, hồ sơ lai và lai + đa dạng.
                Item-KNN dùng top-{eval_summary.get('item_knn_neighbors', 0)} láng giềng item-item.
            </span>
        </div>
        """,
        unsafe_allow_html=True,
    )


def get_similar_books(book_id: int, books_df: pd.DataFrame, n: int = 10) -> List[BookRecommendation]:
    book_id = str(book_id)
    target_matches = books_df[books_df["book_id"].astype(str) == book_id]
    if target_matches.empty:
        return []

    target_book = target_matches.iloc[0]
    target_genre = target_book["genre"]
    target_author = target_book["author"]

    recommendations = []

    for _, book in books_df.iterrows():
        if str(book["book_id"]) == book_id:
            continue

        score = 0.0
        reason = ""

        if book["genre"] == target_genre:
            score += 0.6
            reason = f"Same genre: {target_genre}"

        if book["author"] == target_author:
            score += 0.3
            reason = f"Same author: {target_author}"

        rating_diff = abs(book["rating"] - target_book["rating"])
        if rating_diff < 0.3:
            score += 0.1

        if score > 0:
            score += max(0.0, 0.18 - rating_diff * 0.2)
            recommendations.append(BookRecommendation(
                title=book["title"],
                author=book["author"],
                genre=book["genre"],
                image_url=book.get("image_url", ""),
                score=round(min(0.99, score), 3),
                rating=book["rating"],
                ratings_count=book["ratings_count"],
                year=book["year"],
                bestseller=book["bestseller"],
                reason=reason,
                book_id=str(book["book_id"]),
            ))

    recommendations.sort(key=lambda x: x.score, reverse=True)
    return recommendations[:n]


def get_recommendations_by_mood(
    mood: str,
    books_df: pd.DataFrame,
    n: int = 10,
    exploration_level: float = 45.0,
) -> List[BookRecommendation]:
    target_genres = set(READING_MOODS.get(mood, []))
    max_popularity = float(np.log1p(pd.to_numeric(books_df["ratings_count"], errors="coerce").fillna(0)).max())

    recommendations = []
    for idx, book in books_df.iterrows():
        if _book_matches_mood(book["genre"], target_genres):
            popularity = np.log1p(float(book["ratings_count"])) / max_popularity if max_popularity > 0 else 0.0
            score = 0.58 * (float(book["rating"]) / 5.0) + 0.28 * popularity + 0.14 * (1.0 - popularity)
            recommendations.append(BookRecommendation(
                title=book["title"],
                author=book["author"],
                genre=book["genre"],
                image_url=book.get("image_url", ""),
                score=round(score, 3),
                rating=book["rating"],
                ratings_count=book["ratings_count"],
                year=book["year"],
                bestseller=book["bestseller"],
                reason=f"Phù hợp với tâm trạng {translate_mood(mood)}",
                book_id=str(book["book_id"]),
            ))

    recommendations.sort(key=lambda x: (x.score, x.rating), reverse=True)
    return _rank_recommendations_for_diversity(
        recommendations,
        n,
        exploration_to_diversity_weight(exploration_level, base=0.04, max_add=0.14),
    )


def get_user_recommendations(user_id: str, ratings_df: pd.DataFrame,
                            books_df: pd.DataFrame, n: int = 10,
                            exploration_level: float = 45.0,
                            feedback_votes: dict | None = None,
                            saved_items: list | None = None) -> List[BookRecommendation]:
    scored = _score_user_candidates(
        user_id,
        ratings_df,
        books_df,
        exclude_rated=True,
        feedback_votes=feedback_votes,
        saved_items=saved_items,
    )
    ranked = _rank_scored_candidates(
        scored,
        n,
        diversity_weight=exploration_to_diversity_weight(exploration_level),
    )
    return [_candidate_to_recommendation(candidate) for candidate in ranked]


if __name__ == "__main__":
    main()
