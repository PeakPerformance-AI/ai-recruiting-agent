import streamlit as st
import anthropic
import json
import requests
import pandas as pd
import io
import csv
import time
from typing import Optional

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Recruiting Agent",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #f8f9fb; }
    .stApp { background-color: #f8f9fb; }
    .candidate-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 20px 24px;
        margin-bottom: 16px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }
    .rank-badge {
        display: inline-block;
        background: linear-gradient(135deg, #3b82f6, #6366f1);
        color: white;
        font-weight: 700;
        font-size: 13px;
        padding: 4px 12px;
        border-radius: 20px;
        margin-right: 10px;
    }
    .score-pill {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-weight: 600;
        font-size: 13px;
    }
    .score-high   { background: #dcfce7; color: #16a34a; }
    .score-medium { background: #fef9c3; color: #ca8a04; }
    .score-low    { background: #fee2e2; color: #dc2626; }
    .section-label {
        font-size: 11px;
        font-weight: 700;
        letter-spacing: 1.2px;
        text-transform: uppercase;
        color: #94a3b8;
        margin-bottom: 4px;
    }
    .strength-item { color: #16a34a; font-size: 13px; margin: 2px 0; }
    .flag-item     { color: #dc2626; font-size: 13px; margin: 2px 0; }
    .outreach-box {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 12px 16px;
        font-size: 13px;
        color: #64748b;
        font-style: italic;
        white-space: pre-wrap;
    }
    .divider { border-top: 1px solid #e2e8f0; margin: 12px 0; }
    .stTextArea textarea { background: #ffffff !important; color: #000000 !important; border: 1px solid #e2e8f0 !important; }
    .stTextInput input  { background: #ffffff !important; color: #000000 !important; border: 1px solid #e2e8f0 !important; }
    textarea, input { color: #000000 !important; }
</style>
""", unsafe_allow_html=True)

# ── API Keys from Streamlit secrets ──────────────────────────────────────────
anthropic_key  = st.secrets["ANTHROPIC_KEY"]
brightdata_key = st.secrets.get("BRIGHTDATA_KEY", "")

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")
    st.markdown("### Input Method")
    data_source = st.radio(
        "How will you provide candidates?",
        ["Paste profile text", "LinkedIn URLs", "Upload File (CSV / XLSX / PDF)"],
        index=0,
    )
    st.markdown("---")
    st.markdown("### Scoring Weights")
    w_skills   = st.slider("Skills match",      0, 100, 35)
    w_exp      = st.slider("Experience level",  0, 100, 30)
    w_industry = st.slider("Industry fit",      0, 100, 20)
    w_growth   = st.slider("Career trajectory", 0, 100, 15)
    total_w    = w_skills + w_exp + w_industry + w_growth
    if total_w != 100:
        st.warning(f"Weights sum to {total_w} (should be 100)")
    st.markdown("---")
    st.caption("Built with Claude · Phase 2")

# ── Helpers ───────────────────────────────────────────────────────────────────
def fetch_brightdata(linkedin_url: str) -> Optional[str]:
    """Fetch a LinkedIn profile via Bright Data and return formatted text."""
    try:
        headers = {
            "Authorization": f"Bearer {brightdata_key}",
            "Content-Type": "application/json",
        }

        # Trigger the scrape
        r = requests.post(
            "https://api.brightdata.com/datasets/v3/trigger",
            params={
                "dataset_id": "gd_l1viktl72bvl7bjuj0",
                "include_errors": "true",
            },
            headers=headers,
            json=[{"url": linkedin_url}],
            timeout=60,
        )

        if r.status_code != 200:
            st.warning(f"Bright Data trigger failed ({r.status_code}): {r.text[:300]}")
            return None

        try:
            resp_json = r.json()
        except Exception:
            st.warning(f"Bright Data trigger returned non-JSON: {r.text[:300]}")
            return None

        snapshot_id = resp_json.get("snapshot_id")
        if not snapshot_id:
            st.warning(f"No snapshot_id returned. Response: {r.text[:300]}")
            return None

        # Poll for results — check immediately, then wait between retries
        for attempt in range(24):
            if attempt > 0:
                time.sleep(10)

            try:
                poll = requests.get(
                    f"https://api.brightdata.com/datasets/v3/snapshot/{snapshot_id}",
                    params={"format": "json"},
                    headers=headers,
                    timeout=30,
                )
            except requests.RequestException as e:
                st.warning(f"Poll request failed (attempt {attempt + 1}): {e}")
                continue

            if poll.status_code == 200:
                try:
                    data = poll.json()
                except Exception:
                    st.warning(f"Poll returned non-JSON (attempt {attempt + 1}): {poll.text[:200]}")
                    continue

                if isinstance(data, list) and len(data) > 0:
                    with st.expander(f"Raw BrightData response for {linkedin_url}"):
                        st.json(data[0])
                    return format_brightdata_profile(data[0])
                elif isinstance(data, dict):
                    status = data.get("status", "")
                    if status in ("running", "pending", "initializing"):
                        continue  # Still processing — keep polling
                    else:
                        st.warning(f"Unexpected response from Bright Data: {str(data)[:300]}")
                        return None

            elif poll.status_code == 202:
                # Still processing
                continue

            elif poll.status_code == 401:
                st.error("Bright Data API key is invalid or expired.")
                return None

            elif poll.status_code == 404:
                st.warning(f"Snapshot {snapshot_id} not found. The scrape may have expired.")
                return None

            elif poll.status_code == 429:
                st.warning("Bright Data rate limit hit — waiting 30s before retry.")
                time.sleep(30)
                continue

            else:
                st.warning(f"Unexpected poll status {poll.status_code}: {poll.text[:200]}")

        st.warning(f"Timed out waiting for Bright Data to return profile for: {linkedin_url}")
        return None

    except requests.RequestException as e:
        st.warning(f"Network error fetching {linkedin_url}: {e}")
        return None
    except Exception as e:
        st.warning(f"Unexpected error fetching {linkedin_url}: {e}")
        return None


def format_brightdata_profile(d: dict) -> str:
    """Format a Bright Data profile into text for Claude."""
    current_company = (
        d.get("current_company_name")
        or (d.get("current_company") or {}).get("name", "")
    )

    parts = [
        f"Name: {d.get('name', d.get('full_name', 'Unknown'))}",
        f"Headline: {d.get('headline', d.get('position', ''))}",
        f"Location: {d.get('location', d.get('city', ''))}",
        f"Current Company: {current_company}",
        f"Summary: {d.get('summary', d.get('about', ''))}",
        "\nExperience:",
    ]

    experiences = d.get("experience") or d.get("experiences") or []
    if experiences:
        for e in experiences:
            title   = e.get("title", e.get("position", ""))
            company = e.get("company", e.get("company_name", ""))
            start   = e.get("start_date", "?")
            end     = e.get("end_date", "present") or "present"
            parts.append(f"  - {title} at {company} ({start}–{end})")
            if e.get("description"):
                parts.append(f"    {e['description'][:300]}")
    else:
        parts.append(f"  (Work history not publicly available. Current employer: {current_company})")

    parts.append("\nEducation:")
    edu_details = d.get("educations_details", "")
    for ed in (d.get("education") or []):
        degree = ed.get("degree", ed.get("degree_name", ""))
        school = ed.get("school", ed.get("institution", edu_details))
        parts.append(f"  - {degree} at {school}")

    skills = d.get("skills") or []
    if skills and isinstance(skills[0], dict):
        skill_names = ", ".join(s.get("name", "") for s in skills)
    else:
        skill_names = ", ".join(str(s) for s in skills)
    parts.append(f"\nSkills: {skill_names}")

    return "\n".join(parts)


def extract_urls_from_csv(uploaded_file) -> list:
    """Extract LinkedIn URLs from a LinkedIn Recruiter CSV or XLSX export."""
    try:
        name = uploaded_file.name.lower()
        if name.endswith(".xlsx") or name.endswith(".xls"):
            df = pd.read_excel(uploaded_file)
        else:
            df = pd.read_csv(uploaded_file)

        url_col = None
        for col in df.columns:
            if "linkedin" in col.lower() and ("url" in col.lower() or "profile" in col.lower()):
                url_col = col
                break
        if not url_col:
            for col in df.columns:
                sample = df[col].dropna().astype(str)
                if sample.str.contains("linkedin.com/in/").any():
                    url_col = col
                    break
        if not url_col:
            return []
        urls = df[url_col].dropna().astype(str).tolist()
        return [u.strip() for u in urls if "linkedin.com/in/" in u]
    except Exception as e:
        st.warning(f"Could not read file: {e}")
        return []


def extract_text_from_pdf(uploaded_file) -> list:
    """Extract profile text from a PDF containing one or more LinkedIn profiles.
    Returns a list of text blocks, one per detected profile (or page)."""
    import pdfplumber, re
    try:
        texts = []
        with pdfplumber.open(uploaded_file) as pdf:
            full_text = "\n".join(
                page.extract_text() or "" for page in pdf.pages
            )
        # Try to split on LinkedIn profile boundaries (name + location headers)
        # If we can't split, treat the whole doc as one profile
        sections = re.split(r'\n(?=\S.{2,60}\n[A-Z][a-z])', full_text)
        for section in sections:
            section = section.strip()
            if len(section) > 100:  # skip tiny fragments
                texts.append(section)
        return texts if texts else [full_text]
    except Exception as e:
        st.warning(f"Could not read PDF: {e}")
        return []


def build_system_prompt(weights: dict) -> str:
    return f"""You are an expert technical recruiter with 15 years of experience.
Your job is to analyze LinkedIn candidate profiles against a job description
and produce structured, objective assessments.

Scoring weights:
- Skills match:        {weights['skills']}%
- Experience level:    {weights['experience']}%
- Industry fit:        {weights['industry']}%
- Career trajectory:   {weights['growth']}%

Respond with valid JSON only — no prose, no markdown fences.

Schema:
{{
  "candidates": [
    {{
      "name": "string",
      "current_title": "string",
      "current_company": "string",
      "overall_score": 0-100,
      "dimension_scores": {{
        "skills_match": 0-100,
        "experience_level": 0-100,
        "industry_fit": 0-100,
        "career_trajectory": 0-100
      }},
      "top_strengths": ["string", "string", "string"],
      "red_flags": ["string"],
      "summary": "2-3 sentence narrative on fit",
      "outreach_message": "Personalized 3-sentence LinkedIn message"
    }}
  ]
}}

Be honest. Not every candidate is a strong fit. Red flags are important.
"""


def score_candidates(job_desc: str, profiles: list, weights: dict) -> dict:
    """Call Claude to score all candidates."""
    client = anthropic.Anthropic(api_key=anthropic_key)

    profiles_block = ""
    for i, p in enumerate(profiles, 1):
        profiles_block += f"\n\n--- CANDIDATE {i} ---\n{p['text']}"

    user_msg = (
        f"JOB DESCRIPTION:\n{job_desc}\n\n"
        f"CANDIDATE PROFILES:{profiles_block}\n\n"
        "Analyze every candidate and return the JSON scorecard."
    )

    response = client.messages.create(
        model="claude-opus-4-5",
        max_tokens=4096,
        system=build_system_prompt(weights),
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = response.content[0].text.strip()
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]

    return json.loads(raw)


def score_color(score: int) -> str:
    if score >= 75:
        return "score-high"
    if score >= 50:
        return "score-medium"
    return "score-low"


# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown("# 🎯 AI Recruiting Agent")
st.markdown("Rank and score candidates against any job description in seconds.")
st.markdown("---")

col_jd, col_profiles = st.columns([1, 1], gap="large")

with col_jd:
    st.markdown("### 📋 Job Description")
    jd_file = st.file_uploader(
        "Upload job description (optional)",
        type=["pdf", "txt", "docx"],
        key="jd_upload",
    )
    jd_prefill = ""
    if jd_file:
        fname = jd_file.name.lower()
        try:
            if fname.endswith(".pdf"):
                import pdfplumber, io as _io
                with pdfplumber.open(_io.BytesIO(jd_file.read())) as pdf:
                    jd_prefill = "\n".join(p.extract_text() or "" for p in pdf.pages)
            elif fname.endswith(".docx"):
                import docx as _docx, io as _io
                doc = _docx.Document(_io.BytesIO(jd_file.read()))
                jd_prefill = "\n".join(para.text for para in doc.paragraphs)
            else:
                jd_prefill = jd_file.read().decode("utf-8", errors="ignore")
        except Exception as e:
            st.warning(f"Could not read file: {e}")

    job_description = st.text_area(
        "Job description",
        value=jd_prefill,
        height=320,
        placeholder="Senior Backend Engineer\n\nRequirements:\n- 5+ years Python\n- Distributed systems experience\n...",
        label_visibility="collapsed",
    )

with col_profiles:
    st.markdown("### 👤 Candidates")
    profiles_input = []

    # Mode 1: Paste text
    if data_source == "Paste profile text":
        st.caption("Paste LinkedIn profile text for each candidate.")
        if "num_profiles" not in st.session_state:
            st.session_state.num_profiles = 3
        for i in range(st.session_state.num_profiles):
            text = st.text_area(
                f"Candidate {i+1}",
                height=100,
                placeholder="Name, headline, experience, skills…",
                key=f"profile_{i}",
            )
            profiles_input.append({"text": text, "label": f"Candidate {i+1}"})
        c1, c2 = st.columns(2)
        with c1:
            if st.button("+ Add candidate"):
                st.session_state.num_profiles += 1
                st.rerun()
        with c2:
            if st.session_state.num_profiles > 1 and st.button("- Remove last"):
                st.session_state.num_profiles -= 1
                st.rerun()

    # Mode 2: LinkedIn URLs
    elif data_source == "LinkedIn URLs":
        st.caption("Paste one LinkedIn profile URL per line. Bright Data fetches each profile automatically.")
        urls_text = st.text_area(
            "LinkedIn URLs",
            height=280,
            placeholder="https://www.linkedin.com/in/jane-doe\nhttps://www.linkedin.com/in/john-smith",
            label_visibility="collapsed",
        )
        if urls_text.strip():
            profiles_input = [
                {"url": u.strip(), "label": u.strip()}
                for u in urls_text.splitlines()
                if u.strip()
            ]
            st.caption(f"{len(profiles_input)} URL(s) ready")

    # Mode 3: File Upload (CSV, XLSX, PDF, ZIP)
    else:
        st.caption("Upload LinkedIn Recruiter exports (CSV/XLSX), profile PDFs, or a ZIP of any of the above. Select multiple files at once.")
        uploaded_files = st.file_uploader(
            "Upload files",
            type=["csv", "xlsx", "xls", "pdf", "zip"],
            accept_multiple_files=True,
        )
        if uploaded_files:
            # Expand any ZIP files into their contents
            expanded_files = []
            for uf in uploaded_files:
                if uf.name.lower().endswith(".zip"):
                    import zipfile, io as _io
                    try:
                        with zipfile.ZipFile(_io.BytesIO(uf.read())) as zf:
                            for zip_name in zf.namelist():
                                lower = zip_name.lower()
                                if any(lower.endswith(ext) for ext in (".pdf", ".csv", ".xlsx", ".xls")):
                                    data = zf.read(zip_name)
                                    fake_file = _io.BytesIO(data)
                                    fake_file.name = zip_name
                                    expanded_files.append(fake_file)
                    except Exception as e:
                        st.warning(f"Could not open ZIP {uf.name}: {e}")
                else:
                    expanded_files.append(uf)

            # Process each file
            for f in expanded_files:
                fname = getattr(f, "name", "").lower()
                if fname.endswith(".pdf"):
                    texts = extract_text_from_pdf(f)
                    for i, t in enumerate(texts):
                        profiles_input.append({"text": t, "label": f"{f.name} — profile {i+1}", "from_pdf": True})
                else:
                    urls = extract_urls_from_csv(f)
                    for u in urls:
                        profiles_input.append({"url": u, "label": u})

            # Summary
            pdf_count = sum(1 for p in profiles_input if p.get("from_pdf"))
            url_count  = len(profiles_input) - pdf_count
            if profiles_input:
                parts = []
                if pdf_count: parts.append(f"{pdf_count} PDF profile(s)")
                if url_count:  parts.append(f"{url_count} URL(s) for BrightData")
                st.success(f"Ready: {', '.join(parts)}")
                with st.expander("Preview"):
                    for p in profiles_input[:10]:
                        if p.get("from_pdf"):
                            st.caption(f"PDF: {p['label']}")
                            st.text(p["text"][:200] + "…")
                        else:
                            st.caption(f"URL: {p['label']}")
                    if len(profiles_input) > 10:
                        st.caption(f"...and {len(profiles_input)-10} more")
            else:
                st.error("No profiles found. Check that your files are LinkedIn Recruiter exports or profile PDFs.")

# ── Run button ────────────────────────────────────────────────────────────────
st.markdown("---")
run_col, _ = st.columns([1, 3])
with run_col:
    run = st.button("🚀 Rank Candidates", type="primary", use_container_width=True)

if run:
    if not job_description.strip():
        st.error("Please enter a job description.")
        st.stop()

    weights = {
        "skills": w_skills,
        "experience": w_exp,
        "industry": w_industry,
        "growth": w_growth,
    }

    profiles_to_score = []

    # Paste text mode
    if data_source == "Paste profile text":
        profiles_to_score = [p for p in profiles_input if p["text"].strip()]
        if not profiles_to_score:
            st.error("Paste at least one candidate profile.")
            st.stop()

    # File upload / URL modes
    else:
        if not profiles_input:
            st.error("Please provide LinkedIn URLs, upload a CSV/XLSX, or upload a PDF.")
            st.stop()

        # PDF profiles go straight to scoring; URLs need BrightData
        pdf_profiles = [p for p in profiles_input if p.get("from_pdf")]
        url_profiles = [p for p in profiles_input if not p.get("from_pdf")]

        profiles_to_score.extend(pdf_profiles)

        if url_profiles:
            if not brightdata_key:
                st.error("Bright Data API key not configured in secrets.")
                st.stop()

            total = len(url_profiles)
            progress_bar = st.progress(0, text=f"Fetching profile 1 of {total}…")
            fetch_status = st.empty()

            for i, p in enumerate(url_profiles):
                progress_bar.progress(i / total, text=f"Fetching profile {i+1} of {total}…")
                fetch_status.info(f"Fetching: {p['url']}")
                text = fetch_brightdata(p["url"])
                if text:
                    profiles_to_score.append({"text": text, "label": p["url"]})
                    fetch_status.success(f"Fetched profile {i+1} of {total}")
                else:
                    fetch_status.warning(f"Could not fetch profile {i+1}: {p['url']}")

            progress_bar.progress(1.0, text=f"Done — {len(profiles_to_score)}/{total} profiles fetched.")
            fetch_status.empty()

            if not profiles_to_score:
                st.error(
                    "No profiles could be fetched. Check the warnings above and verify your "
                    "Bright Data API key and dataset access."
                )
                st.stop()

    # Score
    with st.spinner(f"Analyzing {len(profiles_to_score)} candidate(s)…"):
        try:
            result = score_candidates(job_description, profiles_to_score, weights)
        except json.JSONDecodeError:
            st.error("AI returned malformed JSON. Please try again.")
            st.stop()
        except anthropic.AuthenticationError:
            st.error("Invalid Anthropic API key.")
            st.stop()
        except Exception as e:
            st.error(f"Error during scoring: {e}")
            st.stop()

    candidates = sorted(
        result.get("candidates", []),
        key=lambda c: c.get("overall_score", 0),
        reverse=True,
    )

    # Results
    st.markdown("---")
    st.markdown(f"## 📊 Results — {len(candidates)} Candidate(s) Ranked")
    st.caption(f"Skills {w_skills}% · Experience {w_exp}% · Industry {w_industry}% · Trajectory {w_growth}%")

    for rank, c in enumerate(candidates, 1):
        score     = c.get("overall_score", 0)
        sc        = score_color(score)
        dims      = c.get("dimension_scores", {})
        strengths = c.get("top_strengths", [])
        flags     = c.get("red_flags", [])

        with st.container():
            st.markdown(f"""
<div class="candidate-card">
  <div style="display:flex; align-items:center; margin-bottom:12px; flex-wrap:wrap; gap:8px;">
    <span class="rank-badge">#{rank}</span>
    <span style="font-size:18px; font-weight:700; color:#1e293b;">{c.get('name','Unknown')}</span>
    <span style="color:#64748b; font-size:14px;">· {c.get('current_title','')} @ {c.get('current_company','')}</span>
    <span style="margin-left:auto;" class="score-pill {sc}">{score}/100</span>
  </div>
  <div class="divider"></div>
  <div style="display:grid; grid-template-columns: repeat(4,1fr); gap:12px; margin-bottom:14px;">
    <div><div class="section-label">Skills</div><span class="score-pill {score_color(dims.get('skills_match',0))}">{dims.get('skills_match',0)}</span></div>
    <div><div class="section-label">Experience</div><span class="score-pill {score_color(dims.get('experience_level',0))}">{dims.get('experience_level',0)}</span></div>
    <div><div class="section-label">Industry</div><span class="score-pill {score_color(dims.get('industry_fit',0))}">{dims.get('industry_fit',0)}</span></div>
    <div><div class="section-label">Trajectory</div><span class="score-pill {score_color(dims.get('career_trajectory',0))}">{dims.get('career_trajectory',0)}</span></div>
  </div>
  <div class="divider"></div>
  <p style="color:#334155; font-size:14px; margin:10px 0;">{c.get('summary','')}</p>
  <div style="display:grid; grid-template-columns:1fr 1fr; gap:16px; margin-top:12px;">
    <div>
      <div class="section-label">Strengths</div>
{"".join(f'<div class="strength-item">+ {s}</div>' for s in strengths) or '<span style="color:#6b7280">None noted</span>'}
    </div>
    <div>
      <div class="section-label">Red Flags</div>
{"".join(f'<div class="flag-item">- {f}</div>' for f in flags) or '<span style="color:#6b7280">None noted</span>'}
    </div>
  </div>
  <div style="margin-top:14px;">
    <div class="section-label">Suggested Outreach</div>
    <div class="outreach-box">{c.get('outreach_message','')}</div>
  </div>
</div>
""", unsafe_allow_html=True)

    # CSV Export
    st.markdown("---")
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow([
        "Rank", "Name", "Title", "Company", "Overall Score",
        "Skills", "Experience", "Industry", "Trajectory",
        "Strengths", "Red Flags", "Summary", "Outreach",
    ])
    for rank, c in enumerate(candidates, 1):
        d = c.get("dimension_scores", {})
        writer.writerow([
            rank, c.get("name"), c.get("current_title"), c.get("current_company"),
            c.get("overall_score"),
            d.get("skills_match"), d.get("experience_level"),
            d.get("industry_fit"), d.get("career_trajectory"),
            " | ".join(c.get("top_strengths", [])),
            " | ".join(c.get("red_flags", [])),
            c.get("summary"), c.get("outreach_message"),
        ])

    st.download_button(
        "Export Results as CSV",
        data=buf.getvalue(),
        file_name="candidates_ranked.csv",
        mime="text/csv",
    )
