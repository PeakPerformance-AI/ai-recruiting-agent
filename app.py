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
supabase_url   = st.secrets.get("SUPABASE_URL", "")
supabase_key   = st.secrets.get("SUPABASE_KEY", "")

# ── Supabase client ───────────────────────────────────────────────────────────
_sb = None
if supabase_url and supabase_key:
    try:
        from supabase import create_client
        _sb = create_client(supabase_url, supabase_key)
    except Exception:
        pass

def save_search(job_description: str, candidates: list, weights: dict):
    """Save a completed search to Supabase."""
    if not _sb:
        return
    try:
        # Use first candidate's title as a label, or fall back to first line of JD
        job_title = next(
            (c.get("current_title", "") for c in candidates if c.get("current_title")),
            job_description.strip().splitlines()[0][:80] if job_description.strip() else "Untitled search"
        )
        _sb.table("searches").insert({
            "job_title":       job_title,
            "job_description": job_description,
            "candidates":      candidates,
            "weights":         weights,
        }).execute()
    except Exception:
        pass

def load_searches() -> list:
    """Return past searches ordered by most recent first."""
    if not _sb:
        return []
    try:
        resp = _sb.table("searches").select("*").order("created_at", desc=True).limit(30).execute()
        return resp.data or []
    except Exception:
        return []

def delete_search(search_id: str):
    """Delete a saved search by ID."""
    if not _sb:
        return
    try:
        _sb.table("searches").delete().eq("id", search_id).execute()
    except Exception:
        pass

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## ⚙️ Configuration")

    # ── Past Searches ─────────────────────────────────────────────────────────
    if _sb:
        st.markdown("### 📂 Past Searches")
        if st.button("↺ Refresh", key="refresh_searches"):
            st.rerun()
        past = load_searches()
        if past:
            for s in past:
                created = s.get("created_at", "")[:10]
                n = len(s.get("candidates") or [])
                label = f"{s.get('job_title','Untitled')} · {created} · {n} candidates"
                col_load, col_del = st.columns([5, 1])
                with col_load:
                    if st.button(label, key=f"load_{s['id']}"):
                        st.session_state["loaded_search"] = s
                        st.rerun()
                with col_del:
                    if st.button("✕", key=f"del_{s['id']}"):
                        delete_search(s["id"])
                        st.rerun()
        else:
            st.caption("No saved searches yet.")
        st.markdown("---")

    st.markdown("### Input Method")
    data_source = st.radio(
        "How will you provide candidates?",
        ["Paste profile text", "LinkedIn URLs", "Upload File (CSV / XLSX / PDF)"],
        index=2,
    )
    st.markdown("---")
    st.markdown("### Scoring Weights")
    w_skills   = st.slider("Skills match",      0, 100, 35, step=5)
    w_exp      = st.slider("Experience level",  0, 100, 30, step=5)
    w_industry = st.slider("Industry fit",      0, 100, 20, step=5)
    w_growth   = st.slider("Career trajectory", 0, 100, 15, step=5)
    total_w = w_skills + w_exp + w_industry + w_growth
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
    """Extract text from a PDF. Treats the entire PDF as one candidate profile."""
    import pdfplumber, io as _io
    try:
        with pdfplumber.open(_io.BytesIO(uploaded_file.read())) as pdf:
            full_text = "\n".join(
                page.extract_text() or "" for page in pdf.pages
            ).strip()
        return [full_text] if full_text else []
    except Exception as e:
        st.warning(f"Could not read PDF: {e}")
        return []


def build_system_prompt(weights: dict) -> str:
    return f"""You are an expert technical recruiter with 15 years of experience.
Your job is to analyze LinkedIn candidate profiles against a job description
and produce structured, objective assessments.

BIAS PREVENTION — these rules are mandatory and override all other considerations:
1. Names must have zero influence on any score, flag, strength, or summary. Do not make any inference about ethnicity, nationality, gender, religion, or background from a candidate's name.
2. Location is a factual data point only. Flag a location mismatch only if the job description explicitly requires a specific location. Never make assumptions about work ethic, communication style, cultural fit, or professionalism based on where a candidate is from.
3. Gender is irrelevant. Ignore any gender signals from names, pronouns, or profile language. Do not adjust scores or language based on perceived gender.
4. Every score, strength, and red flag must be directly traceable to a specific requirement stated in the job description. If you cannot point to a JD requirement, it is not a valid flag or strength.
5. Apply identical standards to all candidates regardless of their background. Job hopping, gaps, and missing skills must be flagged consistently — not selectively based on who the candidate is.
6. Do not use language that could reflect bias in summaries or outreach messages. Evaluate all candidates as professionals defined only by their skills, experience, and fit for the role.

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
      "top_strengths": ["string"],
      "red_flags": ["string"],
      "summary": "2-3 sentence narrative on fit",
      "outreach_message": "Personalized 3-sentence LinkedIn message"
    }}
  ]
}}

Be honest. Not every candidate is a strong fit. Red flags are important.

List a number of top_strengths that is equal to or close to the number of red_flags for each candidate. If there are 4 red flags, aim for 4 strengths. If there are no red flags, list 2-3 strengths. Keep the lists balanced. Cap both lists at 6-7 items maximum — include only the most important ones. Do not pad with minor or obvious points.

RED FLAG RULES — always check for these and flag them explicitly if present:
1. Job hopping: any role lasting less than 12 months at a single company (exclude internships, contract roles, or layoffs if clearly stated)
2. Employment gaps: any gap between roles exceeding 4 months — note the approximate dates and duration
3. Missing required skills or certifications: if the job description lists required skills, tools, or certifications that are absent from the profile, flag each one individually
4. Location mismatch: if the candidate's current location differs from the target location in the job description, flag it — note both locations
5. Use your own judgment for any other meaningful red flags (e.g. no management experience for a management role, consistent downward career trajectory, etc.)

If data is missing or the profile is incomplete, flag that too rather than assuming the best.
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
        max_tokens=8192,
        system=build_system_prompt(weights),
        messages=[{"role": "user", "content": user_msg}],
    )

    raw = response.content[0].text.strip()

    # Strip markdown code fences if present
    if "```" in raw:
        parts = raw.split("```")
        for part in parts:
            part = part.strip()
            if part.startswith("json"):
                part = part[4:].strip()
            if part.startswith("{"):
                raw = part
                break

    # Extract JSON object even if there's surrounding text
    start = raw.find("{")
    end   = raw.rfind("}") + 1
    if start != -1 and end > start:
        raw = raw[start:end]

    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        st.expander("Raw Claude response (debug)").code(raw[:3000])
        raise


def score_color(score: int) -> str:
    if score >= 75:
        return "score-high"
    if score >= 50:
        return "score-medium"
    return "score-low"


def render_exports(candidates: list, job_description: str = ""):
    """Render the three export download buttons."""
    st.markdown("---")
    st.markdown("#### ⬇️ Export")
    export_cols = st.columns(3)

    # Full results CSV
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
    with export_cols[0]:
        st.download_button(
            "📊 Full results (CSV)",
            data=buf.getvalue(),
            file_name="candidates_ranked.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Outreach-only CSV
    out_csv = io.StringIO()
    out_writer = csv.writer(out_csv)
    out_writer.writerow(["Rank", "Name", "Title", "Company", "Score", "Outreach Message"])
    for rank, c in enumerate(candidates, 1):
        out_writer.writerow([
            rank, c.get("name"), c.get("current_title"),
            c.get("current_company"), c.get("overall_score"),
            c.get("outreach_message"),
        ])
    with export_cols[1]:
        st.download_button(
            "💬 Outreach messages (CSV)",
            data=out_csv.getvalue(),
            file_name="outreach_messages.csv",
            mime="text/csv",
            use_container_width=True,
        )

    # Outreach-only TXT
    today = time.strftime("%Y-%m-%d")
    jd_title = job_description.strip().splitlines()[0][:60] if job_description.strip() else "Search"
    txt_lines = [
        "AI Recruiting Agent — Outreach Messages",
        f"Job: {jd_title}",
        f"Generated: {today}",
        "=" * 48,
    ]
    for rank, c in enumerate(candidates, 1):
        txt_lines += [
            "",
            f"#{rank} — {c.get('name','')} (Score: {c.get('overall_score','')})",
            f"{c.get('current_title','')} @ {c.get('current_company','')}",
            "-" * 48,
            c.get("outreach_message", ""),
        ]
    with export_cols[2]:
        st.download_button(
            "💬 Outreach messages (TXT)",
            data="\n".join(txt_lines),
            file_name="outreach_messages.txt",
            mime="text/plain",
            use_container_width=True,
        )


def summary_bar(candidates: list):
    """Show a summary bar with strong / maybe / weak counts."""
    strong = sum(1 for c in candidates if c.get("overall_score", 0) >= 75)
    maybe  = sum(1 for c in candidates if 50 <= c.get("overall_score", 0) < 75)
    weak   = sum(1 for c in candidates if c.get("overall_score", 0) < 50)
    parts  = []
    if strong: parts.append(f'<span class="score-pill score-high">✓ {strong} Strong fit{"s" if strong != 1 else ""}</span>')
    if maybe:  parts.append(f'<span class="score-pill score-medium">~ {maybe} Maybe{"s" if maybe != 1 else ""}</span>')
    if weak:   parts.append(f'<span class="score-pill score-low">✗ {weak} Weak fit{"s" if weak != 1 else ""}</span>')
    st.markdown(
        f'<div style="display:flex; gap:10px; flex-wrap:wrap; margin-bottom:16px;">{"".join(parts)}</div>',
        unsafe_allow_html=True,
    )


def apply_pin_order(candidates: list, pinned: list, order: list) -> list:
    """Return candidates sorted by: pinned first (in pin order), then by custom order list."""
    name_to_c = {c.get("name", ""): c for c in candidates}
    # Build ordered list: pinned names first, then remaining in order
    all_names_ordered = pinned + [n for n in order if n not in pinned]
    # Add any names not in order list at the end
    for c in candidates:
        if c.get("name", "") not in all_names_ordered:
            all_names_ordered.append(c.get("name", ""))
    return [name_to_c[n] for n in all_names_ordered if n in name_to_c]


def render_candidates(candidates: list, state_key: str):
    """Render candidate cards with pin and reorder controls."""
    pinned_key = f"{state_key}_pinned"
    order_key  = f"{state_key}_order"

    if pinned_key not in st.session_state:
        st.session_state[pinned_key] = []
    if order_key not in st.session_state:
        st.session_state[order_key] = [c.get("name", "") for c in candidates]

    # Sync order list if candidates changed
    current_names = [c.get("name", "") for c in candidates]
    for n in current_names:
        if n not in st.session_state[order_key]:
            st.session_state[order_key].append(n)

    pinned  = st.session_state[pinned_key]
    ordered = apply_pin_order(candidates, pinned, st.session_state[order_key])

    summary_bar(candidates)

    for rank, c in enumerate(ordered, 1):
        name      = c.get("name", "Unknown")
        score     = c.get("overall_score", 0)
        sc        = score_color(score)
        dims      = c.get("dimension_scores", {})
        strengths = c.get("top_strengths", [])
        flags     = c.get("red_flags", [])
        is_pinned = name in pinned
        pin_label = "📌 Pinned" if is_pinned else "📌 Pin"
        pin_style = "color:#3b82f6; font-weight:700;" if is_pinned else "color:#94a3b8;"

        with st.container():
            st.markdown(f"""
<div class="candidate-card">
  <div style="display:flex; align-items:center; margin-bottom:12px; flex-wrap:wrap; gap:8px;">
    <span class="rank-badge">#{rank}</span>
    <span style="font-size:18px; font-weight:700; color:#1e293b;">{name}</span>
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

            # Copy outreach button (components.html embeds text directly — no clipboard permission issues)
            outreach = c.get("outreach_message", "")
            if outreach:
                import streamlit.components.v1 as _components
                outreach_json = json.dumps(outreach)
                _components.html(f"""
                <button id="btn" style="background:#f1f5f9; color:#475569; border:1px solid #e2e8f0;
                    padding:4px 12px; border-radius:6px; cursor:pointer; font-size:12px;
                    font-family:sans-serif; margin-top:-6px;">
                    📋 Copy Outreach Message
                </button>
                <script>
                document.getElementById('btn').addEventListener('click', function() {{
                    var text = {outreach_json};
                    var btn = this;
                    function done() {{
                        btn.textContent = '✓ Copied!';
                        setTimeout(function() {{ btn.textContent = '📋 Copy Outreach Message'; }}, 1500);
                    }}
                    if (navigator.clipboard && navigator.clipboard.writeText) {{
                        navigator.clipboard.writeText(text).then(done).catch(function() {{ fallback(text); done(); }});
                    }} else {{
                        fallback(text); done();
                    }}
                    function fallback(t) {{
                        var ta = document.createElement('textarea');
                        ta.value = t; document.body.appendChild(ta);
                        ta.select(); document.execCommand('copy');
                        document.body.removeChild(ta);
                    }}
                }});
                </script>
                """, height=38)

            # Controls row
            ctrl1, ctrl2, ctrl3, _ = st.columns([1, 1, 1, 5])
            with ctrl1:
                if st.button(pin_label, key=f"{state_key}_pin_{name}"):
                    if is_pinned:
                        st.session_state[pinned_key].remove(name)
                    else:
                        st.session_state[pinned_key].insert(0, name)
                    st.rerun()
            with ctrl2:
                cur_order = st.session_state[order_key]
                idx = cur_order.index(name) if name in cur_order else -1
                if idx > 0:
                    if st.button("▲", key=f"{state_key}_up_{name}"):
                        cur_order[idx], cur_order[idx - 1] = cur_order[idx - 1], cur_order[idx]
                        st.rerun()
            with ctrl3:
                if idx < len(cur_order) - 1 and idx != -1:
                    if st.button("▼", key=f"{state_key}_down_{name}"):
                        cur_order[idx], cur_order[idx + 1] = cur_order[idx + 1], cur_order[idx]
                        st.rerun()


# ── Back to top button (fixed, always visible) ────────────────────────────────
import streamlit.components.v1 as _components_top
_components_top.html("""
<style>
  #back-to-top {
    position: fixed;
    bottom: 28px;
    right: 28px;
    z-index: 9999;
    background: linear-gradient(135deg, #3b82f6, #6366f1);
    color: white;
    border: none;
    padding: 10px 16px;
    border-radius: 50px;
    cursor: pointer;
    font-size: 13px;
    font-family: sans-serif;
    font-weight: 600;
    box-shadow: 0 4px 12px rgba(99,102,241,0.4);
    opacity: 0;
    transition: opacity 0.3s;
    pointer-events: none;
  }
  #back-to-top.visible {
    opacity: 1;
    pointer-events: all;
  }
</style>
<button id="back-to-top" onclick="window.parent.scrollTo({top:0, behavior:'smooth'})">
  ↑ Top
</button>
<script>
  window.parent.addEventListener('scroll', function() {
    var btn = document.getElementById('back-to-top');
    if (window.parent.scrollY > 400) {
      btn.classList.add('visible');
    } else {
      btn.classList.remove('visible');
    }
  });
</script>
""", height=0)

# ── Load a past search into session state ─────────────────────────────────────
if "loaded_search" in st.session_state:
    s = st.session_state.pop("loaded_search")
    st.session_state["jd_textarea"]      = s.get("job_description", "")
    st.session_state["loaded_candidates"] = s.get("candidates", [])
    st.session_state["loaded_weights"]    = s.get("weights", {})

# ── Main UI ───────────────────────────────────────────────────────────────────
st.markdown("# 🎯 AI Recruiting Agent")
st.markdown("Rank and score candidates against any job description in seconds.")
st.markdown("---")

col_jd, col_profiles = st.columns([1, 1], gap="large")

with col_jd:
    st.markdown("### 📋 Job Description")

    # Initialize session state
    if "jd_textarea" not in st.session_state:
        st.session_state["jd_textarea"] = ""
    if "jd_last_file" not in st.session_state:
        st.session_state["jd_last_file"] = None

    jd_file = st.file_uploader(
        "Upload job description (optional — PDF, DOCX, or TXT)",
        type=["pdf", "txt", "docx"],
        key="jd_upload",
    )

    # Only extract when a new file is uploaded
    if jd_file is not None and jd_file.name != st.session_state["jd_last_file"]:
        st.session_state["jd_last_file"] = jd_file.name
        fname = jd_file.name.lower()
        try:
            if fname.endswith(".pdf"):
                import pdfplumber, io as _io
                with pdfplumber.open(_io.BytesIO(jd_file.read())) as pdf:
                    st.session_state["jd_textarea"] = "\n".join(p.extract_text() or "" for p in pdf.pages)
            elif fname.endswith(".docx"):
                import docx as _docx, io as _io
                doc = _docx.Document(_io.BytesIO(jd_file.read()))
                st.session_state["jd_textarea"] = "\n".join(para.text for para in doc.paragraphs)
            else:
                st.session_state["jd_textarea"] = jd_file.read().decode("utf-8", errors="ignore")
        except Exception as e:
            st.warning(f"Could not read file: {e}")

    job_description = st.text_area(
        "Job description",
        height=320,
        placeholder="Senior Backend Engineer\n\nRequirements:\n- 5+ years Python\n- Distributed systems experience\n...",
        label_visibility="collapsed",
        key="jd_textarea",
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
        "skills":     w_skills,
        "experience": w_exp,
        "industry":   w_industry,
        "growth":     w_growth,
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

    # Auto-save to Supabase
    save_search(job_description, candidates, weights)

    # Results
    st.markdown("---")
    st.markdown(f"## 📊 Results — {len(candidates)} Candidate(s) Ranked")
    st.caption(f"Skills {w_skills}% · Experience {w_exp}% · Industry {w_industry}% · Trajectory {w_growth}%")
    render_exports(candidates, job_description)
    render_candidates(candidates, "results")

# ── Show loaded past search (if user clicked one in sidebar) ──────────────────
elif "loaded_candidates" in st.session_state:
    loaded_candidates = st.session_state["loaded_candidates"]
    loaded_weights    = st.session_state.get("loaded_weights", {})
    st.markdown("---")
    st.markdown(f"## 📂 Loaded Search — {len(loaded_candidates)} Candidate(s)")
    st.caption(
        f"Skills {loaded_weights.get('skills', '?')}% · "
        f"Experience {loaded_weights.get('experience', '?')}% · "
        f"Industry {loaded_weights.get('industry', '?')}% · "
        f"Trajectory {loaded_weights.get('growth', '?')}%"
    )
    if st.button("✕ Clear loaded search"):
        del st.session_state["loaded_candidates"]
        st.rerun()

    render_exports(loaded_candidates, st.session_state.get("jd_textarea", ""))
    render_candidates(loaded_candidates, "loaded")
