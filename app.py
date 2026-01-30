import re
import requests
import streamlit as st
import xml.etree.ElementTree as ET
from openai import OpenAI

# -------------------- Page --------------------
st.set_page_config(page_title="ED Copilot (Prototype)", layout="centered")

st.title("ED Copilot (Prototype)")
st.caption("Author: Douglas Moss, MD")

with st.expander("Disclaimer", expanded=True):
    st.write(
        "This is a prototype demonstration. It may be wrong or incomplete. "
        "Do not use for real patient care. Do not enter patient identifiers."
    )

# -------------------- Demo passcode gate --------------------
pw = st.text_input("Passcode", type="password")
if pw != st.secrets.get("APP_PASSWORD", ""):
    st.stop()

# -------------------- Session state --------------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "pending_prompt" not in st.session_state:
    st.session_state["pending_prompt"] = None
if "last_hits" not in st.session_state:
    st.session_state["last_hits"] = None

# -------------------- Controls --------------------
mode = st.selectbox(
    "Mode",
    ["Workup/Treatment/Disposition", "Discharge instructions (patient-friendly)"],
    index=0,
)

with st.expander("Retrieval settings", expanded=False):
    retmax = st.slider("PubMed results to pull", 3, 10, 5)
    include_abstracts = st.toggle("Include abstracts (slower, better grounding)", value=False)

# Quick prompts (hide after chat starts)
samples = [
    "Chest pain, rule-out ACS with high-sensitivity troponin in the ED",
    "Suspected pulmonary embolism, when to image vs D-dimer",
    "Sepsis initial bundle in the ED, antibiotics and fluids",
    "New-onset atrial fibrillation with RVR, rate vs rhythm and disposition",
    "DKA initial management, potassium and insulin",
]

def _on_pick_sample():
    sel = st.session_state.get("quick_pick", "")
    if sel and sel != "Select a sample...":
        st.session_state["pending_prompt"] = sel

if len(st.session_state["messages"]) == 0:
    st.selectbox(
        "Quick prompt",
        ["Select a sample..."] + samples,
        key="quick_pick",
        on_change=_on_pick_sample,
    )

# Clear chat
if st.button("Clear chat"):
    st.session_state["messages"] = []
    st.session_state["pending_prompt"] = None
    st.session_state["last_hits"] = None
    st.session_state["quick_pick"] = ""
    st.rerun()

# -------------------- PubMed helpers --------------------
NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

STOPWORDS = {
    "adult","peds","pediatric","initial","management","workup","labs","lab","treatment","treatments",
    "criteria","admission","disposition","dx","ddx","ed","em","er","the","a","an","and","or","to",
    "for","of","with","without","in","on","at","by","from","vs","versus","suspected","possible",
    "patient","patients","male","female","man","woman","yo","y/o","year","old","criteria","consider",
    "how","what","when","why","should","could","would","can","do","does","did","best","ways","way",
    "manage","management","treat","treatment","workup","evaluation","approach"
}

SYNONYMS = {
    "dka": "diabetic ketoacidosis",
    "pe": "pulmonary embolism",
    "acs": "acute coronary syndrome",
    "ich": "intracerebral hemorrhage",
    "tbi": "traumatic brain injury",
    "uti": "urinary tract infection",
    "copd": "chronic obstructive pulmonary disease",
    "afib": "atrial fibrillation",
    "rvr": "rapid ventricular response",
}

def make_pubmed_term(q: str) -> str:
    """
    Produces a reasonable first-pass PubMed query for natural language.
    We avoid quoting the full question because that almost always kills recall.
    """
    q = (q or "").strip()
    if not q:
        return ""

    raw = q.lower()
    for k, v in SYNONYMS.items():
        raw = re.sub(rf"\b{k}\b", v, raw)

    cleaned = re.sub(r"[^a-z0-9\s]", " ", raw)
    tokens = [t for t in cleaned.split() if t and t not in STOPWORDS]

    seen = set()
    uniq = []
    for t in tokens:
        if t not in seen:
            uniq.append(t)
            seen.add(t)

    key = uniq[:6] if uniq else []

    # Prefer OR across key terms in title/abstract, let PubMed do its own mapping too.
    # Example: (chest[tiab] OR pain[tiab] OR troponin[tiab]) OR (chest pain troponin)
    if key:
        tiab_or = " OR ".join([f"{t}[tiab]" for t in key])
        raw_fallback = " ".join(key)
        return f"({tiab_or}) OR ({raw_fallback})"

    return q

@st.cache_data(ttl=3600)
def pubmed_search(term: str, retmax: int = 5):
    """
    Progressive relaxation so natural language questions still get hits.
    """
    q = (term or "").strip()
    if not q:
        return []

    cooked = make_pubmed_term(q)

    candidates = [
        cooked,                                  # structured-ish
        cooked.replace(" AND ", " OR "),         # (in case any ANDs slip in)
        re.sub(r"\[Title/Abstract\]", "[tiab]", cooked),
        " ".join(re.findall(r"[A-Za-z0-9]+", q)[:8]),  # raw-ish keywords, PubMed translation helps
        q,                                       # absolute fallback: raw question
    ]

    for t in candidates:
        t = (t or "").strip()
        if not t:
            continue

        r = requests.get(
            NCBI_ESEARCH,
            params={
                "db": "pubmed",
                "term": t,
                "retmode": "json",
                "retmax": retmax,
                "sort": "relevance",
            },
            timeout=20,
        )
        r.raise_for_status()
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if ids:
            return ids

    return []

@st.cache_data(ttl=3600)
def pubmed_summaries(pmids):
    if not pmids:
        return []

    r = requests.get(
        NCBI_ESUMMARY,
        params={"db": "pubmed", "id": ",".join(pmids), "retmode": "json"},
        timeout=20,
    )
    r.raise_for_status()
    data = r.json()

    out = []
    for pmid in pmids:
        item = data.get("result", {}).get(pmid, {})
        if not item:
            continue
        out.append(
            {
                "pmid": pmid,
                "title": (item.get("title", "") or "").strip().rstrip("."),
                "journal": item.get("fulljournalname", "") or "",
                "year": ((item.get("pubdate", "") or "").split(" ")[0]) or "",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        )
    return out

@st.cache_data(ttl=3600)
def pubmed_abstracts(pmids):
    """
    Fetch abstracts via EFetch (XML). Returns {pmid: abstract_text}.
    """
    if not pmids:
        return {}

    r = requests.get(
        NCBI_EFETCH,
        params={
            "db": "pubmed",
            "id": ",".join(pmids),
            "retmode": "xml",
        },
        timeout=25,
    )
    r.raise_for_status()

    root = ET.fromstring(r.text)
    out = {}

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//MedlineCitation/PMID")
        pmid = (pmid_el.text or "").strip() if pmid_el is not None else ""
        if not pmid:
            continue

        abs_parts = []
        for a in article.findall(".//Abstract/AbstractText"):
            label = a.attrib.get("Label")
            txt = "".join(a.itertext()).strip()
            if not txt:
                continue
            abs_parts.append(f"{label}: {txt}" if label else txt)

        if abs_parts:
            out[pmid] = "\n".join(abs_parts)

    return out

def build_metadata_context(summaries, abstracts=None, max_items=5, abstract_chars=900):
    use = summaries[:max_items]
    lines = []
    allowed_pmids = []
    abstracts = abstracts or {}

    for h in use:
        pmid = h["pmid"]
        allowed_pmids.append(pmid)

        base = f"- {h['title']} ({h['journal']}, {h['year']}). PMID {pmid}. {h['url']}"
        ab = (abstracts.get(pmid) or "").strip()

        if ab:
            ab = ab[:abstract_chars].rstrip()
            base += f"\n  Abstract (truncated): {ab}"

        lines.append(base)

    return "\n".join(lines) if lines else "No PubMed results returned.", allowed_pmids

# -------------------- LLM --------------------
def generate_answer(prior_messages, question: str, meta_context: str, allowed_pmids, mode: str):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    allowed_str = ", ".join(allowed_pmids) if allowed_pmids else "none"

    system = (
        "You are an emergency medicine attending helping another ED clinician on shift. "
        "Be concise and practical. "
        "Do not ask for or include PHI. "
        "Use only the provided PubMed metadata/abstracts for evidence. "
        "If abstracts are not provided, explicitly note that evidence grounding is limited. "
        "If critical details are missing, ask up to 3 clarifying questions first, then give a best-effort answer. "
        "Only cite PMIDs that appear in Allowed PMIDs. "
        "If Allowed PMIDs is not 'none', you MUST cite at least 1 PMID from it."
    )

    pmid_rule = (
        f"Allowed PMIDs: {allowed_str}\n"
        "RULE: If Allowed PMIDs is not 'none', end with 'Citations: ' followed by 1–3 PMIDs from Allowed PMIDs.\n"
        "Do not write 'none' if Allowed PMIDs is not 'none'.\n"
    )

    if mode == "Discharge instructions (patient-friendly)":
        user = f"""User question:
{question}

PubMed results (metadata{' + abstracts' if 'Abstract (truncated):' in meta_context else ''}):
{meta_context}

{pmid_rule}

Write patient-friendly discharge instructions at about an 8th-grade reading level.
Include: brief explanation, what to do at home, meds if relevant (general), red flags to return, follow-up.
Keep it brief.
End with: "This is not medical advice and is for demo only."
"""
        max_tokens = 350
    else:
        user = f"""User question:
{question}

PubMed results (metadata{' + abstracts' if 'Abstract (truncated):' in meta_context else ''}):
{meta_context}

{pmid_rule}

Output (keep brief):
- Quick take (max 3 bullets)
- Workup (labs/imaging) (max 6 bullets)
- Treatment (max 6 bullets)
- Disposition (max 4 bullets)
- Citations: 1–3 PMIDs (required if Allowed PMIDs is not 'none')
"""
        max_tokens = 450

    convo = [{"role": "system", "content": system}] + prior_messages + [{"role": "user", "content": user}]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=convo,
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content

# -------------------- Render history --------------------
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# -------------------- Chat input --------------------
typed = st.chat_input("Ask an ED question (no PHI).")
prompt = (typed or "").strip() if typed else None

# Auto-submit sample if selected
if not prompt and st.session_state["pending_prompt"]:
    prompt = st.session_state["pending_prompt"]
    st.session_state["pending_prompt"] = None

# -------------------- Main interaction --------------------
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching PubMed..."):
            pmids = pubmed_search(prompt, retmax=retmax)
            summaries = pubmed_summaries(pmids)

            if not summaries:
                st.write("No PubMed results found. Try fewer words or more general terms.")
                meta_context, allowed_pmids = "No PubMed results returned.", []
                abstract_map = {}
            else:
                # Only fetch abstracts for the items we will show/use
                show_pmids = [h["pmid"] for h in summaries[:retmax]]
                abstract_map = pubmed_abstracts(show_pmids) if include_abstracts else {}

                st.subheader("Top PubMed results")
                for i, h in enumerate(summaries[:retmax], start=1):
                    meta = " · ".join([x for x in [h["journal"], h["year"], f"PMID {h['pmid']}"] if x])
                    st.markdown(f"**{i}. [{h['title'] or '(No title returned)'}]({h['url']})**")
                    st.caption(meta)

                    if include_abstracts:
                        ab = (abstract_map.get(h["pmid"]) or "").strip()
                        if ab:
                            with st.expander("Abstract", expanded=False):
                                st.write(ab)

                meta_context, allowed_pmids = build_metadata_context(
                    summaries, abstracts=abstract_map, max_items=retmax
                )

        with st.spinner("Generating answer..."):
            try:
                prior = st.session_state["messages"][-6:-1]  # keep it light
                answer = generate_answer(prior, prompt, meta_context, allowed_pmids, mode)
                st.write(answer)

                pmids_in_answer = re.findall(r"\b\d{7,8}\b", answer)
                pmids_in_answer = list(dict.fromkeys(pmids_in_answer))
                if pmids_in_answer:
                    st.caption("PMIDs cited:")
                    st.markdown(" ".join([f"[{p}](https://pubmed.ncbi.nlm.nih.gov/{p}/)" for p in pmids_in_answer]))

                st.session_state["messages"].append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Error: {e}")
