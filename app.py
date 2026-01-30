import re
import requests
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="ED Copilot (Prototype)", layout="centered")

st.title("ED Copilot (Prototype)")
st.caption("Author: Douglas Moss, MD · Prototype for demo only. Not for clinical use. No PHI.")


with st.expander("Disclaimer", expanded=True):
    st.write(
        "This is a prototype demonstration. It may be wrong or incomplete. "
        "Do not use for real patient care. Do not enter patient identifiers."
    )

# Passcode gate (demo only)
pw = st.text_input("Passcode", type="password")
if pw != st.secrets.get("APP_PASSWORD", ""):
    st.stop()

# ----- Controls -----
mode = st.selectbox(
    "Mode",
    ["Workup/Treatment/Disposition", "Discharge instructions (patient-friendly)"],
    index=0,
)

with st.expander("Retrieval settings", expanded=False):
    retmax = st.slider("PubMed results to pull", 3, 10, 5)

samples = [
    "Chest pain, rule-out ACS with high-sensitivity troponin in the ED",
    "Suspected pulmonary embolism, when to image vs D-dimer",
    "Sepsis initial bundle in the ED, antibiotics and fluids",
    "New-onset atrial fibrillation with RVR, rate vs rhythm and disposition",
    "DKA initial management, potassium and insulin",
]

col1, col2 = st.columns([2, 1])
with col1:
    pick = st.selectbox("Quick prompt", ["Custom"] + samples, index=0)
with col2:
    send_sample = st.button("Send", use_container_width=True)

# ----- PubMed helpers (metadata only) -----
NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

STOPWORDS = {
    "adult","peds","pediatric","initial","management","workup","labs","lab","treatment","treatments",
    "criteria","admission","disposition","dx","ddx","ed","em","er","the","a","an","and","or","to",
    "for","of","with","without","in","on","at","by","from","vs","versus","suspected","possible",
    "patient","patients","male","female","man","woman","yo","y/o","year","old","criteria","consider"
}
SYNONYMS = {
    "dka": "diabetic ketoacidosis",
    "pe": "pulmonary embolism",
    "acs": "acute coronary syndrome",
    "ich": "intracerebral hemorrhage",
    "tbi": "traumatic brain injury",
    "uti": "urinary tract infection",
    "copd": "chronic obstructive pulmonary disease",
}

def make_pubmed_term(q: str) -> str:
    q = (q or "").strip()
    if not q:
        return q

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

    key = uniq[:5] if uniq else []
    phrase = f"\"{q}\""
    if key:
        and_bits = " AND ".join([f"{t}[Title/Abstract]" for t in key])
        return f"({phrase}) OR ({and_bits})"
    return phrase

@st.cache_data(ttl=3600)
def pubmed_search(term: str, retmax: int = 5):
    r = requests.get(
        NCBI_ESEARCH,
        params={
            "db": "pubmed",
            "term": make_pubmed_term(term),
            "retmode": "json",
            "retmax": retmax,
            "sort": "relevance",
        },
        timeout=20,
    )
    r.raise_for_status()
    return r.json().get("esearchresult", {}).get("idlist", [])

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

def build_metadata_context(summaries, max_items=5):
    use = summaries[:max_items]
    lines = []
    allowed_pmids = []
    for h in use:
        allowed_pmids.append(h["pmid"])
        lines.append(
            f"- {h['title']} ({h['journal']}, {h['year']}). PMID {h['pmid']}. {h['url']}"
        )
    return "\n".join(lines) if lines else "No PubMed results returned.", allowed_pmids

# ----- LLM -----
def generate_answer(prior_messages, question: str, meta_context: str, allowed_pmids, mode: str):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    allowed_str = ", ".join(allowed_pmids) if allowed_pmids else "none"

    system = (
        "You are an emergency medicine attending helping another ED clinician on shift. "
        "Be concise and practical. "
        "Do not ask for or include PHI. "
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

PubMed results (metadata only):
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

PubMed results (metadata only):
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

# ----- Chat state -----
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "pending_prompt" not in st.session_state:
    st.session_state["pending_prompt"] = None
if "last_hits" not in st.session_state:
    st.session_state["last_hits"] = None

# Clear chat button
if st.button("Clear chat"):
    st.session_state["messages"] = []
    st.session_state["pending_prompt"] = None
    st.session_state["last_hits"] = None
    st.rerun()

# If they clicked Send on a sample/custom
if send_sample and pick != "Custom":
    st.session_state["pending_prompt"] = pick


# Render history
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# Chat input
typed = st.chat_input("Ask an ED question (no PHI).")
prompt = (typed or "").strip() if typed else None

# Use pending prompt if set
if not prompt and st.session_state["pending_prompt"]:
    prompt = st.session_state["pending_prompt"]
    st.session_state["pending_prompt"] = None

if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching PubMed..."):
            pmids = pubmed_search(prompt, retmax=retmax)
            summaries = pubmed_summaries(pmids)

            # show top hits
            if not summaries:
                st.write("No PubMed results found. Try fewer words or more general terms.")
                meta_context, allowed_pmids = "No PubMed results returned.", []
            else:
                st.subheader("Top PubMed results")
                for i, h in enumerate(summaries[:retmax], start=1):
                    meta = " · ".join([x for x in [h["journal"], h["year"], f"PMID {h['pmid']}"] if x])
                    st.markdown(f"**{i}. [{h['title'] or '(No title returned)'}]({h['url']})**")
                    st.caption(meta)

                meta_context, allowed_pmids = build_metadata_context(summaries, max_items=retmax)

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
