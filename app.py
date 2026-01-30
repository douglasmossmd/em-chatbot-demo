import re
import requests
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="ED Copilot (Prototype)", layout="centered")

st.title("ED Copilot (Prototype)")
st.caption("Prototype for interview demo only. Not for clinical use. No PHI.")

with st.expander("Disclaimer", expanded=True):
    st.write(
        "This is a prototype demonstration. It may be wrong or incomplete. "
        "Do not use for real patient care. Do not enter patient identifiers."
    )

# Simple access gate (demo only)
pw = st.text_input("Passcode", type="password")
if pw != st.secrets.get("APP_PASSWORD", ""):
    st.stop()

NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

STOPWORDS = {
    "adult","peds","pediatric","initial","management","workup","labs","lab","treatment","treatments",
    "criteria","admission","disposition","dx","ddx","ed","em","er","the","a","an","and","or","to",
    "for","of","with","without","in","on","at","by","from","vs","versus","suspected","possible",
    "patient","patients","male","female","man","woman","yo","y/o","year","old"
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
def pubmed_search(user_query: str, retmax: int = 5):
    term = make_pubmed_term(user_query)

    r = requests.get(
        NCBI_ESEARCH,
        params={
            "db": "pubmed",
            "term": term,
            "retmode": "json",
            "retmax": retmax,
            "sort": "relevance",
        },
        timeout=20,
    )
    r.raise_for_status()
    pmids = r.json().get("esearchresult", {}).get("idlist", [])
    if not pmids:
        return []

    r2 = requests.get(
        NCBI_ESUMMARY,
        params={"db": "pubmed", "id": ",".join(pmids), "retmode": "json"},
        timeout=20,
    )
    r2.raise_for_status()
    data = r2.json()

    results = []
    for pmid in pmids:
        item = data.get("result", {}).get(pmid, {})
        if not item:
            continue
        results.append(
            {
                "pmid": pmid,
                "title": (item.get("title", "") or "").strip().rstrip("."),
                "journal": item.get("fulljournalname", "") or "",
                "year": ((item.get("pubdate", "") or "").split(" ")[0]) or "",
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        )
    return results

def build_context(hits):
    if not hits:
        return "No PubMed results returned."
    lines = []
    for h in hits:
        lines.append(
            f"- {h['title']} ({h['journal']}, {h['year']}). PMID {h['pmid']}. {h['url']}"
        )
    return "\n".join(lines)

def generate_answer(question: str, hits, mode: str):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    context = build_context(hits)

    system = (
        "You are an emergency medicine attending helping another ED clinician on shift. "
        "Be concise and practical. "
        "Do not ask for or include PHI. "
        "If critical details are missing, ask up to 3 clarifying questions first, then give a best-effort answer. "
        "Only cite PMIDs that appear in the provided PubMed results list. "
        "If you did not use a source, do not cite it."
    )


    if mode == "Discharge instructions (patient-friendly)":
        user = f"""Question:
{question}

PubMed results you may cite (do not invent citations beyond this list):
{context}

Write patient-friendly discharge instructions at about an 8th-grade reading level.
Include: brief explanation, what to do at home, meds if relevant (general), red flags to return, follow-up.
Keep it brief.
End with: "This is not medical advice and is for demo only."
"""
        max_tokens = 350
    else:
        user = f"""Question:
{question}

PubMed results you may cite (do not invent citations beyond this list):
{context}

Output (keep brief):
- Quick take (max 3 bullets)
- Workup (labs/imaging) (max 6 bullets)
- Treatment (max 6 bullets)
- Disposition (max 4 bullets)
- Citations: list PMIDs you used (or say "none")
"""
        max_tokens = 450

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content

# ---- Interview demo UI ----
mode = st.selectbox(
    "Mode",
    ["Workup/Treatment/Disposition", "Discharge instructions (patient-friendly)"],
    index=0,
)

samples = [
    "Chest pain, rule-out ACS with high-sensitivity troponin in the ED",
    "Suspected pulmonary embolism, when to image vs D-dimer",
    "Sepsis initial bundle in the ED, antibiotics and fluids",
    "New-onset atrial fibrillation with RVR, rate vs rhythm and disposition",
    "DKA initial management, potassium and insulin",
]
pick = st.selectbox("Question", ["Custom"] + samples, index=0)

if pick == "Custom":
    final_q = st.text_input("Custom question", placeholder="Type your ED question here")
else:
    final_q = pick

col1, col2 = st.columns(2)
with col1:
    retmax = st.slider("PubMed hits", 3, 10, 5)
with col2:
    run = st.button("Run demo")

if run:
    if not (final_q or "").strip():
        st.warning("Type a question first.")
    else:
        with st.spinner("Searching PubMed..."):
            hits = pubmed_search(final_q, retmax=retmax)

        st.subheader("Top PubMed results")
        if not hits:
            st.write("No results found. Try fewer words or more general terms.")
        else:
            for i, h in enumerate(hits, start=1):
                meta = " · ".join([x for x in [h["journal"], h["year"], f"PMID {h['pmid']}"] if x])
                st.markdown(f"**{i}. [{h['title'] or '(No title returned)'}]({h['url']})**")
                st.caption(meta)

        st.subheader("Answer (prototype)")
        with st.spinner("Generating..."):
            try:
                answer = generate_answer(final_q, hits, mode)
                st.write(answer)

                pmids = re.findall(r"\b\d{7,8}\b", answer)
                pmids = list(dict.fromkeys(pmids))
                if pmids:
                    st.caption("PMIDs cited:")
                    st.markdown(" ".join([f"[{p}](https://pubmed.ncbi.nlm.nih.gov/{p}/)" for p in pmids]))
            except KeyError:
                st.error("Missing OPENAI_API_KEY or APP_PASSWORD in Streamlit Secrets.")
            except Exception as e:
                st.error(f"OpenAI error: {e}")
