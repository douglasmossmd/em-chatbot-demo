import re
import requests
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="EM Chatbot Demo", layout="centered")

st.title("EM Chatbot Demo")
st.caption("Prototype for interview demo only. Not for clinical use. No PHI.")

with st.expander("Disclaimer", expanded=True):
    st.write(
        "This is a prototype demonstration. It may be wrong or incomplete. "
        "Do not use for real patient care. Do not enter patient identifiers."
    )

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
    """
    PubMed can over-AND long queries. This converts long free-text into a smaller AND query
    using 3–5 high-signal keywords, while also adding a quoted phrase fallback.
    """
    q = (q or "").strip()
    if not q:
        return q

    raw = q.lower()
    # expand common acronyms
    for k, v in SYNONYMS.items():
        raw = re.sub(rf"\b{k}\b", v, raw)

    # tokenise
    cleaned = re.sub(r"[^a-z0-9\s]", " ", raw)
    tokens = [t for t in cleaned.split() if t and t not in STOPWORDS]

    # keep unique order
    seen = set()
    uniq = []
    for t in tokens:
        if t not in seen:
            uniq.append(t)
            seen.add(t)

    # take top few tokens to avoid brittle over-AND
    key = uniq[:5] if uniq else []

    # Build term:
    # - quoted phrase (broad fallback)
    # - AND of key tokens in Title/Abstract (less brittle)
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


def generate_em_answer(question: str, hits):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])
    context = build_context(hits)

    system = (
        "You are an emergency medicine attending helping another ED clinician on shift. "
        "Give concise, practical guidance. "
        "Do not ask for or include PHI. "
        "If critical details are missing, ask up to 3 clarifying questions first, then give a best-effort answer."
    )

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

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.2,
        max_tokens=450,  # keeps it short/cheap
    )
    return resp.choices[0].message.content


question = st.text_input(
    "Type an ED question (demo):",
    placeholder="e.g., adult DKA initial management labs insulin potassium",
)

col1, col2 = st.columns(2)
with col1:
    retmax = st.slider("PubMed hits", 3, 10, 5)
with col2:
    run = st.button("Generate answer")

if run:
    if not question.strip():
        st.warning("Type a question first.")
    else:
        with st.spinner("Searching PubMed..."):
            hits = pubmed_search(question, retmax=retmax)

        st.subheader("Top PubMed results")
        if not hits:
            st.write("No results found. Try fewer words or more general terms.")
        else:
            for i, h in enumerate(hits, start=1):
                meta = " · ".join([x for x in [h["journal"], h["year"], f"PMID {h['pmid']}"] if x])
                st.markdown(f"**{i}. [{h['title'] or '(No title returned)'}]({h['url']})**")
                st.caption(meta)

        st.subheader("Draft EM-focused answer (prototype)")
        with st.spinner("Generating..."):
            try:
                st.write(generate_em_answer(question, hits))
            except KeyError:
                st.error("Missing OPENAI_API_KEY in Streamlit Secrets.")
            except Exception as e:
                st.error(f"OpenAI error: {e}")

