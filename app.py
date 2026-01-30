import re
import requests
import streamlit as st
import xml.etree.ElementTree as ET
from openai import OpenAI

# ---------------- UI / page ----------------
st.set_page_config(page_title="ED Copilot (Prototype)", layout="centered")

st.title("ED Copilot (Prototype)")
st.caption("Prototype for interview demo only. Not for clinical use. No PHI.")

with st.expander("Disclaimer", expanded=True):
    st.write(
        "This is a prototype demonstration. It may be wrong or incomplete. "
        "Do not use for real patient care. Do not enter patient identifiers."
    )

# Passcode gate (demo only)
pw = st.text_input("Passcode", type="password")
if pw != st.secrets.get("APP_PASSWORD", ""):
    st.stop()

# ---------------- Settings controls ----------------
mode = st.selectbox(
    "Mode",
    ["Workup/Treatment/Disposition", "Discharge instructions (patient-friendly)"],
    index=0,
)

with st.expander("Retrieval settings", expanded=False):
    retmax = st.slider("PubMed results to pull", 3, 10, 5)
    abstracts_to_use = st.slider("Abstracts to include in context", 1, 5, 3)
    max_abstract_chars = st.slider("Max chars per abstract", 400, 2000, 1200, step=100)

colA, colB = st.columns(2)
with colA:
    if st.button("Clear chat"):
        st.session_state["messages"] = []
        st.session_state["last_evidence"] = None
        st.rerun()
with colB:
    st.caption("Chat persists during this session only.")

# ---------------- PubMed helpers ----------------
NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
NCBI_EFETCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"

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
    pmids = r.json().get("esearchresult", {}).get("idlist", [])
    return pmids

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
def pubmed_fetch_abstracts(pmids):
    """
    Fetch abstracts via EFetch XML. Many entries will have no abstract; we handle that.
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
        timeout=30,
    )
    r.raise_for_status()
    xml_text = r.text

    root = ET.fromstring(xml_text)
    abstracts = {}

    for article in root.findall(".//PubmedArticle"):
        pmid_el = article.find(".//PMID")
        if pmid_el is None:
            continue
        pmid = pmid_el.text.strip()

        abs_nodes = article.findall(".//Abstract/AbstractText")
        if not abs_nodes:
            abstracts[pmid] = ""
            continue

        parts = []
        for n in abs_nodes:
            label = n.attrib.get("Label")
            text = "".join(n.itertext()).strip()
            if not text:
                continue
            if label:
                parts.append(f"{label}: {text}")
            else:
                parts.append(text)
        abstracts[pmid] = "\n".join(parts).strip()

    # Ensure every requested PMID has a key
    for p in pmids:
        abstracts.setdefault(p, "")

    return abstracts

def build_evidence_block(summaries, abstracts_map, abstracts_to_use=3, max_chars=1200):
    """
    Build a compact evidence context: title + PMID + abstract (trimmed).
    """
    use = summaries[:abstracts_to_use]
    lines = []
    allowed_pmids = []
    for h in use:
        pmid = h["pmid"]
        allowed_pmids.append(pmid)
        abstract = (abstracts_map.get(pmid) or "").strip()
        if abstract:
            if len(abstract) > max_chars:
                abstract = abstract[:max_chars].rstrip() + "…"
        else:
            abstract = "(No abstract available via PubMed.)"

        lines.append(
            f"PMID {pmid}\nTitle: {h['title']}\nJournal/Year: {h['journal']} ({h['year']})\nAbstract:\n{abstract}\nLink: {h['url']}"
        )
    return "\n\n---\n\n".join(lines), allowed_pmids

# ---------------- LLM answer ----------------
def generate_answer(messages, question: str, evidence_text: str, allowed_pmids, mode: str):
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

    if mode == "Discharge instructions (patient-friendly)":
        user = f"""User question:
{question}

Allowed PMIDs: {allowed_str}

Evidence (abstracts):
{evidence_text}

Write patient-friendly discharge instructions at about an 8th-grade reading level.
Include: brief explanation, what to do at home, meds if relevant (general), red flags to return, follow-up.
Keep it brief.
End with: "This is not medical advice and is for demo only."
Then end with: Citations: <1–3 PMIDs from Allowed PMIDs> (required if Allowed PMIDs is not 'none').
"""
        max_tokens = 350
    else:
        user = f"""User question:
{question}

Allowed PMIDs: {allowed_str}

Evidence (abstracts):
{evidence_text}

Output (keep brief):
- Quick take (max 3 bullets)
- Workup (labs/imaging) (max 6 bullets)
- Treatment (max 6 bullets)
- Disposition (max 4 bullets)
- Citations: <1–3 PMIDs from Allowed PMIDs> (required if Allowed PMIDs is not 'none')
"""
        max_tokens = 450

    # Include prior conversation turns, but keep it light: just the message contents.
    # Streamlit stores messages as {"role": "user"/"assistant", "content": "..."}.
    convo = [{"role": "system", "content": system}] + messages + [{"role": "user", "content": user}]

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=convo,
        temperature=0.2,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content

# ---------------- Chat state ----------------
if "messages" not in st.session_state:
    st.session_state["messages"] = []
if "last_evidence" not in st.session_state:
    st.session_state["last_evidence"] = None

# Render chat history
for m in st.session_state["messages"]:
    with st.chat_message(m["role"]):
        st.write(m["content"])

# Chat input
prompt = st.chat_input("Ask an ED question (no PHI).")
if prompt:
    st.session_state["messages"].append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching PubMed and fetching abstracts..."):
            pmids = pubmed_search(prompt, retmax=retmax)
            summaries = pubmed_summaries(pmids)
            abstracts_map = pubmed_fetch_abstracts(pmids)

            evidence_text, allowed_pmids = build_evidence_block(
                summaries,
                abstracts_map,
                abstracts_to_use=abstracts_to_use,
                max_chars=max_abstract_chars,
            )

            # Store last evidence for transparency/debug
            st.session_state["last_evidence"] = {
                "summaries": summaries,
                "allowed_pmids": allowed_pmids,
                "evidence_text": evidence_text,
            }

        with st.spinner("Generating answer..."):
            try:
                answer = generate_answer(
                    messages=st.session_state["messages"][-6:-1],  # last few turns only
                    question=prompt,
                    evidence_text=evidence_text,
                    allowed_pmids=allowed_pmids,
                    mode=mode,
                )
                st.write(answer)

                # PMID quick links from answer
                pmids_in_answer = re.findall(r"\b\d{7,8}\b", answer)
                pmids_in_answer = list(dict.fromkeys(pmids_in_answer))
                if pmids_in_answer:
                    st.caption("PMIDs cited:")
                    st.markdown(" ".join([f"[{p}](https://pubmed.ncbi.nlm.nih.gov/{p}/)" for p in pmids_in_answer]))

                # Optional: show what abstracts were fed in
                with st.expander("Evidence used (abstracts sent to model)", expanded=False):
                    st.text(evidence_text if evidence_text else "(none)")

                st.session_state["messages"].append({"role": "assistant", "content": answer})

            except Exception as e:
                st.error(f"Error: {e}")
