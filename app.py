import requests
import streamlit as st
from openai import OpenAI

st.set_page_config(page_title="EM Chatbot Demo", layout="centered")

st.title("EM Chatbot Demo")
st.caption("Prototype for interview demo only. Not for clinical use.")

with st.expander("Disclaimer", expanded=True):
    st.write(
        "This is a prototype demonstration. It may be wrong or incomplete. "
        "Do not use for real patient care. No PHI."
    )

NCBI_ESEARCH = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
NCBI_ESUMMARY = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"

@st.cache_data(ttl=3600)
def pubmed_search(query: str, retmax: int = 5):
    r = requests.get(
        NCBI_ESEARCH,
        params={
            "db": "pubmed",
            "term": query,
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
                "journal": item.get("fulljournalname", ""),
                "year": (item.get("pubdate", "") or "").split(" ")[0],
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        )
    return results

def build_context(hits):
    # Minimal context for the LLM: titles + metadata + links
    lines = []
    for h in hits:
        lines.append(
            f"- {h['title']} ({h.get('journal','')}, {h.get('year','')}). PMID {h['pmid']}. {h['url']}"
        )
    return "\n".join(lines)

def generate_em_answer(question: str, hits):
    client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

    context = build_context(hits) if hits else "No PubMed results returned."
    system = (
        "You are an emergency medicine attending helping another ED clinician on shift. "
        "This is a prototype demo, not for real clinical use. "
        "Do not request or include any patient identifiers. "
        "If the question lacks key details, ask 2-4 focused clarifying questions first. "
        "When you answer, be practical and ED-focused."
    )
    user = f"""Clinical question:
{question}

PubMed results (use as citations, do not invent citations not in this list):
{context}

Output format:
1) Quick take (2-3 bullets)
2) Labs/diagnostics to consider (bullets)
3) Treatments (bullets)
4) Disposition/admission criteria (bullets)
5) Red flags / must-not-miss (bullets)
6) Citations: list the relevant PMIDs you used
Important: If evidence is weak or mixed, say so plainly.
"""

    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content

question = st.text_input(
    "Type an ED question (demo):",
    placeholder="e.g., suspected septic shock initial workup and antibiotics",
)

col1, col2 = st.columns(2)
with col1:
    retmax = st.slider("How many PubMed hits to pull", 3, 10, 5)
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
            st.write("No results found. Try different keywords.")
        else:
            for i, h in enumerate(hits, start=1):
                title = h["title"] or "(No title returned)"
                meta = " · ".join([x for x in [h["journal"], h["year"], f"PMID {h['pmid']}"] if x])
                st.markdown(f"**{i}. [{title}]({h['url']})**")
                st.caption(meta)

        st.subheader("Draft EM-focused answer (prototype)")
        with st.spinner("Generating answer..."):
            try:
                answer = generate_em_answer(question, hits)
                st.write(answer)
            except KeyError:
                st.error("Missing OPENAI_API_KEY in Streamlit Secrets.")
            except Exception as e:
                st.error(f"Error generating answer: {e}")
