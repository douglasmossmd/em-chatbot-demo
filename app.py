import requests
import streamlit as st

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
    # 1) Find top PMIDs
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

    # 2) Get title/journal/year for those PMIDs
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
                "title": item.get("title", "").strip().rstrip("."),
                "journal": item.get("fulljournalname", ""),
                "year": (item.get("pubdate", "") or "").split(" ")[0],
                "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
            }
        )
    return results


question = st.text_input(
    "Type a clinical question (demo):",
    placeholder="e.g., adult chest pain workup troponin delta high sensitivity",
)

if st.button("Search PubMed"):
    if not question.strip():
        st.warning("Type a question first.")
    else:
        with st.spinner("Searching PubMed..."):
            hits = pubmed_search(question, retmax=5)

        st.subheader("Top PubMed results")
        if not hits:
            st.write("No results found. Try different keywords.")
        else:
            for i, h in enumerate(hits, start=1):
                title = h["title"] or "(No title returned)"
                meta = " · ".join([x for x in [h["journal"], h["year"], f"PMID {h['pmid']}"] if x])
                st.markdown(f"**{i}. [{title}]({h['url']})**")
                st.caption(meta)

st.divider()
st.write("Next step: use these citations as context for an EM-styled answer.")
