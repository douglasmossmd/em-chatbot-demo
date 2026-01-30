import streamlit as st

st.set_page_config(page_title="EM Chatbot Demo", page_icon="🩺", layout="centered")

st.title("EM Chatbot Demo")
st.caption("Prototype for interview demo only. Not for clinical use.")

with st.expander("Disclaimer", expanded=True):
    st.write(
        "This is a prototype demonstration. It may be wrong or incomplete. "
        "Do not use for real patient care. No PHI."
    )

st.write("If you can see this page, deployment works.")

prompt = st.text_input("Type a clinical question (demo):", placeholder="e.g., workup for adult chest pain")
if st.button("Submit"):
    st.write("You asked:", prompt)
    st.info("Next step: we’ll connect PubMed + an LLM to answer with citations.")
