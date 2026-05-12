import os
import streamlit as st
import pandas as pd

from preprocessing import Preprocessor
from incidence_matrix import IncidenceMatrix
from inverted_index import InvertedIndex
from positional_index import PositionalIndex
from ranking import RankedRetrieval
from spelling_correction import SpellingCorrector
from wildcard_query import WildcardQuery
from utils import load_corpus


# PAGE CONFIG

st.set_page_config(
    page_title="Mini Search Engine",
    page_icon="🔎",
    layout="wide"
)


# CUSTOM CSS

st.markdown("""
<style>

/* Buttons */
.stButton > button {
    background-color: #8d6e63;
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.5rem 1rem;
}

.stButton > button:hover {
    background-color: #a1887f;
    color: white;
}

/* Result box */
.result-box {
    background-color: rgba(141, 110, 99, 0.15);
    padding: 15px;
    border-radius: 10px;
    border: 1px solid rgba(141, 110, 99, 0.4);
    margin-bottom: 10px;
}

/* Metrics */
[data-testid="stMetric"] {
    background-color: rgba(141, 110, 99, 0.12);
    border: 1px solid rgba(141, 110, 99, 0.35);
    padding: 15px;
    border-radius: 12px;
}

</style>
""", unsafe_allow_html=True)


# LOAD DOCUMENTS

DOCUMENT_FOLDER = "documents"

preprocessor = Preprocessor()

documents = load_corpus(
    preprocessor,
    DOCUMENT_FOLDER
)


# BUILD ALL STRUCTURES

incidence = IncidenceMatrix(documents)
incidence.build_vocabulary()
incidence.build_matrix()

inverted = InvertedIndex(documents)
inverted.build_index()

positional = PositionalIndex(documents)
positional.build_index()

ranking = RankedRetrieval(documents)
ranking.build()

corrector = SpellingCorrector(
    inverted.index.keys()
)

wildcard = WildcardQuery(
    inverted.index.keys()
)


# HELPER FUNCTIONS

def correct_tokens(tokens):
    return corrector.correct_query(tokens)


def correct_boolean_tokens(tokens):
    boolean_operators = {"and", "or", "not", "AND", "OR", "NOT"}

    corrected = []

    for token in tokens:
        if token in boolean_operators:
            corrected.append(token)
        else:
            corrected.append(
                corrector.suggest_word(token)
            )

    return corrected


def show_correction_message(original, corrected):
    if original != corrected:
        st.warning(
            f"Spelling correction applied: {' '.join(original)} → {' '.join(corrected)}"
        )


def show_documents_table(doc_ids):
    docs_df = pd.DataFrame({
        "Matching Documents": [
            f"Document {doc_id}"
            for doc_id in sorted(doc_ids)
        ]
    })

    st.dataframe(
        docs_df,
        use_container_width=True,
        hide_index=True
    )


# SIDEBAR

st.sidebar.title("Mini Search Engine")

menu = st.sidebar.radio(
    "Navigation",
    [
        "Home",
        "Preprocessing",
        "Incidence Matrix",
        "Inverted Index",
        "Wildcard Query",
        "Boolean Retrieval",
        "Phrase Query",
        "Ranked Retrieval"
    ]
)


# HOME

if menu == "Home":

    st.markdown("""
    <h1 style='text-align:center;'>
        Mini Search Engine
    </h1>
    """, unsafe_allow_html=True)

    st.markdown("""
    <p style='text-align:center; font-size:18px;'>
        Information Retrieval System using Python
    </p>
    """, unsafe_allow_html=True)

    total_docs = len(documents)

    vocabulary_size = len(incidence.vocabulary)

    total_tokens = sum(
        len(tokens)
        for tokens in documents.values()
    )

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Documents",
            total_docs
        )

    with col2:
        st.metric(
            "Vocabulary Size",
            vocabulary_size
        )

    with col3:
        st.metric(
            "Total Tokens",
            total_tokens
        )

    st.markdown("""
    ### Implemented Features

    - Document preprocessing
    - Incidence matrix
    - Inverted index
    - Boolean retrieval
    - Positional index
    - Phrase querying
    - TF-IDF ranking
    - Cosine similarity
    - Spelling correction
    - Wildcard query processing

    ---
    """)

    st.info(
        "Use the sidebar to explore different retrieval models."
    )


# PREPROCESSING

elif menu == "Preprocessing":

    st.title("Document Preprocessing")

    document_files = sorted(
    [
        file_name
        for file_name in os.listdir(DOCUMENT_FOLDER)
        if file_name.endswith(".txt")
    ],
    key=lambda name: int(
        name.replace("doc", "").replace(".txt", "")
    )
)

    file_name = st.selectbox(
        "Select a document",
        document_files
    )

    if st.button("Show Preprocessing"):

        result = preprocessor.show_example(
            DOCUMENT_FOLDER,
            file_name
        )

        if result:

            col1, col2 = st.columns(2)

            with col1:
                st.subheader("Original Text")

                st.markdown(
                    f"""
                    <div class="result-box">
                    {result["before"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with col2:
                st.subheader("Processed Tokens")

                st.markdown(
                    f"""
                    <div class="result-box">
                    {result["after"]}
                    </div>
                    """,
                    unsafe_allow_html=True
                )


# INCIDENCE MATRIX

elif menu == "Incidence Matrix":

    st.title("Incidence Matrix")

    data = {}

    for term in incidence.vocabulary:
        data[term] = incidence.matrix[term]

    df = pd.DataFrame(data).T

    st.dataframe(
        df,
        use_container_width=True
    )


# INVERTED INDEX

elif menu == "Inverted Index":

    st.title("Inverted Index")

    term = st.text_input(
        "Enter one term only",
        placeholder="example: machine"
    )

    if st.button("Search Term"):

        processed = preprocessor.preprocess(term)

        if not processed:

            st.warning("Please enter a valid term.")

        elif len(processed) > 1:

            st.warning(
                "Inverted index search accepts only one term. "
                "Use Boolean Retrieval for multiple terms."
            )

            st.subheader("Processed Query")
            st.code(processed)

        else:

            corrected = correct_tokens(processed)

            show_correction_message(
                processed,
                corrected
            )

            searched_term = corrected[0]

            st.subheader("Processed Term")
            st.code(searched_term)

            result = inverted.get_posting_list(searched_term)

            st.subheader("Posting List")

            st.write("Term:", result["term"])
            st.write("Document Frequency:", result["document_frequency"])

            if result["documents"]:

                show_documents_table(
                    result["documents"]
                )

            else:

                st.info("No documents contain this term.")


# WILDCARD QUERY

elif menu == "Wildcard Query":

    st.title("Wildcard & Partial Search")

    st.write(
        "Enter a partial word. The system automatically checks prefix, contains, and suffix matches."
    )

    partial_query = st.text_input(
        "Enter partial word",
        placeholder="examples: mach, learn, ing, art"
    )

    selected_match_types = st.multiselect(
        "Choose match types to include",
        ["Prefix", "Contains", "Suffix"],
        default=["Prefix", "Contains", "Suffix"]
    )

    if st.button("Search Partial Match"):

        if not partial_query.strip():

            st.warning("Please enter a partial word.")

        elif not selected_match_types:

            st.warning("Please select at least one match type.")

        else:

            clean_query = partial_query.lower().strip()

            match_result = wildcard.automatic_match(clean_query)

            selected_terms = set()

            if "Prefix" in selected_match_types:
                selected_terms.update(
                    match_result["prefix_terms"]
                )

            if "Contains" in selected_match_types:
                selected_terms.update(
                    match_result["contains_terms"]
                )

            if "Suffix" in selected_match_types:
                selected_terms.update(
                    match_result["suffix_terms"]
                )

            selected_terms = sorted(selected_terms)

            st.subheader("Input Query")
            st.code(match_result["query"])

            st.subheader("Selected Match Types")
            st.info(" + ".join(selected_match_types))

            col1, col2, col3 = st.columns(3)

            with col1:
                st.subheader("Prefix Matches")

                if match_result["prefix_terms"]:
                    st.dataframe(
                        pd.DataFrame({
                            "Terms": match_result["prefix_terms"]
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No prefix matches.")

            with col2:
                st.subheader("Contains Matches")

                contains_only = [
                    term
                    for term in match_result["contains_terms"]
                    if term not in match_result["prefix_terms"]
                    and term not in match_result["suffix_terms"]
                ]

                if contains_only:
                    st.dataframe(
                        pd.DataFrame({
                            "Terms": contains_only
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No contains-only matches.")

            with col3:
                st.subheader("Suffix Matches")

                suffix_only = [
                    term
                    for term in match_result["suffix_terms"]
                    if term not in match_result["prefix_terms"]
                ]

                if suffix_only:
                    st.dataframe(
                        pd.DataFrame({
                            "Terms": suffix_only
                        }),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    st.info("No suffix-only matches.")

            st.subheader("Selected Matched Terms")

            if selected_terms:

                st.dataframe(
                    pd.DataFrame({
                        "Matched Terms": selected_terms
                    }),
                    use_container_width=True,
                    hide_index=True
                )

                result_docs = set()

                for matched_term in selected_terms:
                    posting = inverted.get_posting_list(matched_term)
                    result_docs.update(
                        posting["documents"]
                    )

                st.subheader("Matching Documents")

                if result_docs:
                    show_documents_table(result_docs)
                else:
                    st.info("Matched terms exist, but no documents were found.")

            else:

                st.info(
                    "No terms matched the selected match type."
                )


# BOOLEAN RETRIEVAL

elif menu == "Boolean Retrieval":

    st.title("Boolean Retrieval")

    query = st.text_input(
        "Enter Boolean query",
        placeholder="example: machine and not database"
    )

    index_choice = st.selectbox(
        "Choose indexing method",
        [
            "Inverted Index",
            "Incidence Matrix"
        ]
    )

    if st.button("Run Boolean Query"):

        processed_query = inverted.preprocess_boolean_query(
            query,
            preprocessor
        )

        if not processed_query:

            st.warning("Please enter a valid Boolean query.")

        else:

            corrected_query = correct_boolean_tokens(
                processed_query
            )

            show_correction_message(
                processed_query,
                corrected_query
            )

            st.subheader("Processed Query")
            st.code(corrected_query)

            if index_choice == "Inverted Index":
                result = inverted.boolean_query(corrected_query)

            else:
                result = incidence.boolean_query(corrected_query)

            st.subheader("Indexing Method Used")
            st.info(index_choice)

            st.subheader("Matching Documents")

            if result:

                show_documents_table(result)

            else:

                st.info("No documents matched this Boolean query.")


# PHRASE QUERY

elif menu == "Phrase Query":

    st.title("Phrase Query")

    phrase = st.text_input(
        "Enter phrase",
        placeholder="example: machine learning"
    )

    if st.button("Search Phrase"):

        tokens = preprocessor.preprocess(
            phrase
        )

        if tokens:

            corrected_tokens = correct_tokens(tokens)

            show_correction_message(
                tokens,
                corrected_tokens
            )

            st.subheader("Processed Phrase")
            st.code(corrected_tokens)

            result = positional.phrase_query(
                corrected_tokens
            )

            st.subheader("Indexing Method Used")
            st.info("Positional Index")

            st.subheader("Matching Documents")

            if result:

                show_documents_table(result)

            else:

                st.info(
                    "No documents matched this phrase."
                )

        else:

            st.warning(
                "Phrase became empty after preprocessing."
            )


# RANKED RETRIEVAL

elif menu == "Ranked Retrieval":

    st.title("Ranked Retrieval using TF-IDF")

    query = st.text_input(
        "Enter ranked query",
        placeholder="example: artificial intelligence"
    )

    if st.button("Rank Documents"):

        tokens = preprocessor.preprocess(query)

        if tokens:

            corrected_tokens = correct_tokens(tokens)

            show_correction_message(
                tokens,
                corrected_tokens
            )

            st.subheader("Processed Query")
            st.code(corrected_tokens)

            results = ranking.search(corrected_tokens)

            results = [
                (doc_id, score)
                for doc_id, score in results
                if score > 0
            ]

            if results:

                table = pd.DataFrame(
                    results,
                    columns=[
                        "Document ID",
                        "Similarity Score"
                    ]
                )

                table.index = table.index + 1

                st.subheader("Ranked Results")

                st.dataframe(
                    table,
                    use_container_width=True
                )

            else:

                st.warning(
                    "No matching documents found."
                )

        else:

            st.warning(
                "Query became empty after preprocessing."
            )