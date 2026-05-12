def load_corpus(preprocessor, folder_path):
    """
    Loads and preprocesses all documents once.
    Returns:
        dict: {doc_id: tokens}
    """
    return preprocessor.preprocess_documents(folder_path)