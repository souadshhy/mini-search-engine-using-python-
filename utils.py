def load_corpus(preprocessor, folder_path):
    """
       load_corpus -> dict: {doc_id: tokens}
    """
    return preprocessor.preprocess_documents(folder_path)