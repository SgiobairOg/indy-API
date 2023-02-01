from langchain.llms import OpenAI
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.text_splitter import CharacterTextSplitter
from dotenv import load_dotenv
import requests
import pathlib
import subprocess
import tempfile
import pickle
import time
from dotenv import load_dotenv


load_dotenv()

# Get Sample Data from Wikipedia


def get_wiki_data(title, first_paragraph_only):
    url = f"https://en.wikipedia.org/w/api.php?format=json&action=query&prop=extracts&explaintext=1&titles={title}"
    if first_paragraph_only:
        url += "&exintro=1"
    data = requests.get(url).json()
    return Document(
        page_content=list(data["query"]["pages"].values())[0]["extract"],
        metadata={"source": f"https://en.wikipedia.org/wiki/{title}"},
    )

# Yield successive n-sized
# chunks from list.


def divide_chunks(list, numberOfChunks):

    for i in range(0, len(list), numberOfChunks):
        yield list[i:i + numberOfChunks]


def source_index():

    # Set some sample sources to consult
    source_docs = [
        get_wiki_data("Unix", False),
        get_wiki_data("Microsoft_Windows", False),
        get_wiki_data("Linux", True),
        get_wiki_data("Seinfeld", True),
    ]
    source_chunks = []
    splitter = CharacterTextSplitter(
        separator=" ", chunk_size=1024, chunk_overlap=0)
    for source in source_docs:
        for chunk in splitter.split_text(source.page_content):
            source_chunks.append(
                Document(page_content=chunk, metadata=source.metadata))

    print("Chunk count")
    print(len(source_chunks))

    source_chunk_collections = list(divide_chunks(source_chunks, 20))
    with open("search_index.pickle", "wb") as file:
        for collection in source_chunk_collections:
            pickle.dump(FAISS.from_documents(
                collection, OpenAIEmbeddings()), file)
            time.sleep(60)
        # for source_chunk in source_chunks:
        #    if source_chunk.page_content:
        #        print("chunk::")
        #        print(source_chunk)
        #        pickle.dump(FAISS.from_documents(
        #        source_chunk, OpenAIEmbeddings()), file)
        #        sleep(3)


# Load Lang Chain
chain = load_qa_with_sources_chain(OpenAI(temperature=0))

# Print Answer


def print_answer(question):
    with open("search_index.pickle", "rb") as file:
        search_index = pickle.load(file)

    return chain(
        {
            "input_documents": search_index.similarity_search(question, k=4),
            "question": question,
        },
        return_only_outputs=True,
    )["output_text"]
