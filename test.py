import os
import tempfile
from flask import Flask, render_template, request, redirect, url_for
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import CharacterTextSplitter
from langchain_openai import OpenAI
from langchain_openai import OpenAIEmbeddings
from langchain.vectorstores import Chroma

app = Flask(__name__)

# Set OpenAI API key
os.environ["OPENAI_API_KEY"] = "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"

convos = []  # store all conversation objects in a list


def qa(file, query, chain_type, k):
    # load document
    loader = PyPDFLoader(file)
    documents = loader.load()
    # split the documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    # select which embeddings we want to use
    embeddings = OpenAIEmbeddings()
    # create the vectorestore to use as the index
    db = Chroma.from_documents(texts, embeddings)
    # expose this index in a retriever interface
    retriever = db.as_retriever(search_type="similarity", search_kwargs={"k": k})
    # create a chain to answer questions
    qa = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type=chain_type,
        retriever=retriever,
        return_source_documents=True,
    )
    result = qa({"query": query})
    return result


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        # Get the uploaded PDF file and other form data
        file = request.files["file"]
        query = request.form["query"]
        chain_type = request.form["chain_type"]
        k = int(request.form["k"])

        # Save the PDF file to a temporary location
        _, temp_path = tempfile.mkstemp(suffix=".pdf")
        file.save(temp_path)

        # Run the question-answering script
        result = qa(file=temp_path, query=query, chain_type=chain_type, k=k)

        # Add the conversation to the list
        convos.append({
            "prompt": query,
            "result": result["result"],
            "source_documents": [doc.page_content for doc in result["source_documents"]]
        })

        # Remove the temporary PDF file
        os.remove(temp_path)

    return render_template("index.html", convos=convos)


if __name__ == "__main__":
    app.run(debug=True)
