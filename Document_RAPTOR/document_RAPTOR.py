import os
import json
import pathlib
import requests
import numpy as np
import pandas as pd
import networkx as nx
import streamlit as st
from pyvis.network import Network
from sklearn.cluster import KMeans
from langchain_community.llms import Ollama
from sklearn.mixture import GaussianMixture
import streamlit.components.v1 as components
from langchain.prompts import ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain.memory import ConversationBufferMemory
from sklearn.metrics.pairwise import cosine_similarity
from langchain.chains import ConversationalRetrievalChain
from langchain_core.output_parsers import StrOutputParser
from langchain_experimental.text_splitter import SemanticChunker
from langchain_community.embeddings import HuggingFaceInstructEmbeddings
from langchain_community.document_loaders import (
    CSVLoader,
    PyMuPDFLoader,
    TextLoader,
    UnstructuredPowerPointLoader,
    Docx2txtLoader,
    UnstructuredExcelLoader,
)


FILE_LOADERS = {
    "csv": CSVLoader,
    "docx": Docx2txtLoader,
    "pdf": PyMuPDFLoader,
    "pptx": UnstructuredPowerPointLoader,
    "txt": TextLoader,
    "xlsx": UnstructuredExcelLoader,
}

ACCEPTED_FILE_TYPES = list(FILE_LOADERS)

# Message classes
class Message:
    def __init__(self, content):
        self.content = content

class HumanMessage(Message):
    """Represents a message from the user."""
    pass

class AIMessage(Message):
    """Represents a message from the AI."""
    pass

# Class for chatting with pcap data
class ChatWithDocuments:
    def __init__(self, file_path, file_type):
        self.file_path = file_path
        self.file_type = file_type
        self.load_document()
        with st.spinner("Downloading Instructor XL Embeddings Model locally....please be patient"):
            self.embedding_model=HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-large", model_kwargs={"device": "cuda"})
        self.llm = Ollama(model=st.session_state['selected_model'], base_url="http://ollama:11434")
        self.document_cluster_mapping = {}
        self.conversation_history = []
        self.split_into_chunks()        
        self.root_node = None
        self.create_leaf_nodes()
        self.build_tree()  # This will build the tree
        self.store_in_chroma()
        self.setup_conversation_memory()
        self.setup_conversation_retrieval_chain()

    def load_document(self):
        self.loader = FILE_LOADERS[self.file_type](file_path=self.file_path)
        self.pages = self.loader.load_and_split()

    def split_into_chunks(self):
        self.text_splitter = SemanticChunker(self.embedding_model)
        self.docs = self.text_splitter.split_documents(self.pages)

    def create_leaf_nodes(self):
        self.leaf_nodes = [Node(text=doc.page_content) for doc in self.docs]
        self.embed_leaf_nodes()
        st.write(f"Leaf nodes created. Total count: {len(self.leaf_nodes)}")

    def embed_leaf_nodes(self):
        for leaf_node in self.leaf_nodes:
            try:
                embedding = self.embedding_model.embed_query(leaf_node.text)
                if embedding is not None and not np.isnan(embedding).any():
                    leaf_node.embedding = embedding
                else:
                    # Handle the case where embedding is nan or None
                    st.write(f"Invalid embedding generated for leaf node with text: {leaf_node.text}")
            except Exception as e:
                st.write(f"Error embedding leaf node: {e}")

    def determine_initial_clusters(self, nodes):
        # This is a simple heuristic: take the square root of the number of nodes,
        # capped at a minimum of 2 and a maximum that makes sense for your application.
        return max(2, int(len(nodes)**0.5))

    def cluster_nodes(self, nodes, n_clusters=2):
        st.write(f"Clustering {len(nodes)} nodes into {n_clusters} clusters...")
        embeddings = np.array([node.embedding for node in nodes if node.embedding is not None])
        st.write("Embeddings as of Cluster Nodes:", embeddings)
        # Check if embeddings is empty
        if embeddings.size == 0:
            # Handle the case where there are no embeddings to cluster
            # This could be logging a warning and returning the nodes as a single cluster or any other logic you see fit
            st.write("Warning: No valid embeddings found for clustering. Returning nodes as a single cluster.")
            return [nodes]  # Return all nodes as a single cluster to avoid crashing

        # Check if embeddings is not empty but a 1D array, reshape it
        if embeddings.ndim == 1:
            embeddings = embeddings.reshape(-1, 1)

        # Proceed with KMeans clustering
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)
        try:
            kmeans.fit(embeddings)
        except ValueError as e:
            # Handle possible errors during fit, such as an invalid n_clusters value
            st.write(f"Error during clustering: {e}")
            return [nodes]  # Fallback: return all nodes as a single cluster

        # Initialize clusters based on labels
        clusters = [[] for _ in range(n_clusters)]
        for node, label in zip(nodes, kmeans.labels_):
            clusters[label].append(node)

        st.write(f"Clusters formed: {len(clusters)}")
        return clusters

    def summarize_cluster(self, cluster):
        # Combine texts from all nodes in the cluster
        combined_text = " ".join([node.text for node in cluster])

        # Generate a summary for the combined text using the LLM
        summary = self.invoke_summary_generation(combined_text)
        summary_embedding = self.embedding_model.embed_query(summary)
        return summary

    def recursive_cluster_summarize(self, nodes, depth=0, n_clusters=None):
        st.write(f"Clustering and summarizing at depth {depth} with {len(nodes)} nodes...")
        if len(nodes) <= 1:
            self.root_node = nodes[0]  # If only one node, it is the root
            return nodes[0]  # Base case: only one node, it is the root

        if n_clusters is None:
            n_clusters = self.determine_initial_clusters(nodes)

        clusters = self.cluster_nodes(nodes, n_clusters=n_clusters)
        parent_nodes = []
        for cluster in clusters:
            cluster_summary = self.summarize_cluster(cluster)
            parent_nodes.append(Node(text=cluster_summary, children=cluster))

        # When we make the recursive call, we don't pass n_clusters, assuming
        # the function will determine the appropriate number for the next level.
        st.write(f"Clustering and summarization complete at depth {depth}.")
        return self.recursive_cluster_summarize(parent_nodes, depth + 1)

    def build_tree(self):
        # Determine the number of clusters to start with
        # It could be a function of the number of leaf nodes, or a fixed number
        n_clusters = self.determine_initial_clusters(self.leaf_nodes)
        # Begin recursive clustering and summarization
        self.recursive_cluster_summarize(self.leaf_nodes, n_clusters=n_clusters)
        root_summary_embedding = self.embedding_model.embed_query(self.root_node.text)

    def store_in_chroma(self):
        st.write("Storing in Chroma")
        all_texts = []
        all_summaries = []

        def traverse_and_collect(node):
            # Base case: if it's a leaf node, collect its text.
            if node.is_leaf():
                all_texts.append(node.text)
            else:
                # For non-leaf nodes, collect the summary.
                all_summaries.append(node.text)
                # Recursively process children.
                for child in node.children:
                    traverse_and_collect(child)

        # Start the traversal from the root node.
        traverse_and_collect(self.root_node)

        # Combine leaf texts and summaries.
        combined_texts = all_texts + all_summaries
        # Now, use all_texts to build the vectorstore with Chroma
        
        self.vectordb = Chroma.from_texts(texts=combined_texts, embedding=self.embedding_model)

    def setup_conversation_memory(self):
        self.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

    def setup_conversation_retrieval_chain(self):
        self.qa = ConversationalRetrievalChain.from_llm(self.llm, self.vectordb.as_retriever(search_kwargs={"k": 10}), memory=self.memory)

    def get_optimal_clusters(self, embeddings, max_clusters=50, random_state=1234):
        if embeddings is None or len(embeddings) == 0:
            st.write("No reduced embeddings available for clustering.")
            return 1  # Return a default or sensible value for your application
        max_clusters = min(max_clusters, len(embeddings))
        bics = [
            GaussianMixture(n_components=n, random_state=random_state)
            .fit(embeddings)
            .bic(embeddings)
            for n in range(1, max_clusters + 1)
        ]
        return bics.index(min(bics)) + 1

    def chat(self, question):
        st.write("gnerating response")
        response = self.qa.invoke(question)
        self.conversation_history.append({"You": question, "AI": response}) 
        return response

    def update_conversation_history(self, question, response, relevant_nodes):
        # Store the question, response, and the paths through the tree that led to the response
        self.conversation_history.append({
            "You": question,
            "AI": response,
            "Nodes": relevant_nodes
        })

    def identify_relevant_clusters(self, documents):
        """
        Identify clusters relevant to the given list of documents.

        Parameters:
            documents (list): A list of documents for which to identify relevant clusters.

        Returns:
            set: A set of unique cluster IDs relevant to the given documents.
        """
        cluster_ids = set()
        for i, doc in enumerate(documents):
            cluster_id = self.document_cluster_mapping.get(i)
            if cluster_id is not None:
                cluster_ids.add(cluster_id)
        return cluster_ids

    def identify_relevant_clusters_based_on_query(self, question, node=None, threshold=0.5):
        # Start from the root if no node is specified
        if node is None:
            node = self.root_node

        relevant_clusters = []

        # Base case: if it's a leaf node, return an empty list (as we're looking for clusters, not leaves)
        if node.is_leaf():
            return relevant_clusters

        # Calculate the similarity between the question and the cluster's summary
        question_embedding = self.embedding_model.embed_query(question)
        cluster_summary_embedding = self.embedding_model.embed_query(node.text)  # Assuming node.text holds the cluster summary
        similarity_score = self.calculate_similarity(question_embedding, cluster_summary_embedding)

        # If the similarity score is above the threshold, this cluster is relevant
        if similarity_score > threshold:
            relevant_clusters.append(node)
        else:
            # Recursively check this node's children
            for child in node.children:
                relevant_clusters.extend(self.identify_relevant_clusters_based_on_query(question, child, threshold))

        return relevant_clusters

    def get_document_ids_from_clusters(self, clusters):
        # Assuming each cluster (or node) has a list of document IDs associated with it
        document_ids = []
        for cluster in clusters:
            # Assuming `cluster.documents` holds the IDs of documents in this cluster
            document_ids.extend(cluster.documents)
        return document_ids

    def prepare_clustered_data(self, clusters=None):
        """
        Prepare data for summarization by clustering.
        Can now filter by specific clusters if provided, enhancing dynamic use.
        """
        # Initialize a list for filtered documents
        filtered_docs = []

        # If specific clusters are provided, filter documents belonging to those clusters
        if clusters is not None:
            for i, doc in enumerate(self.docs):
                # Retrieve the cluster ID from the mapping using the document's index
                cluster_id = self.document_cluster_mapping.get(i)
                # If the document's cluster ID is in the specified clusters, include the document
                if cluster_id in clusters:
                    filtered_docs.append(doc)
        else:
            filtered_docs = self.docs

        # Construct a DataFrame from the filtered documents
        # You'll need to adjust how you access the text and cluster_id based on your Document object structure
        df = pd.DataFrame({
            "Text": [doc.page_content for doc in filtered_docs],  # Adjust based on your document's structure
            "Cluster": [self.document_cluster_mapping[i] for i in range(len(filtered_docs))]  # Access cluster IDs from the mapping
        })
        self.clustered_texts = self.format_cluster_texts(df)

    def format_cluster_texts(self, df):
        """Organize texts by their clusters."""
        clustered_texts = {}
        for cluster in df["Cluster"].unique():
            cluster_texts = df[df["Cluster"] == cluster]["Text"].tolist()
            clustered_texts[cluster] = " --- ".join(cluster_texts)
        return clustered_texts

    def generate_summaries(self, clusters=None):
        self.prepare_clustered_data(clusters)
        summaries = {}
        for cluster, text in self.clustered_texts.items():
            summary = self.invoke_summary_generation(text)
            summaries[cluster] = summary
        return summaries

    def invoke_summary_generation(self, text):
        """Invoke the language model to generate a summary for the given text."""
        template = "You are an assistant to create a detailed summary of the text input provided.\nText:\n{text}"
        prompt = ChatPromptTemplate.from_template(template)
        chain = prompt | self.llm.invoke | StrOutputParser()

        summary = chain.invoke({"text": text})
        st.write("Generated Summary:", summary)
        return summary

    def display_summaries(self, node=None, level=0, summaries=None):
        if summaries is None:
            summaries = []
        if node is None:
            node = self.root_node
        if node is not None:
            # Assuming that node.text contains the summary
            summaries.append((' ' * (level * 2)) + node.text)
            for child in node.children:
                self.display_summaries(child, level + 1, summaries)
        return summaries

    def create_tree_graph(self):
        def add_nodes_recursively(graph, node, parent_name=None):
            # Create a unique name for the node
            node_name = f"{node.text[:10]}..." if node.text else "Root"
            graph.add_node(node_name)

            # Link the node to its parent if it exists
            if parent_name:
                graph.add_edge(parent_name, node_name)

            for child in node.children:
                add_nodes_recursively(graph, child, node_name)

        G = nx.Graph()
        add_nodes_recursively(G, self.root_node)
        return G

# Class for leaf nodes
class Node:
    def __init__(self, text, children=None, embedding=None):
        self.text = text  # The original text or the summary text of the node
        self.children = children if children is not None else []  # Child nodes
        self.embedding = embedding  # Embedding of the node's text
        self.cluster_label = None  # The cluster this node belongs to
    
    def is_leaf(self):
        # A leaf node has no children
        return len(self.children) == 0
    
def get_ollama_models(base_url):
    try:       
        response = requests.get(f"{base_url}api/tags")  # Corrected endpoint
        response.raise_for_status()
        models_data = response.json()
        
        # Extract just the model names for the dropdown
        models = [model['name'] for model in models_data.get('models', [])]
        return models
    except requests.exceptions.RequestException as e:
        st.error(f"Failed to get models from Ollama: {e}")
        return []

def upload_and_handle_file():
    """
    Present the file upload context. After upload, determine the file extension
    and save the file. Set session state for the file path and type of file
    for use in the chat interface.

    :return: None
    """
    st.title("Document RAPTOR - Automated AI Summarization and Analysis and Chat with your documents!")
    uploaded_file = st.file_uploader(
        label=(
            f"Choose a {', '.join(ACCEPTED_FILE_TYPES[:-1]).upper()}, or "
            f"{ACCEPTED_FILE_TYPES[-1].upper()} file"
        ),
        type=ACCEPTED_FILE_TYPES
    )
    if uploaded_file:
        # Determine the file type and set accordingly
        file_type = pathlib.Path(uploaded_file.name).suffix
        file_type = file_type.replace(".", "")

        if file_type:  # Will be an empty string if no extension
            csv_pdf_txt_path = os.path.join("temp", uploaded_file.name)
            if not os.path.exists("temp"):
                os.makedirs("temp")
            with open(csv_pdf_txt_path, "wb") as f:
                f.write(uploaded_file.getvalue())
            st.session_state["file_path"] = csv_pdf_txt_path
            st.session_state["file_type"] = file_type  # Store the file type in session state
            st.success(f"{file_type.upper()} file uploaded successfully.")
            # Fetch and display the models in a select box
            models = get_ollama_models("http://ollama:11434/")  # Make sure to use the correct base URL
            if models:
                selected_model = st.selectbox("Select Model", models)
                st.session_state['selected_model'] = selected_model
            
                st.button(
                    "Proceed to Chat",
                    on_click=lambda: st.session_state.update({"page": 2})
                )
        else:
            st.error(
                f"Unsupported file type. Please upload a "
                f"{', '.join(ACCEPTED_FILE_TYPES[:-1]).upper()}, or "
                f"{ACCEPTED_FILE_TYPES[-1].upper()} file."
            )


def chat_interface():
    """
    Main chat interface - invoked after a file has been uploaded.

    :return: None
    """
    st.title("Document RAPTOR - Automated AI Summarization and Analysis and Chat with your documents!")
    file_path = st.session_state.get("file_path")
    file_type = st.session_state.get("file_type")
    if not file_path or not os.path.exists(file_path):
        st.error("File missing. Please go back and upload a file.")
        return

    if "chat_instance" not in st.session_state:
        st.session_state["chat_instance"] = ChatWithDocuments(
            file_path=file_path,
            file_type=file_type
        )

    user_input = st.text_input("Ask a question about the document data:")
    if user_input and st.button("Send"):
        with st.spinner("Thinking..."):
            top_result = st.session_state["chat_instance"].chat(user_input)

            # New: Display the automated AI analysis (summaries) before asking a question
            if st.session_state['chat_instance'].root_node:
                st.subheader("Automated AI Analysis")
                summaries = st.session_state['chat_instance'].display_summaries()
                for summary in summaries:
                    st.markdown(f"* {summary}")

            # Display the top result's answer as markdown for better readability
            if top_result:
                st.markdown("**Top Answer:**")
                st.markdown(f"> {top_result['answer']}")
            else:
                st.write("No top result available.")

            st.subheader("RAPTOR Tree Visualization")
            G = st.session_state['chat_instance'].create_tree_graph()
            nt = Network('800px', '800px', notebook=True, directed=True)
            nt.from_nx(G)
            nt.hrepulsion(node_distance=120, central_gravity=0.0, spring_length=100, spring_strength=0.01, damping=0.09)
            nt.set_options("""
                var options = {
                  "nodes": {
                    "scaling": {
                      "label": true
                    }
                  },
                  "edges": {
                    "color": {
                      "inherit": true
                    },
                    "smooth": false
                  },
                  "interaction": {
                    "hover": true,
                    "tooltipDelay": 300
                  },
                  "physics": {
                    "hierarchicalRepulsion": {
                      "centralGravity": 0.0
                    },
                    "solver": "hierarchicalRepulsion"
                  }
                }
                """)
            nt.show('tree.html')
            HtmlFile = open('tree.html', 'r', encoding='utf-8')
            source_code = HtmlFile.read() 
            components.html(source_code, height=800, width=1000)

            # Display chat history
            st.markdown("**Chat History:**")
            for message in st.session_state["chat_instance"].conversation_history:
                prefix = "*You:* " if isinstance(message, HumanMessage) else "*AI:* "
                st.markdown(f"{prefix}{message}")

if __name__ == "__main__":
    if "page" not in st.session_state:
        st.session_state["page"] = 1

    if st.session_state["page"] == 1:
        upload_and_handle_file()
    elif st.session_state["page"] == 2:
        chat_interface()