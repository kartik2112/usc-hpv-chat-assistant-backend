# Reference: https://nayakpplaban.medium.com/building-a-smart-web-rag-assistant-a-step-by-step-guide-ff5436a1349f
# https://docs.langchain.com/oss/python/langchain/rag

import os
from functools import partial
from typing import Set
import hashlib

import requests
from bs4 import BeautifulSoup, SoupStrainer

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_community.document_loaders import TextLoader, OnlinePDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.agents import create_agent
from collections import defaultdict

# Optional / commented integrations (kept for reference)
# from dotenv import load_dotenv
# from langchain_groq import ChatGroq
# from langchain_community.vectorstores import Chroma as CommunityChroma
# from langchain.chains import ConversationalRetrievalChain
# from scrapegraphai.graphs import SmartScraperGraph
# from crawl4ai import AsyncWebCrawler, CacheMode, CrawlerRunConfig


chroma_setting = Settings(anonymized_telemetry=False)


# # Set Windows event loop policy
# if sys.platform == "win32":
#     asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

# # Apply nest_asyncio to allow nested event loops
# import nest_asyncio  # Import nest_asyncio module for asynchronous operations
# nest_asyncio.apply()  # Apply nest_asyncio to resolve any issues with asyncio event loop


class HPVRAGPipeline:
	def __init__(self, openai_text_model='gpt-5-mini', persist_directory="chroma_db", max_completion_tokens=1200):
		self.persist_directory = persist_directory
		self.collection_metadata = {"hnsw:space": "cosine"}

		self.openai_text_model = openai_text_model

		self.embeddings = OpenAIEmbeddings()
		self.response_llm = ChatOpenAI(model_name=openai_text_model, max_completion_tokens=max_completion_tokens)


		self.CHROMA_CLOUD = True
		# Initialize Vector Store
		if self.CHROMA_CLOUD:
			self.vector_store =  Chroma(
				collection_name="hpv_facts_rag",
				embedding_function=self.embeddings,
				chroma_cloud_api_key=os.getenv("CHROMA_API_KEY"),
				tenant=os.getenv("CHROMA_TENANT"),
				database="Demo",
			)
		else:
			self.vector_store =  Chroma(embedding_function= self.embeddings,
				persist_directory=self.persist_directory,
				client_settings=chroma_setting,
			)

		# Initialize text splitter
		self.text_splitter = RecursiveCharacterTextSplitter(
			chunk_size=1000,
			chunk_overlap=200
		)

		self.existing_urls = set()
		self.existing_urls_to_chroma_ids = defaultdict(list)
		self.urls = [
			"https://www.acog.org/womens-health/faqs/hpv-vaccination",
			"https://www.who.int/news-room/fact-sheets/detail/human-papilloma-virus-and-cancer",
			"https://www.cdc.gov/hpv/hcp/vaccination-considerations/index.html",
			# "https://www.cancer.org/content/dam/CRC/PDF/Public/7978.00.pdf",
			# "https://www.cancer.org/content/dam/cancer-org/cancer-control/en/booklets-flyers/hpv-and-cancer-english.pdf",
			# "https://www.plannedparenthood.org/learn/stds-hiv-safer-sex/hpv",
		]

		# self.qa_chain = None
		self.agent = None
		self._setup_rag_agent()


		# model_kwargs = {"device": "cpu"}
		# encode_kwargs = {"normalize_embeddings": True}

	def _setup_rag_agent(self):
		"""Set up the RAG database"""
		# for url in self.urls:
		# 	self._crawl_webpage_and_add_to_rag(url=url)
		self.fetch_current_chromadb_entries()
		for url in self.urls:
			self._alt_crawl_webpage_and_add_to_rag(url=url)

		# Create QA chain
		# self.qa_chain = ConversationalRetrievalChain.from_llm(
		# 	llm=self.response_llm,
		# 	retriever=self.vector_store.as_retriever(search_type="similarity",
		# 												search_kwargs={"k": 5}),
		# 	return_source_documents=True
		# )

	def get_string_hash(self, text):
		return hashlib.sha256(text.encode('utf-8')).hexdigest()

	def fetch_current_chromadb_entries(self):
		entities = self.vector_store.get(include=["metadatas"])
		print(f"Entities: {len(entities['metadatas'])}")
		if len(entities['metadatas']) > 0:
			for id, metadata in zip(entities["ids"], entities["metadatas"]):
				self.existing_urls.add((metadata["source"], metadata["fulltext_hash"]))
				self.existing_urls_to_chroma_ids[metadata["source"]].append(id)
		if len(self.existing_urls) > 0:
			print(f"Found existing URLs in Chroma Cloud:")
			print(self.existing_urls)

	def _crawl_url(self, url):
		try:
			# Only keep post title, headers, and content from the full HTML.
			# bs4_strainer = SoupStrainer(class_=("post-title", "post-header", "post-content"))
			bs4_strainer = SoupStrainer(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
			loader = WebBaseLoader(
				web_paths=(url, ),
				bs_kwargs={"parse_only": bs4_strainer},
			)
			docs = loader.load()

			assert len(docs) == 1
			print(f"Total characters: {len(docs[0].page_content)}")
			return docs
		except Exception as e:
			try:
				loader = OnlinePDFLoader(url)
				return loader
			except Exception as e:
				print(f"Error loading PDF: {str(e)}")
		return loader

	def _alt_crawl_webpage_and_add_to_rag(self, url):
		
		docs = self._crawl_url(url)

		text_splitter = RecursiveCharacterTextSplitter(
			chunk_size=1000,  # chunk size (characters)
			chunk_overlap=200,  # chunk overlap (characters)
			add_start_index=True,  # track index in original document
		)
		all_splits = text_splitter.split_documents(docs)
		fulltext_hash = self.get_string_hash(docs[0].page_content)

		if (url, fulltext_hash) in self.existing_urls:
			print(f"URL {url} with hash {fulltext_hash} already exists.")
			return
		else:
			self.vector_store.delete(ids=self.existing_urls_to_chroma_ids[url])
			print(f"Adding new content from URL {url}")
			# Compute hashes for all splits
			for split in all_splits:
				split.metadata['fulltext_hash'] = fulltext_hash
				split.metadata['source'] = url
				

			print(f"Split blog post into {len(all_splits)} sub-documents.")

			document_ids = self.vector_store.add_documents(documents=all_splits)
			print(document_ids)

	# def _crawl_webpage_and_add_to_rag(self, url):
	# 	try:
	# 		"""Crawl webpage using BeautifulSoup"""
	# 		headers = {
	# 			'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'
	# 		}
	# 		response = requests.get(url, headers=headers)
	# 		response.raise_for_status()
			
	# 		soup = BeautifulSoup(response.text, 'html.parser')
			
	# 		# Remove script and style elements
	# 		for script in soup(["script", "style"]):
	# 			script.decompose()
				
	# 		# Get text content from relevant tags
	# 		text_elements = soup.find_all(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li', 'div'])
	# 		content = ' '.join([elem.get_text(strip=True) for elem in text_elements])
			
	# 		# Clean up whitespace
	# 		content = ' '.join(content.split())
			
	# 		content = content.encode('utf-8', errors='ignore').decode('utf-8')
				
	# 		# Create a temporary file with proper encoding
	# 		import tempfile
	# 		with tempfile.NamedTemporaryFile(mode='w', encoding='utf-8', delete=False, suffix='.txt') as temp_file:
	# 			temp_file.write(content)
	# 			temp_path = temp_file.name
		
	# 		# Load and process the document
	# 		docs = TextLoader(temp_path, encoding='utf-8').load()
	# 		docs = [Document(page_content=doc.page_content, metadata={"source": url}) for doc in docs]
	# 		chunks = self.text_splitter.split_documents(docs)
	# 		print(f"Length of chunks: {len(chunks)}")
	# 		print(f"First chunk: {chunks[0].metadata['source']}")
			
	# 		if os.path.exists("chroma_db"):
	# 			# Check if the URL is already in the metadata
	# 			print(f"Checking if URL {url} is already in the metadata")
	# 			try:
	# 				if url in self.existing_urls:
	# 					print(f"URL {url} already exists in the vector store")
	# 					# Load the existing vector store
	# 				else:
	# 					# Add new documents to the vector store
	# 					MAX_BATCH_SIZE = 100
	# 					for i in range(0,len(chunks),MAX_BATCH_SIZE):
	# 						#print(f"start of processing: {i}")
	# 						i_end = min(len(chunks),i+MAX_BATCH_SIZE)
	# 						#print(f"end of processing: {i_end}")
	# 						batch = chunks[i:i_end]
	# 						#
	# 						self.vector_store.add_documents(batch)
	# 						print(f"vectors for batch {i} to {i_end} stored successfully...")
	# 				self.existing_urls.add(url)
	# 			finally:
	# 				# Clean up the temporary file
	# 				try:
	# 					os.unlink(temp_path)
	# 				except:
	# 					pass
	# 	except Exception as e:
	# 		raise Exception(f"Error processing URL: {str(e)}")
		
	# def ask_question(self, messages):
	# 	"""Ask a question about the processed content"""
	# 	try:
	# 		result = self.agent.invoke(
	# 				{"messages": messages}
	# 			)
	# 		return result 
	# 	except Exception as e:
	# 		raise Exception(f"Error generating response: {str(e)}")
		
def build_rag_agent(openai_text_model='gpt-5-mini', persist_directory="chroma_db", max_completion_tokens=1200):
	rag_pipeline = HPVRAGPipeline(openai_text_model=openai_text_model, persist_directory=persist_directory, max_completion_tokens=max_completion_tokens)
	@dynamic_prompt
	def _prompt_with_context(request: ModelRequest) -> str:
		"""Inject context into state messages."""
		last_query = request.state["messages"][-1].text
		retrieved_docs = rag_pipeline.vector_store.similarity_search(last_query)

		docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
		# print(docs_content)
		# print([doc for doc in retrieved_docs])

		system_message = (
			"You are a helpful assistant. Use the following context in your response:"
			f"\n\n{docs_content}"
		)

		return system_message	
	agent = create_agent(model=rag_pipeline.openai_text_model, tools=[], middleware=[_prompt_with_context])
	return agent


def ask_rag_question(agent, messages):
	"""Ask a question about the processed content"""
	try:
		result = agent.invoke(
				{"messages": messages}
			)
		return result['messages'][-1]
	except Exception as e:
		raise Exception(f"Error generating response: {str(e)}")