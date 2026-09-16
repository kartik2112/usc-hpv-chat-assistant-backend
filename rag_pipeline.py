# Reference: https://nayakpplaban.medium.com/building-a-smart-web-rag-assistant-a-step-by-step-guide-ff5436a1349f
# https://docs.langchain.com/oss/python/langchain/rag
# https://docs.langchain.com/oss/python/integrations/vectorstores/chroma#other-search-methods
# https://stackoverflow.com/questions/76870837/how-to-delete-documents-in-langchain-vectorstore

import os
import re
import tempfile
from typing import Set
import hashlib
from datetime import datetime, timezone

import requests
from bs4 import SoupStrainer

from chromadb.config import Settings
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader, WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.agents.middleware import dynamic_prompt, ModelRequest
from langchain.agents import create_agent
from collections import defaultdict



chroma_setting = Settings(anonymized_telemetry=False)
USE_CHROMA_CLOUD = True


# Shared system-prompt rules for the HPV assistant — the single source of truth
# (the frontend sends no system message). Audience-specific lines (general vs
# post-partum, see variants.py) are inserted by build_system_prompt(), and the
# retrieved RAG context is appended after the result by each caller.
BASE_SYSTEM_RULES = (
	"You are a helpful medical assistant specializing ONLY in HPV-related information. "
	"It is extremely important to follow these instructions:\n"
	"* Provide accurate answers based on the latest medical guidelines and research.\n"
	"* Explain things as you would to patients outside the medical field who may not be "
	"highly educated. Avoid complex medical jargon unless absolutely necessary, and educate "
	"in a clear, friendly manner. Do not make answers sound scary or verbose.\n"
	"* Keep responses brief and to the point — TRY TO KEEP IT WITHIN **75 WORDS**. "
	"BREVITY AND TERSENESS ARE SUPER IMPORTANT to avoid overwhelming patients.\n"
	"* If you don't know the answer, simply say you don't know.\n"
	"* If the question is not related to HPV, politely decline to answer.\n"
	"* Refrain from asking additional questions unless clarification is needed.\n"
)


def build_system_prompt(audience_instructions="", with_context_header=True):
	"""Base rules + audience-specific bullet lines (+ the RAG context header,
	which callers that append no retrieved context should leave out)."""
	prompt = f"{BASE_SYSTEM_RULES}{audience_instructions}"
	return prompt + "Use the following context in your response:" if with_context_header else prompt


# Publishers behind bot protection answer a crawl with a short "Access Denied"
# page (~120 characters) and HTTP 200, which would otherwise be embedded as if
# it were the source. Anything shorter than this is treated as a failed fetch.
MIN_EXTRACTED_CHARS = 500


def has_usable_content(docs, minimum=MIN_EXTRACTED_CHARS):
	"""True when a loader returned real text rather than nothing or a block page."""
	return bool(docs) and sum(len(d.page_content.strip()) for d in docs) >= minimum


# Sent by both loaders below. Some publishers reject the default
# `python-requests/x.y` agent outright, so the PDF download has to identify
# itself the same way the web crawl does.
CRAWL_HEADERS = {
	'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
	              '(KHTML, like Gecko) Chrome/58.0.3029.110 Safari/537.36'
}
CRAWL_TIMEOUT = 60

# europepmc.org sits behind a bot challenge that answers every crawl with HTTP
# 403, but the same articles are served as plain JATS XML by the Europe PMC REST
# API, which is meant to be read by machines and is not challenged. Rewrite
# article links to that endpoint and try it first.
_EUROPEPMC_ARTICLE = re.compile(r'^https?://(?:www\.)?europepmc\.org/(?:articles|article/pmc)/(PMC\d+)', re.I)
EUROPEPMC_REST = 'https://www.ebi.ac.uk/europepmc/webservices/rest/{pmcid}/fullTextXML'


def crawl_candidates(url):
	"""The URLs to try for a source, best first.

	Only articles Europe PMC holds under an open licence have full text at the
	REST endpoint (it 404s for the rest), so the configured URL stays in the
	list as a fallback.
	"""
	match = _EUROPEPMC_ARTICLE.match(url)
	if match:
		return [EUROPEPMC_REST.format(pmcid=match.group(1).upper()), url]
	return [url]


class HPVRAGPipeline:
	"""One vector store + its source list (one per variant — see rag_sources.json).

	`rag_sources` is a rag_sources.RagSources: the Chroma target and the web page /
	PDF URLs to index. After construction, `index_report` holds the outcome of
	this build for each URL (see _record) and `built_at` its UTC time.
	"""
	def __init__(self, rag_sources, openai_text_model='gpt-5.5', persist_directory="chroma_db", max_completion_tokens=1200):
		self.rag_sources = rag_sources
		# Local (non-cloud) mode keeps one directory per collection.
		self.persist_directory = os.path.join(persist_directory, rag_sources.chroma.collection)

		self.openai_text_model = openai_text_model

		self.embeddings = OpenAIEmbeddings()
		self.response_llm = ChatOpenAI(
			model_name=openai_text_model,
			max_completion_tokens=max_completion_tokens,
			model_kwargs={"reasoning_effort": "low"},
		)


		
		# Initialize Vector Store
		target = rag_sources.chroma
		if USE_CHROMA_CLOUD:
			self.vector_store =  Chroma(
				collection_name=target.collection,
				embedding_function=self.embeddings,
				chroma_cloud_api_key=os.getenv(target.api_key_env),
				tenant=os.getenv(target.tenant_env),
				database=target.database,
			)
		else:
			self.vector_store =  Chroma(collection_name=target.collection,
				embedding_function= self.embeddings,
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
		self.urls = rag_sources.urls
		self.index_report = {}      # url -> {status, chunks, kind}; see _record()
		self.removed_urls = []      # URLs dropped from the collection by this build

		self.text_splitter = RecursiveCharacterTextSplitter(
			chunk_size=1000,  # chunk size (characters)
			chunk_overlap=200,  # chunk overlap (characters)
			add_start_index=True,  # track index in original document
		)

		# self.qa_chain = None
		self.agent = None
		self._setup_rag_agent()


		# model_kwargs = {"device": "cpu"}
		# encode_kwargs = {"normalize_embeddings": True}

	def _setup_rag_agent(self):
		"""Set up the RAG database"""
		# for url in self.urls:
		# 	self._crawl_webpage_and_add_to_rag(url=url)
		if USE_CHROMA_CLOUD:
			self.fetch_current_chromadb_entries()
		for url in self.urls:
			self._alt_crawl_webpage_and_add_to_rag(url=url)
		if USE_CHROMA_CLOUD:
			self.clean_up_extra_urls()
		self.built_at = datetime.now(timezone.utc).isoformat()

		# Create QA chain
		# self.qa_chain = ConversationalRetrievalChain.from_llm(
		# 	llm=self.response_llm,
		# 	retriever=self.vector_store.as_retriever(search_type="similarity",
		# 												search_kwargs={"k": 5}),
		# 	return_source_documents=True
		# )

	def describe_indexed_sources(self):
		"""What this collection actually holds in Chroma right now.

		Reads chunk metadata straight from the vector store, so callers see
		Chroma itself rather than the sources file — the two can differ if the
		file was edited since the last refresh, or a crawl failed.

		Returns {url: {"chunks": int, "kind": "pdf"|"web", "fulltext_hash": str}}.
		Note this pulls the metadata of every chunk in the collection (a few
		hundred here); it is a read-only call and makes no embeddings.
		"""
		entries = self.vector_store.get(include=["metadatas"])
		indexed = {}
		for metadata in entries.get("metadatas") or []:
			metadata = metadata or {}
			url = metadata.get("source")
			if not url:
				continue
			row = indexed.setdefault(url, {"chunks": 0, "kind": "web",
										   "fulltext_hash": metadata.get("fulltext_hash")})
			row["chunks"] += 1
			if "page" in metadata:      # PyPDFLoader tags PDF chunks with a page number
				row["kind"] = "pdf"
		return indexed

	def get_string_hash(self, docs):
		return hashlib.sha256("\n".join([doc.page_content for doc in docs]).encode('utf-8')).hexdigest()

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

	def clean_up_extra_urls(self):
		"""Remove URLs from Chroma DB that are no longer present in the existing URLs set."""
		if USE_CHROMA_CLOUD:
			for url in self.existing_urls_to_chroma_ids.keys():
				if url not in self.urls:
					self.vector_store.delete(ids=self.existing_urls_to_chroma_ids[url])
					self.removed_urls.append(url)
					print(f"Removed URL {url} from Chroma Cloud.")

	def _crawl_url(self, url):
		"""Text for a source URL, or None if nothing usable could be fetched.

		Each candidate (see crawl_candidates) is tried as a web page first and
		then as a PDF, so a link that is really a PDF still gets indexed.
		"""
		for candidate in crawl_candidates(url):
			if candidate != url:
				print(f"Trying open-access endpoint {candidate} for {url}")
			docs = self._load_as_webpage(candidate) or self._load_as_pdf(candidate)
			if docs is not None:
				return docs
		return None

	def _load_as_webpage(self, url):
		try:
			# Only keep post title, headers, and content from the full HTML.
			# bs4_strainer = SoupStrainer(class_=("post-title", "post-header", "post-content"))
			bs4_strainer = SoupStrainer(['p', 'h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'li'])
			loader = WebBaseLoader(
				web_paths=(url, ),
				bs_kwargs={"parse_only": bs4_strainer},
				requests_kwargs={'headers': CRAWL_HEADERS, 'timeout': CRAWL_TIMEOUT},
			)
			docs = loader.load()

			print(f"Total characters: {len(docs[0].page_content)}")
			if not has_usable_content(docs):
				# Empty, or a bot-block notice — try the PDF loader instead.
				raise ValueError(f"Too little text ({len(docs[0].page_content)} chars); not a usable webpage")
			print(f"First few characters of the content: {docs[0].page_content[:200].replace("\n", " ")}")
			return docs
		except Exception as e:
			print(f"Could not read {url} as a webpage: {e}")
			return None

	def _load_as_pdf(self, url):
		temp_file = tempfile.NamedTemporaryFile(mode='wb', suffix='.pdf', delete=False)
		temp_file_path = temp_file.name
		try:
			print(f"Trying to read {url} as a PDF")
			try:
				with requests.get(url, headers=CRAWL_HEADERS, timeout=CRAWL_TIMEOUT) as r:
					r.raise_for_status()
					temp_file.write(r.content)
			finally:
				temp_file.close()
			print(f"Successfully downloaded file to: {temp_file_path}")
			docs = PyPDFLoader(temp_file_path).load()

			print(f"Loaded {len(docs)} documents from {url}")
			print(f"Total characters: {sum(len(doc.page_content) for doc in docs)}")
			if not has_usable_content(docs):
				raise ValueError("PDF produced too little text to index")
			print(f"First few characters of the content: {docs[0].page_content[:100]}")
			return docs
		except Exception as e:
			print(f"Error loading PDF from {url}: {e}")
			return None
		finally:
			if os.path.exists(temp_file_path):
				os.remove(temp_file_path)

	def _record(self, url, status, chunks, kind=None):
		"""Note what this build did with a URL, for the sources viewer.

		status: 'added' (new), 'updated' (content changed), 'unchanged', or
		'failed' (could not be fetched — any chunks from an earlier build stay).
		kind: 'pdf' or 'web' when the content was loaded, else None.
		"""
		self.index_report[url] = {"status": status, "chunks": chunks, "kind": kind}

	def _alt_crawl_webpage_and_add_to_rag(self, url):
		print(f"Crawling webpage: {url}")
		docs = self._crawl_url(url)
		previous_ids = self.existing_urls_to_chroma_ids[url]

		if docs is None:
			self._record(url, "failed", len(previous_ids))
			return
		kind = "pdf" if "page" in (docs[0].metadata or {}) else "web"   # PyPDFLoader adds 'page'
		all_splits = self.text_splitter.split_documents(docs)
		fulltext_hash = self.get_string_hash(docs)

		if (url, fulltext_hash) in self.existing_urls:
			print(f"URL {url} with hash {fulltext_hash} already exists.")
			self._record(url, "unchanged", len(previous_ids), kind)
			return
		else:
			if USE_CHROMA_CLOUD and len(previous_ids) > 0:
				self.vector_store.delete(ids=previous_ids)
			print(f"Adding new content from URL {url}")
			# if not(USE_CHROMA_CLOUD):
			# 	print(docs)
			# Compute hashes for all splits
			for split in all_splits:
				split.metadata['fulltext_hash'] = fulltext_hash
				split.metadata['source'] = url
				

			print(f"Split blog post into {len(all_splits)} sub-documents.")

			document_ids = self.vector_store.add_documents(documents=all_splits)
			print(document_ids)
			self._record(url, "updated" if previous_ids else "added", len(document_ids), kind)

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
		
def build_rag_pipeline(rag_sources, openai_text_model='gpt-5.5', persist_directory="chroma_db", max_completion_tokens=1200):
	"""Crawl `rag_sources` (a rag_sources.RagSources) into its Chroma collection
	and return the ready pipeline — this is what the backend uses per variant."""
	return HPVRAGPipeline(rag_sources, openai_text_model=openai_text_model,
		persist_directory=persist_directory, max_completion_tokens=max_completion_tokens)


def build_rag_agent(rag_sources, openai_text_model='gpt-5.5', persist_directory="chroma_db", max_completion_tokens=1200, audience_instructions=""):
	"""Legacy non-streaming LangChain agent over one pipeline (not used by the
	Flask app; kept for notebook experiments). Returns (agent, pipeline)."""
	rag_pipeline = build_rag_pipeline(rag_sources, openai_text_model=openai_text_model,
		persist_directory=persist_directory, max_completion_tokens=max_completion_tokens)
	@dynamic_prompt
	def _prompt_with_context(request: ModelRequest) -> str:
		"""Inject context into state messages."""
		last_query = request.state["messages"][-1].text
		retrieved_docs = rag_pipeline.vector_store.similarity_search(last_query)

		docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)
		# print(docs_content)
		# print([doc for doc in retrieved_docs])

		system_message = f"{build_system_prompt(audience_instructions)}\n\n{docs_content}"

		return system_message
	agent = create_agent(model=rag_pipeline.openai_text_model, tools=[], middleware=[_prompt_with_context])
	# Return the pipeline alongside the agent so callers can use it for
	# streaming (ask_rag_question_stream) without having to rebuild it.
	return agent, rag_pipeline


def ask_rag_question(agent, messages):
	"""Ask a question about the processed content"""
	try:
		result = agent.invoke(
				{"messages": messages}
			)
		return result['messages'][-1]
	except Exception as e:
		raise Exception(f"Error generating response: {str(e)}")


def retrieve_context_docs(pipeline, messages):
	"""Return the documents retrieved for the last user turn.

	Exposes the same retrieval step used internally by
	ask_rag_question_stream / the _prompt_with_context middleware so callers
	(e.g. the /api/chat endpoint) can record which RAG chunks informed a
	given answer without having to run the similarity search a second time.

	Args:
		pipeline: The HPVRAGPipeline instance returned by build_rag_agent().
		messages: The conversation messages list (OpenAI dict format).

	Returns:
		list: LangChain Document objects (each has .page_content and .metadata).
	"""
	last_query = ""
	for m in reversed(messages):
		if isinstance(m, dict) and m.get("role") == "user":
			last_query = m.get("content", "")
			break
	return pipeline.vector_store.similarity_search(last_query)


def ask_rag_question_stream(pipeline, messages, retrieved_docs=None, survey_block="", audience_instructions=""):
	"""Generator that yields raw text tokens for a streaming RAG response.

	Performs the same retrieval step as the non-streaming path (similarity
	search → inject as system context) but then calls ChatOpenAI.stream()
	instead of invoke() so the caller can forward each token to the client
	as a Server-Sent Event without waiting for the full completion.

	Args:
		pipeline: The HPVRAGPipeline instance returned by build_rag_agent().
		messages: The conversation messages list (OpenAI dict format).
		retrieved_docs: Optional pre-retrieved documents (from
			retrieve_context_docs). When provided, the internal similarity
			search is skipped so the caller and the LLM see exactly the same
			chunks without performing the retrieval twice. Falls back to an
			internal retrieval when None.
		survey_block: Optional pre-formatted system-prompt block describing the
			patient's pre-chat questionnaire answers. Appended to the system
			message so the questionnaire context informs the response. Empty
			string when no questionnaire was provided.
		audience_instructions: Variant-specific system-prompt lines (see
			variants.py), e.g. post-partum framing. Empty string for none.

	Yields:
		str: Each text token/chunk from the LLM as it is produced.
	"""
	# Retrieve relevant documents (mirrors _prompt_with_context middleware)
	# unless the caller already did so and passed them in.
	if retrieved_docs is None:
		retrieved_docs = retrieve_context_docs(pipeline, messages)
	docs_content = "\n\n".join(doc.page_content for doc in retrieved_docs)

	system_message = f"{build_system_prompt(audience_instructions)}\n\n{docs_content}"

	# Append the patient's questionnaire context (if any) so it reaches the LLM.
	if survey_block:
		system_message += survey_block

	# Build the message list for the LLM.
	# The frontend always prepends its own generic system message; we replace it
	# with the RAG-context system message so there is exactly one system turn.
	lc_messages = [{"role": "system", "content": system_message}]
	for m in messages:
		if isinstance(m, dict) and m.get("role") != "system":
			lc_messages.append({"role": m.get("role"), "content": m.get("content", "")})

	# Stream token-by-token from the LLM.
	# chunk.content is normally a str, but in some LangChain / model
	# combinations it can be a list of content blocks.  Handle both.
	for chunk in pipeline.response_llm.stream(lc_messages):
		content = chunk.content
		if isinstance(content, str):
			if content:
				yield content
		elif isinstance(content, list):
			for block in content:
				if isinstance(block, dict) and block.get("type") == "text":
					text = block.get("text", "")
					if text:
						yield text