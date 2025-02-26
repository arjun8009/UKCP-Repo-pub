import warnings
warnings.filterwarnings("ignore")
import pandas as pd
import numpy as np
from prompt_automated import system_instruction, user_prompt
from query_prompt import query_system_instruction, query_prompt
import os
import openai
import pickle
import pandas as pd
import numpy as np
from scipy import spatial
import warnings
import itertools
from haystack import Document
from haystack.nodes import CohereRanker,DiversityRanker,LostInTheMiddleRanker
from llama_index.core.retrievers import VectorIndexRetriever
from get_pickle import get_vectorizer

os.environ["OPENAI_API_KEY"] = "<Enter your API key>"
client = openai.OpenAI()
model = "gpt-4o-mini"


def generate_queries(original_query, multiple_query_generation, query_k):

    generated_queries = [original_query]

    if multiple_query_generation:
        global query_prompt
        global query_system_instruction

        chat_history = []

        system_prompt_dup = query_system_instruction
        query_system_instruction = query_system_instruction.replace("<NUMBER>", str(query_k))
        chat_history.append({"role": "system", "content": query_system_instruction})
        query_system_instruction = system_prompt_dup

        user_prompt_dup = query_prompt
        query_prompt = query_prompt.replace("<QUESTION>", original_query)
        chat_history.append({"role": "user", "content": query_prompt})
        query_prompt = user_prompt_dup

        response = client.chat.completions.create(
            model=model,
            messages=chat_history,
            temperature=0,
            max_tokens=2048,
        )

        new_queries = response.choices[0].message.content.strip().split("\n")
        generated_queries += [q[3:] for q in new_queries]

    query_embeddings = [client.embeddings.create(model=embedding_model,input=query).data[0].embedding for query in generated_queries]
    print("In multiple queries ", generated_queries)
    return generated_queries, query_embeddings

def reciprocal_rank_fusion(search_results_dict, top_k, k=60):
    fused_scores = {}
        
    for chunk_indices in search_results_dict.values():
        for rank, chunk_index in enumerate(chunk_indices):
            if chunk_index not in fused_scores.keys():
                fused_scores[chunk_index] = 0
            fused_scores[chunk_index] += 1 / (rank + k)

    reranked_results = [int(chunk_index) for chunk_index, score in sorted(fused_scores.items(), key=lambda x: x[1], reverse=True)][:top_k]
    return reranked_results

def hierarchical_retrieval(queries, query_embeddings, hierarchical_k, hierarchical_reranking, ranker):

    path = path = os.path.join(os.getcwd(),"python backend","data")
    data = pd.read_parquet(f"{path}/document_summaries.parquet")
    relatedness_fn = lambda a, b: 1 - spatial.distance.cosine(a, b) 

    retrieved_ids, retrieved_chunks = {}, {}
    for i, query_embedding in enumerate(query_embeddings):
        question_relevance = np.round(np.array([relatedness_fn(query_embedding, embedding) for embedding in data.embedding]).astype(float), 16)  # distance between question embeddings and summary embeddings
        retrieved_ids[i] = np.array(data.file_id)[np.argsort(question_relevance)[::-1]][:hierarchical_k]
        retrieved_chunks[i] = np.array(data.chunk)[np.argsort(question_relevance)[::-1]][:hierarchical_k]

    reranked_file_ids = retrieved_ids[0] if hierarchical_reranking == False else re_ranker(queries[0], retrieved_ids, retrieved_chunks, ranker, top_k=hierarchical_k)

    return reranked_file_ids

def small_to_big_retrieval(retrieved_nodes):

    similarity_top_k=5
    desired_sections =  list(dict.fromkeys([node.metadata['section_id'] for node in retrieved_nodes]))[:similarity_top_k]

    path = "<Enter Path to chunks>"
    section_data = pd.read_parquet(f"{path}/data_chunks_sections.parquet")
    section_data = section_data[section_data.section_id.isin(desired_sections)]

    final_chunks = section_data.chunk.tolist()
    final_headings = section_data.heading.tolist()
    final_file_names = section_data.file_name.tolist()
    
    return final_chunks, final_headings, final_file_names


def get_llm_response(chunks, query, file_names, headings):

    global user_prompt

    chat_history = []
    chat_history.append({"role": "system", "content": system_instruction})
    user_prompt_dup = user_prompt
    user_prompt = user_prompt.replace("<EXCERPTS>", "<SEP>".join(chunks))
    user_prompt = user_prompt.replace("<QUESTION>", query)
    # user_prompt = user_prompt.replace("<FILES>", "<SEP>".join(file_names))
    # user_prompt = user_prompt.replace("<SECTIONS>", "<SEP>".join(headings))
    chat_history.append({"role": "user", "content": user_prompt})
    user_prompt = user_prompt_dup

    response = client.chat.completions.create(model="gpt-4o-mini",messages=chat_history,temperature=0,max_tokens=1024,)
    response_message = response.choices[0].message.content

    return response_message


def re_ranker(query, retrieved_ids, retrieved_chunks, ranker, top_k=25):

    if ranker == 'reciprocal':
        return  reciprocal_rank_fusion(retrieved_ids, top_k)
    
    # indices method
    retrieved_ids = list(itertools.chain.from_iterable(retrieved_ids.values())) 
    retrieved_chunks = list(itertools.chain.from_iterable(retrieved_chunks.values())) 
    retrieved_chunks_ids_set = {retrieved_chunks[i]: retrieved_ids[i] for i in range(len(retrieved_chunks))}
    documents = [Document(content=key,id=val) for key, val in retrieved_chunks_ids_set.items()]

    if ranker == 'cohere':
        cohere_api_key = "<Enter Cohere API Key>"
        ranker = CohereRanker(api_key=cohere_api_key, model_name_or_path="rerank-multilingual-v2.0", top_k=top_k,) 

    elif ranker == 'diversity':
        # ranker = DiversityRanker(model_name_or_path="all-MiniLM-L6-v2", top_k=top_k, use_gpu=True, devices=["torch.device"], similarity="cosine",)
        ranker = DiversityRanker(model_name_or_path="all-MiniLM-L6-v2", top_k=top_k, use_gpu=False, similarity="cosine",)
    
    elif ranker == 'lost':
        ranker = LostInTheMiddleRanker(word_count_threshold=2048, top_k=top_k,)

    ranked_docs = ranker.predict(query, documents, top_k=top_k)
    ranked_docs_ids = [int(doc.id) for doc in ranked_docs]

    return ranked_docs_ids



def retrieve_nodes(queries, chunking_method, similarity_top_k, ranker=None, metadata=None):

    print("queries", queries)
    retrieved_nodes = get_vectorizer(chunking_method,similarity_top_k,metadata,queries) # Methods to get Vectorizer
    retrieved_ids = {key: [node.id_ for node in val] for key, val in retrieved_nodes.items()}
    retrieved_chunks = {key: [node.text for node in val] for key, val in retrieved_nodes.items()}
    print("retrieved chunks", retrieved_chunks)
    reranked_ids = retrieved_ids[0] if ranker == None else re_ranker(queries[0], retrieved_ids, retrieved_chunks, ranker, top_k=similarity_top_k) # discuss how to provide chunks without reranking

    final_nodes = []
    for id in reranked_ids:
        for node in list(itertools.chain.from_iterable(retrieved_nodes.values())):
            if int(id) == int(node.id_):
                final_nodes.append(node)
                break

    return final_nodes


def answer_generation(question, chunking_method, retrieval_type, hierarchical_k, chunking_k, query_k, multiple_query_generation, 
         hierarchical_reranking, ranker, model='gpt-4o-mini'):

    

    # generate multiple queries and their embeddings, and place them both in lists
    queries, query_embeddings = generate_queries(question, multiple_query_generation, query_k)

    metadata = None
    
    if 'hierarchical' in retrieval_type:
        # semantic search between queries and whole document summaries, reduce dataset before searching for smaller chunks
        metadata = hierarchical_retrieval(queries, query_embeddings, hierarchical_k, hierarchical_reranking, ranker)
        
    # semantic search for most relevant chunks
    retrieved_nodes = retrieve_nodes(queries, chunking_method=chunking_method, similarity_top_k=20, ranker=ranker, metadata=metadata)
    
    if 'small-to-big' in retrieval_type:
        # extract section chunks from which the most relevant paragraph / fixed length chunks derive    
        final_chunks, final_headings, final_file_names = small_to_big_retrieval(retrieved_nodes)
    else:
        final_chunks = [node.text for node in retrieved_nodes][:chunking_k]
        final_headings = [node.metadata['heading'] for node in retrieved_nodes][:chunking_k]
        final_file_names = [node.metadata['file_name'] for node in retrieved_nodes][:chunking_k]

    response = get_llm_response(final_chunks, question, final_file_names, final_headings)
    # print("Response generated.")

    return response, final_file_names, final_headings, final_chunks

def get_GPT_answer(question):
    chat_history = []
    chat_history.append({"role": "system", "content": "Give a good answer related to UKCP"})
    chat_history.append({"role":"user","content":question})
    response = client.chat.completions.create(model=model,messages=chat_history,temperature=0,max_tokens=1024,)
    response_message = response.choices[0].message.content
    return response_message,["GPT-info"],["GPT-info"] ,["GPT-info"]

def response_from_pipeline(question,pipeline_index):
    if pipeline_index == "1" or pipeline_index == 1:
        response,final_file_names, final_headings, final_chunks = answer_generation(question,chunking_method="paragraph",retrieval_type=[None],chunking_k=20,hierarchical_k=None,query_k=None,multiple_query_generation=False,hierarchical_reranking=False,ranker=None,model=model)
    elif pipeline_index == "2" or pipeline_index == 2:
        response,final_file_names, final_headings, final_chunks = answer_generation(question,chunking_method="paragraph",retrieval_type=["small-to-big","hierarchical"],chunking_k=20,hierarchical_k=15,query_k=None,multiple_query_generation=False,hierarchical_reranking=False,ranker=None,model=model)
    elif pipeline_index =="3" or pipeline_index == 3:
        response,final_file_names, final_headings, final_chunks = answer_generation(question,chunking_method="paragraph",retrieval_type=["small-to-big","hierarchical"],chunking_k=20,hierarchical_k=20,query_k=None,multiple_query_generation=False,hierarchical_reranking=True,ranker="cohere",model=model)
    elif pipeline_index == "4" or pipeline_index == 4:
        print("Entered 4")
        response,final_file_names, final_headings, final_chunks = answer_generation(question,chunking_method="paragraph",retrieval_type=["small-to-big","hierarchical"],chunking_k=20,hierarchical_k=20,query_k=5,multiple_query_generation=True,hierarchical_reranking=True,ranker="cohere",model=model)
    else:
        print("Entered 5")
        response,final_file_names,final_headings, final_chunks = get_GPT_answer(question)

    return response,final_file_names, final_headings, final_chunks
