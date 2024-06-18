import os, json, tqdm, nltk, argparse, torch, random
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer, AutoModel
from utils_topic import load_topic
import torch.nn.functional as F
from torch import Tensor

stopwords = set(nltk.corpus.stopwords.words('english'))


def retrieve_from_keywords(topic_str, document):
    topic_kws = nltk.word_tokenize(topic_str.lower())
    topic_kws = [kw for kw in topic_kws if kw not in stopwords]

    doc_kws = nltk.word_tokenize(document.lower())
    score = sum([1 for kw in topic_kws if kw in doc_kws]) / len(topic_kws)
    return score

def retrieve_from_vector(topic_str, document, sentence_model):
    topic_vec = sentence_model.encode([topic_str])
    doc_vec = sentence_model.encode([document])

    score = cosine_similarity(topic_vec, doc_vec)[0][0]
    return score

def average_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    last_hidden = last_hidden_states.masked_fill(~attention_mask[..., None].bool(), 0.0)
    return last_hidden.sum(dim=1) / attention_mask.sum(dim=1)[..., None]

def get_position_ids(input_ids: Tensor, max_original_positions: int=512, encode_max_length: int=4096) -> Tensor:

    position_ids = list(range(input_ids.size(1)))
    factor = max(encode_max_length // max_original_positions, 1)
    if input_ids.size(1) <= max_original_positions:
        position_ids = [(pid * factor) for pid in position_ids]
        
    position_ids = torch.tensor(position_ids, dtype=torch.long)
    position_ids = position_ids.unsqueeze(0).expand_as(input_ids)
    
    return position_ids

def retrieve_from_longembed(input_texts, tokenizer, model):
    # Tokenize the input texts
    batch_dict = tokenizer(input_texts, max_length=4096, padding=True, truncation=True, return_tensors='pt')
    batch_dict['position_ids'] = get_position_ids(batch_dict['input_ids'], max_original_positions=512, encode_max_length=4096)

    outputs = model(**batch_dict)
    embeddings = average_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    # normalize embeddings
    embeddings = F.normalize(embeddings, p=2, dim=1)
    scores = (embeddings[-1:] @ embeddings[:-1].T) * 100 ## the last one is the query
    return scores



def retrieve_cohere(documents, query):
    response = co.rerank(
    model="rerank-english-v3.0",
    query=query,
    documents=documents,
    top_n=len(documents),
    )
    idx_scores = [(x.index, x.relevance_score) for x in response.results] ## 0-based index score
    idx2score = {}
    for idx, score in idx_scores:
        idx2score[idx] = score
    return idx2score




def populate_retrieval(args):
    topic, _ = load_topic(args.fn)

    tokenizer_longembed = AutoTokenizer.from_pretrained(args.longembed_model)
    model_longembed = AutoModel.from_pretrained(args.longembed_model)
    sentence_model = SentenceTransformer(args.sembed_model)
    for subtopic in tqdm.tqdm(topic["subtopics"], desc=f"Populated retrievers for {args.fn}"):
        if "retriever" not in subtopic:
            subtopic["retriever"] = {}
        for retriever_type in ["keywords", "vector", "oracle", args.longembed_model, "random", "rerank3"]:
            if retriever_type not in subtopic["retriever"]:
                subtopic["retriever"][retriever_type] = {}

        subtopic_id = subtopic["subtopic_id"]
        subtopic_str = subtopic["query"] + " " + subtopic["subtopic"]
        max_subtopic = 0
        for doc in topic['documents']:
            doc_str = doc['document_text']
            doc_id = doc["document_id"]
            if doc_id not in subtopic["retriever"]["keywords"]:
                subtopic["retriever"]["keywords"][doc_id] = retrieve_from_keywords(subtopic_str, doc_str)
            if doc_id not in subtopic["retriever"]["vector"] and not args.skip_vectors:
                subtopic["retriever"]["vector"][doc_id] = float(retrieve_from_vector(subtopic_str, doc_str, sentence_model))
            if doc_id not in subtopic["retriever"]["oracle"]:
                subtopic["retriever"]["oracle"][doc_id] = int(doc[f"label_{subtopic_id}"])
            if doc_id not in subtopic["retriever"][args.longembed_model] and not args.skip_vectors:
                # TODO - can batch for efficiency
                docs_str_longembed = [f"passage: {doc_str}"]
                docs_str_longembed.append(f"query: {subtopic_str}")
                subtopic["retriever"][args.longembed_model][doc_id] = float(retrieve_from_longembed(docs_str_longembed, tokenizer=tokenizer_longembed, model=model_longembed)[0].tolist()[0])
            if doc_id not in subtopic["retriever"]["random"]:
                subtopic["retriever"]["random"][doc_id] = float(random.random())
        if topic['documents'][0]['document_id'] not in subtopic["retriever"]["rerank3"] and args.run_rerank3:
            idx2score = retrieve_cohere([doc['document_text'] for doc in topic['documents']], subtopic_str)
            for doc_idx, doc in enumerate(topic['documents']):
                score = idx2score[doc_idx]
                subtopic["retriever"]["rerank3"][doc["document_id"]] = score
        os.makedirs("data", exist_ok=True)
        with open(args.fn, "w") as f:
            json.dump(topic, f, indent=2)

if __name__ == "__main__":

    # get the fn and evaluator model card
    parser = argparse.ArgumentParser()
    parser.add_argument("--fn", type=str, required=True)
    # parser.add_argument("--domain", type=str, required=True)
    parser.add_argument("--sembed_model", type=str, default="sentence-transformers/all-mpnet-base-v2")
    parser.add_argument("--skip_vectors", action="store_true")
    parser.add_argument("--longembed_model", type=str, default="dwzhu/e5-base-4k")
    parser.add_argument("--run_rerank3", action="store_true")

    args = parser.parse_args()

    if args.run_rerank3:
        import cohere
        co = cohere.Client(os.environ["COHERE_API_KEY"])

    populate_retrieval(args)
