
from collections import defaultdict
from anyllm import generate # Replace with custom access to a generative system
import json, tqdm, random
import argparse
import tiktoken

def bullet_processor(summary):
    lines = summary.split("\n")
    lines = [l.strip() for l in lines if l.strip() != ""]
    return lines

def summarize(relev_docs, topic, subtopic, model_card):
    relev_docs_str = ""
    for doc in relev_docs:
        if args.domain == "conversation":
            relev_docs_str += f"Conversation {doc[1]+1}:\n{doc[0]}\n\n"
        else:
            relev_docs_str += f"Article {doc[1]+1}:\n{doc[0]}\n\n"

    if args.domain == "conversation":
        prompt_summarization = open("prompts/summarization_conv.txt").read()
        prompt_summarization_populated = prompt_summarization.replace("[N_conversations]", str(len(relev_docs))).replace("[SCENARIO]", topic["topic"]).replace("[PARTICIPANTS]", ", ".join(topic["topic_metadata"]["participants"])).replace("[CONVERSATIONS]", relev_docs_str).replace("[QUERY]", subtopic["query"]).replace("[N_insights]", str(len(subtopic["insights"])))
    else:
        prompt_summarization = open("prompts/summarization_news.txt").read()
        insights_discussed = get_insights_discussed(topic['documents'], subtopic)
        num_insights_discussed = len(insights_discussed)
        prompt_summarization_populated = prompt_summarization.replace("[N_articles]", str(len(relev_docs))).replace("[ARTICLES]", relev_docs_str).replace("[TOPIC]", topic['topic']).replace("[N_insights]", str(num_insights_discussed)).replace("[SUBTOPIC]", subtopic['subtopic'])

    max_tokens = 10000 if "o1-" in model_card else 1000 # Let it have room to think
    response_summary = generate([{"role": "user", "content": prompt_summarization_populated}], model=model_card, step="sohard-summ-gen", max_tokens=max_tokens, timeout=400)
    return response_summary

def get_insights_discussed(documents, subtopic):
    subtopic_insights = set([insight['insight_id'] for insight in subtopic['insights']])
    counter = defaultdict(int)
    for doc in documents[:]:
        cur_insights_discussed = doc['insights_included']
        for insight in cur_insights_discussed:
            if insight in subtopic_insights:
                counter[insight] += 1
    return [x for x, y in counter.items()]

def get_docs_token_limit(documents, subtopic, retriever, max_retrieval_tokens):
    encoding = tiktoken.get_encoding("cl100k_base")
    sorted_docs = sorted(documents, key=lambda x: subtopic["retriever"][retriever][x["document_id"]], reverse=True)
    docs_final = []
    total_tokens = 0
    for doc in sorted_docs:
        doc_str = doc['document_text']
        toks = encoding.encode(doc_str)
        if total_tokens + len(toks) >= max_retrieval_tokens:
            diff = max_retrieval_tokens - total_tokens
            toks_to_add = encoding.decode(toks[:diff])
            docs_final.append((toks_to_add, doc['idx']))
            break
        total_tokens += len(toks)
        docs_final.append((doc_str, doc['idx'])) ## assume that idx is now part of the schema
    return docs_final
    
def populate_subtopic_summaries(args):
    topic = json.load(open(args.fn, "r" ))

    for idx, doc in enumerate(topic["documents"]):
        if topic["documents"][idx].get("idx") is None:
            topic["documents"][idx]["idx"] = idx

    model_cards = args.model_cards
    if len(model_cards) > 1:
        model_cards = tqdm.tqdm(model_cards, desc=f"Populating summaries of {args.fn}")
    for model_card in model_cards:
        for subtopic in tqdm.tqdm(topic["subtopics"]):
            if "summaries" not in subtopic:
                subtopic["summaries"] = {}
                subtopic["eval_summaries"] = {}
            if args.retrieval_summ:
                retrievers = subtopic["retriever"].keys()
                for retriever in retrievers:
                    pop_key = f"summary_subtopic_{retriever}_{model_card}"
                    if pop_key in subtopic["summaries"]:
                        continue
                    relev_docs = get_docs_token_limit(topic["documents"], subtopic, retriever, args.token_limit)
                    response_summary = summarize(relev_docs, topic, subtopic, model_card)
                    subtopic["summaries"][pop_key] = bullet_processor(response_summary)
                    with open(args.fn, "w") as f:
                        json.dump(topic, f, indent=2)
            if args.full_summ or args.full_summ_sorted or args.full_summ_reverse_sorted or args.full_summ_middle:
                if args.full_summ:
                    pop_key = f"summary_subtopic_{model_card}"
                elif args.full_summ_sorted:
                    pop_key = f"summary_subtopic_fl-ctxt-sort_{model_card}"
                elif args.full_summ_reverse_sorted:
                    pop_key = f"summary_subtopic_fl-ctxt-rev-sort_{model_card}"
                elif args.full_summ_middle:
                    pop_key = f"summary_subtopic_fl-ctxt-middle_{model_card}"
                if pop_key in subtopic["summaries"]:
                    continue
                prepped_docs = [(doc['document_text'], doc['idx'], doc["document_id"]) for doc in topic["documents"]]
                if args.full_summ_sorted or args.full_summ_reverse_sorted:
                    prepped_docs = sorted(prepped_docs, key=lambda x: subtopic["retriever"]["oracle"][x[2]], reverse=args.full_summ_sorted)

                if args.full_summ_middle:
                    # We need to set document scores
                    doc_scores = {doc["document_id"]: subtopic["retriever"]["oracle"][doc["document_id"]] for doc in topic["documents"]}
                    # for the ones that receive a 0, half of them should be switched to 10 (so they're at the top), the others will be at the bottom
                    for doc_id in doc_scores:
                        if doc_scores[doc_id] == 0 and random.random() < 0.5:
                            doc_scores[doc_id] = 10
                    prepped_docs = sorted(prepped_docs, key=lambda x: doc_scores[x[2]], reverse=True)

                response_summary = summarize(prepped_docs, topic, subtopic, model_card)
                subtopic["summaries"][pop_key] = bullet_processor(response_summary)
                with open(args.fn, "w") as f:
                    json.dump(topic, f, indent=2)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--fn", type=str, required=True)
    parser.add_argument("--domain", type=str, required=True, choices=["conversation", "news"])
    parser.add_argument("--token_limit", type=int, default=15000, help="Maximum number of context tokens to retrieve")
    parser.add_argument("--model_cards", nargs='+', default=["full"], required=True, help="List of model cards to use for summarization")
    parser.add_argument('--retrieval_summ', action='store_true', help="Whether to summarize the retrieved documents")
    parser.add_argument('--full_summ', action='store_true', help="Whether to summarize the full set of documents")
    parser.add_argument('--full_summ_sorted', action='store_true', help="Whether to summarize the full set of documents")
    parser.add_argument('--full_summ_reverse_sorted', action='store_true', help="Whether to summarize the full set of documents")
    parser.add_argument('--full_summ_middle', action='store_true', help="Whether to summarize the full set of documents")

    args = parser.parse_args()

    assert not (args.full_summ_sorted and args.full_summ_reverse_sorted), "Cannot have both full_summ_sorted and full_summ_reverse_sorted"

    if args.model_cards == ["full"]:
        args.model_cards = ["gpt4-turbo", "gpt-4o", "claude3-haiku", "claude3-sonnet", "claude3-opus", "command-r", "command-r-plus", "gemini-1.5-flash", "gemini-1.5-pro"]

    print("Running with models: ", args.model_cards)

    populate_subtopic_summaries(args)
