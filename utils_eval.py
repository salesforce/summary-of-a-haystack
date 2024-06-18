import json, numpy as np, re

def build_ref_insight2docids(topic):
    insight_id2references = {}
    for i, doc in enumerate(topic["documents"]):
        doc_id = i + 1
        for insight_id in doc["insights_included"]:
            if insight_id not in insight_id2references:
                insight_id2references[insight_id] = set([])
            insight_id2references[insight_id].add(doc_id)
    return insight_id2references

def extract_citations(bullet):
    # matches digits or commas
    matches = re.findall(r"\[([\d, ]+)\]", bullet)
    ref_ids = []
    for match in matches:
        ref_ids += [int(m.strip()) for m in match.split(",") if len(m.strip()) > 0]
    return ref_ids

def compute_single_sample_scores(bullets, evals, topic, insightid2ref_citations=None, partial_score=0.5, cite_offset=0): # the cite offset should be one for the annotators (but not for the model eval)
    if insightid2ref_citations is None:
        insightid2ref_citations = build_ref_insight2docids(topic)

    coverage_scores, citation_scores, joint_scores = [], [], []
    citation_precisions, citation_recalls = [], []
    for e in evals:
        cov_score, cit_score, cit_prec, cit_rec = 0.0, 0.0, 0.0, 0.0
        if e["coverage"] in ["PARTIAL_COVERAGE", "FULL_COVERAGE"]:
            cov_score = 1.0 if e["coverage"] == "FULL_COVERAGE" else partial_score
            insight_id = e["insight_id"]
            try:
                bullet_match_idx = int(e["bullet_id"])
            except:
                bullet_match_idx = -1
            bullet_match = bullets[bullet_match_idx - 1]

            gen_citations = set([cite+cite_offset for cite in extract_citations(bullet_match)])
            ref_citations = insightid2ref_citations[insight_id]

            P = 0 if len(gen_citations) == 0 else len(gen_citations & ref_citations) / len(gen_citations)
            R = 0 if len(ref_citations) == 0 else len(gen_citations & ref_citations) / len(ref_citations)
            F1 = 0 if P + R == 0 else 2 * P * R / (P + R)
            cit_prec, cit_rec, cit_score = P, R, F1
            citation_scores.append(cit_score)
            citation_precisions.append(cit_prec)
            citation_recalls.append(cit_rec)

        coverage_scores.append(cov_score)
        joint_scores.append(cov_score * cit_score)
    return {"coverage_score": coverage_scores, "citation_score": citation_scores, "joint_score": joint_scores, "citation_precision": citation_precisions, "citation_recall": citation_recalls}

def compute_single_sample_results(bullets, evals, topic, insightid2ref_citations=None, partial_score=0.5, cite_offset=0):
    scores = compute_single_sample_scores(bullets, evals, topic, insightid2ref_citations, partial_score, cite_offset=cite_offset)
    return {k: np.mean(v) for k, v in scores.items()}

def compute_full_results(fn, partial_score=0.5, skip_mean=False):
    with open(fn, "r") as f:
        topic = json.load(f)

    models_run = list(topic["subtopics"][0]["summaries"].keys())
    models_run = [m.replace("summary_subtopic_", "") for m in set(models_run)]

    insightid2ref_citations = build_ref_insight2docids(topic)

    flat_results = []
    for model in models_run:
        summary_key = f"summary_subtopic_{model}"
        coverage_scores, citation_scores, joint_scores, citation_precisions, citation_recalls = [], [], [], [], []
        for subtopic in topic["subtopics"]:
            if summary_key in subtopic["summaries"] and summary_key in subtopic["eval_summaries"]:
                sample_scores = compute_single_sample_scores(subtopic["summaries"][summary_key], subtopic["eval_summaries"][summary_key], topic, insightid2ref_citations, partial_score)
                coverage_scores += sample_scores["coverage_score"]
                citation_scores += sample_scores["citation_score"]
                joint_scores += sample_scores["joint_score"]
                citation_precisions += sample_scores["citation_precision"]
                citation_recalls += sample_scores["citation_recall"]
        if not skip_mean:
            flat_results.append({"model": model, "coverage_score": np.mean(coverage_scores), "citation_score": np.mean(citation_scores), "joint_score": np.mean(joint_scores), "citation_precision": np.mean(citation_precisions), "citation_recall": np.mean(citation_recalls)})
        else:
            flat_results.append({"model": model, "coverage_score": coverage_scores, "citation_score": citation_scores, "joint_score": joint_scores, "citation_precision": citation_precisions, "citation_recall": citation_recalls})
    if not skip_mean:
        flat_results = sorted(flat_results, key=lambda x: x["joint_score"])
    return flat_results

def resort_columns(results, sorted_columns):
    new_results = {}
    for key in results:
        new_results[key] = {col: results[key].get(col, np.nan) for col in sorted_columns}
    return new_results

def compute_2d_rag_results(fns, partial_score=0.5, retrivers_skip=[], summarizer_skip=[], sort_key=None, sort_retriever_key=None, score_keys=["coverage_score", "citation_score", "joint_score"]):
    if "count" not in score_keys:
        score_keys += ["count"]

    results = {score_key: {} for score_key in score_keys}

    for fn in fns:
        flat_results = compute_full_results(fn, partial_score, skip_mean=True)

        for score_key in score_keys:
            for r in flat_results:
                if r["model"].count("_") == 1:
                    retriever, summarizer = r["model"].split("_")
                else:
                    retriever, summarizer = "full-ctxt", r["model"]

                if retriever == "dwzhu/e5-base-4k":
                    retriever = "longembed"

                if retriever in retrivers_skip or summarizer in summarizer_skip:
                    continue

                retriever = retriever[-10:] # Shorten the name

                if summarizer not in results[score_key]:
                    results[score_key][summarizer] = {"Summarizer": summarizer}

                if retriever not in results[score_key][summarizer]:
                    results[score_key][summarizer][retriever] = []
                
                if score_key == "count":
                    results[score_key][summarizer][retriever] += [1] * len(r["coverage_score"])
                else:
                    results[score_key][summarizer][retriever] += r[score_key]

    # N_samples_per_condition = Counter([len(results["coverage_score"][summarizer][retriever]) for summarizer in results["coverage_score"] for retriever in results["coverage_score"][summarizer] if retriever != "Summarizer"])
    # print(N_samples_per_condition)

    for score_key in score_keys:
        for summarizer in results[score_key]:
            for retriever in results[score_key][summarizer]:
                if retriever == "Summarizer":
                    continue
                if score_key == "count":
                    results[score_key][summarizer][retriever] = len(results[score_key][summarizer][retriever])
                else:
                    results[score_key][summarizer][retriever] = np.mean(results[score_key][summarizer][retriever])

    all_retrievers = set([r for row in results[score_keys[0]].values() for r in row.keys()])
    all_retrievers = [r for r in all_retrievers if r != "Summarizer"]
    all_summarizers = list(results[score_keys[0]].keys())

    if sort_key is not None:
        sort_summ_scores = {}
        for summarizer in all_summarizers:
            if sort_retriever_key is not None:
                sort_summ_scores[summarizer] = results[sort_key][summarizer][sort_retriever_key]
            else:
                sort_summ_scores[summarizer] = np.mean([results[sort_key][summarizer][retriever] for retriever in all_retrievers if retriever in results[sort_key][summarizer]])
        all_summarizers = sorted(all_summarizers, key=lambda x: sort_summ_scores[x])
        
        sort_retriever_scores = {}
        for retriever in all_retrievers:
            sort_retriever_scores[retriever] = np.mean([results[sort_key][summarizer][retriever] for summarizer in all_summarizers if retriever in results[sort_key][summarizer]])
        all_retrievers = sorted(all_retrievers, key=lambda x: sort_retriever_scores[x])
        # make sure that if "full-ctxt" is a retriever, it comes first
        if "full-ctxt" in all_retrievers:
            all_retrievers = [ret for ret in all_retrievers if ret != "full-ctxt"] + ["full-ctxt"]

    sorted_columns = ["Summarizer"] + all_retrievers
    results = {score_key: resort_columns(results[score_key], sorted_columns) for score_key in score_keys}

    results = {score_key: [results[score_key][summarizer] for summarizer in all_summarizers] for score_key in score_keys}

    return results
