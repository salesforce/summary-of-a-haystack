from anyllm import generate_json # Replace with an function that can run GPT-4o
import json, tqdm


def evaluate_insights(insights, bullets, evaluator_model_card, eval_prompt_fn="prompts/eval_summhay.txt", return_cost=False):
    with open(eval_prompt_fn, "r") as f:   
        prompt_eval = f.read()

    eval_cost = 0.0

    bullets_str = json.dumps({"bullets": [{"bullet_id": i+1, "text": bullet} for i, bullet in enumerate(bullets)]}, indent=1)
    insight_scores = []
    for insight in insights:
        populated_prompt = prompt_eval.replace("[[BULLETS]]", bullets_str).replace("[[INSIGHT]]", insight["insight"])
        
        response_all = generate_json([{"role": "user", "content": populated_prompt}], model=evaluator_model_card, step="sohard-insight-eval", return_metadata=True)
        eval_cost += response_all["total_usd"]        
        response_json  = response_all["message"]
        response_json["insight_id"] = insight["insight_id"]
        insight_scores.append(response_json)
    if return_cost:
        return insight_scores, eval_cost
    else:
        return insight_scores

def populate_insight_evaluation(fn, evaluator_model_card="gpt-4o"):
    with open(fn, "r") as f:
        topic = json.load(f)

    for subtopic in tqdm.tqdm(topic["subtopics"], desc=f"Populating insights evaluation of {fn}"):
        if "eval_summaries" not in subtopic:
            subtopic["eval_summaries"] = {}
        summ_keys = subtopic.get("summaries", {}).keys()
        for summ_key in summ_keys:
            bullets = subtopic["summaries"][summ_key]
            if summ_key in subtopic["eval_summaries"] and len(subtopic["eval_summaries"][summ_key]) == len(subtopic["insights"]):
                continue

            subtopic["eval_summaries"][summ_key] = evaluate_insights(subtopic["insights"], bullets, evaluator_model_card)
            with open(fn, "w") as f:
                json.dump(topic, f, indent=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--fn", type=str, required=True)
    parser.add_argument("--evaluator_model_card", type=str, default="gpt-4o")
    args = parser.parse_args()

    populate_insight_evaluation(args.fn, args.evaluator_model_card)
