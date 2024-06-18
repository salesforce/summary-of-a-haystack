import json

def load_topic(fn):
    with open(fn, "r") as f:
        scenario = json.load(f)
    documents = scenario["documents"]

    insight_id2subtopic_id = {}
    all_ids = []
    for subtopic in scenario["subtopics"]:
        insight_id2subtopic_id.update({insight["insight_id"]: subtopic["subtopic_id"] for insight in subtopic["insights"]})
        all_ids.append(subtopic["subtopic_id"])
        all_ids.extend([insight["insight_id"] for insight in subtopic["insights"]])

    # for retrieval-only evaluation
    for document in documents:
        for subtopic_id in all_ids:
            document[f"label_{subtopic_id}"] = 0
        for insight_id in document["insights_included"]:
            document[f"label_{insight_id}"] = 1
            document[f"label_{insight_id2subtopic_id[insight_id]}"] = document.get(f"label_{insight_id2subtopic_id[insight_id]}", 0) + 1

    return scenario, documents
