# Summary of a Haystack

This repository contains the data and code for the experiments in the SummHay paper.

<p align="center">
  <img height="400" src="SummHay_Illustration.png">
</p>

# Accessing the Data

We publicly release the 10 Haystacks (5 in conversational domain, 5 in the news domain). They are provided in the `data/` folder. There is one Haystack per `.json` file, and they each follow the following schema:
```
{
    "topic_id": "ObjectId()",
    "topic": "",
    "topic_metadata": {"participants": []}, // can be domain specific
    "subtopics": [
        {
            "subtopic_id": "ObjectId()",
            "subtopic_name": "",
            "subtopic": "",
            "insights": [
                {
                    "insight_id": "ObjectId()",
                    "insight_name": "",
                    "insight": ""
                }
            ],
            "query": "question reformulation of the subtopic",
            "retriever": {
                "retriever_method": {
                    "document_id": "0|1"
                }
            },
            "summaries": {
                "summarization_method_xyz": ["line1", "line2", "line3"],
                "{retriever}-{llm_summarizer}": ["line1", "line2", "line3"],
                "summarization_method_abc": ["line1", "line2", "line3"]
            },
            "eval_summaries": {
                "summarization_method_xyz": [
                    {
                        "insight_id": "",
                        "coverage": "NO_COVERAGE|PARTIAL_COVERAGE|FULL_COVERAGE",
                        "bullet_id": "line_number"
                    }
                ]
            }
        }
    ],
    "documents": [
        {
            "document_id": "ObjectId()",
            "document_text": "",
            "document_metadata": [], // domain specific information
            "insights_included": [] // list of insight_ids
        }
    ]
}
```

# Running the Pipeline

The pipeline can be run with three consecutive scripts: (1) `populate_retriever_scores.py` (optional, if implementing a new retriever), (2) `populate_summaries.py` which populates the summary outputs, (3) `populate_eval.py` which generates the evaluation scores (using GPT-4o by default.
Some notes:
- In order to introduce a new retriever/summarizer, one should modify the `generate` functions (which currently map to our internal LLM API) to link to the generative system that should be evaluated.
- We recommend keeping the prompts unmodified (they are provided in `prompts/`), but if you modify the prompt, we highly recommend stating so when reporting results. We did not perform extensive prompt engineering optimization in the results reported in the paper.
- Each script has `argparse` arguments that can help with specific use.

An example of running on the pipeline might look like:
```sh
python populate_summaries.py --fn data/topic_news1.json --domain news --model_cards claude4 --full_sum --retrieval_summ
python populate_eval.py --fn data/topic_news1.json
```

The above would run the SummHay experiment (i.e., generate summaries) for all retrievers, and for the full-context settings, for a model named Claude4 (whose access would have to be implemented within `populate_summaries.py`), followed by the automatic evaluation on those summaries.


# Visualizazing Results on SummHay

The `Evaluation.ipynb` notebook contains the scripts that can be used to compile and visualize results, these are the exact scripts that were used to generate Tables in the paper.



