def exact_match_accuracy(predictions, references):
    if not references:
        return 0.0
    correct = sum(1 for p, r in zip(predictions, references) if p == r)
    return correct / len(references)

def eval_step_trace(generated_traces, expected_traces):
    if not expected_traces:
        return 0.0
    score = sum(1 for g, e in zip(generated_traces, expected_traces) if g == e)
    return score / len(expected_traces)

def compare_models_v1_placeholder(model_out, fallback_out, groq_out, ground_truth):
    return {
        "model_exact_match": model_out == ground_truth,
        "fallback_exact_match": fallback_out == ground_truth,
        "groq_exact_match": groq_out == ground_truth
    }

def categorize_error_v1_placeholder(prediction, reference):
    if not prediction:
        return "empty_output"
    if len(prediction) < len(reference) / 2:
        return "severe_truncation"
    if prediction.lower() == reference.lower():
        return "casing_mismatch"
    return "logic_or_hallucination"

def run_error_analysis_v1_placeholder(predictions, references):
    report = {}
    for p, r in zip(predictions, references):
        if p != r:
            category = categorize_error_v1_placeholder(p, r)
            report[category] = report.get(category, 0) + 1
    return report