import os
import json
import glob
from model.checkpoint_utils import create_dummy_checkpoint, validate_checkpoint
from inference.router import standalone_inference
from inference.eval_harness import (
    exact_match_accuracy,
    eval_step_trace,
    compare_models_v1_placeholder,
    run_error_analysis_v1_placeholder,
)

def load_benchmark_data(benchmark_dir="eval/benchmarks"):
    inputs = []
    ground_truths = []
    expected_traces = []
    
    if not os.path.exists(benchmark_dir):
        return inputs, ground_truths, expected_traces
        
    for filepath in glob.glob(os.path.join(benchmark_dir, "*.json")):
        with open(filepath, 'r') as f:
            data = json.load(f)
            for item in data:
                inputs.append(item.get("input", ""))
                ground_truths.append(item.get("output", ""))
                expected_traces.append(item.get("trace", []))
                
    return inputs, ground_truths, expected_traces

def run_end_to_end_pipeline():
    checkpoint_path = "checkpoints/dummy_model.pt"
    is_dummy = "dummy" in checkpoint_path
    
    if is_dummy:
        expected_shapes = {
            "linear.weight": (10, 10),
            "linear.bias": (10,)
        }
        if not os.path.exists(checkpoint_path):
            create_dummy_checkpoint(checkpoint_path)
    else:
        expected_shapes = {
            "CalculusSolverModel.real_key_placeholder": (0, 0)
        }
        
    validate_checkpoint(checkpoint_path, expected_shapes)
    
    inputs, ground_truths, expected_traces = load_benchmark_data()
    
    predictions = []
    generated_traces = []
    
    for x in inputs:
        output = standalone_inference(checkpoint_path, x, strategy="beam")
        if isinstance(output, dict):
            predictions.append(output.get("prediction", ""))
            generated_traces.append(output.get("trace", []))
        else:
            predictions.append(output)
            generated_traces.append([])
        
    em_score = exact_match_accuracy(predictions, ground_truths)
    trace_score = eval_step_trace(generated_traces, expected_traces)
    error_report = run_error_analysis_v1_placeholder(predictions, ground_truths)
    
    print(f"Accuracy: {em_score}")
    print(f"Trace Score: {trace_score}")
    print(f"Errors: {error_report}")

if __name__ == "__main__":
    run_end_to_end_pipeline()