def standalone_inference(checkpoint_path, input_data, strategy="beam"):
    from inference.core import CalculusSolverInference
    
    inferencer = CalculusSolverInference(checkpoint_path)
    
    if strategy == "beam":
        return inferencer.beam_search_decode(input_data)
    elif strategy == "standard":
        return inferencer.solve(input_data)
        
    raise ValueError("Unknown strategy")