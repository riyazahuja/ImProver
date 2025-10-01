
Mathlib = Module("Mathlib")

dataset = ImProverDataset(files=[Mathlib.Analysis, Mathlib.Algebra]) # Or from a file, etc., can also add your own id

prompts = (decl.prompt for decl in dataset) # Memory-efficient generator; everything is cached for later

metric = Metric("length") # From configs: can also build your own from within Python
model = ImProverModel("gpt-5") # Supports both local and API models; optionally accepts more parameters for ray stuff and whatnot

dataset.use_metric(metric)
dataset.use_model(model)
dataset.iteration = 1 # can save data from multiple distinct iterations

for decl in dataset:
    if decl.generated_proof.is_correct and decl.generated_proof.score > decl.original_proof.score: 
        # Once again, all cached; everything is batched behind the scenes when these attributes are accessed
        print(f"Improved proof for {decl.name}:\n{decl.generated_proof.proof}\n")

dataset.to_alpaca(criterion=lambda decl: decl.generated_proof.is_correct and decl.generated_proof.score > decl.original_proof.score)


