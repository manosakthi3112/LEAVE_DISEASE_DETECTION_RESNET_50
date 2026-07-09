from transformers import GPT2LMHeadModel, GPT2Tokenizer

# Step 1: Load the Pre-trained GPT-2 Model and Tokenizer
model_name = "gpt2"  # You can use "gpt2-medium" or another variant
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name)

# Step 2: Save the Model and Tokenizer
save_directory = "./gpt2_plant_disease_model"
model.save_pretrained(save_directory)
tokenizer.save_pretrained(save_directory)
print(f"Model and tokenizer saved to {save_directory}")

# Step 3: Load the Saved Model and Tokenizer
model = GPT2LMHeadModel.from_pretrained(save_directory)
tokenizer = GPT2Tokenizer.from_pretrained(save_directory)
print("Model and tokenizer loaded successfully")

def generate_curing_methods(prompt, max_length=50):
    """
    Generate curing methods for a plant disease based on a given prompt.
    
    :param prompt: The initial text to guide the method generation.
    :param max_length: The maximum length of the generated text.
    :return: Generated text as a string.
    """
    inputs = tokenizer.encode(prompt, return_tensors="pt")
    outputs = model.generate(
        inputs, 
        max_length=max_length, 
        num_return_sequences=1, 
        do_sample=True, 
        temperature=0.7, 
        top_k=50, 
        top_p=0.95, 
        repetition_penalty=1.2,  # Penalizes repetition
        pad_token_id=tokenizer.eos_token_id
    )
    methods = tokenizer.decode(outputs[0], skip_special_tokens=True)
    return methods
if __name__ == "__main__":
    # Example prompt about a plant disease
    prompt = "Suggest methods to cure powdery mildew on plants.. Describe its symptoms on plants."
    description = generate_disease_description(prompt, max_length=100)
    print("\nGenerated Description:")
    print(description)

