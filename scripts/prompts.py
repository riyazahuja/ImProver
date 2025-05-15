def generate_prompts_file(filename="prompts.txt", num_prompts=200):
    """
    Generate a text file with various prompts, one per line.
    
    Args:
        filename (str): Name of the output file
        num_prompts (int): Number of prompts to generate
    """
    
    prompts = [
        # General knowledge questions
        "What is the capital of France?",
        "Who wrote the novel 'Pride and Prejudice'?",
        "What is the chemical symbol for gold?",
        "What year did the first moon landing occur?",
        "Who painted the Mona Lisa?",
        "What is the largest planet in our solar system?",
        "What is the boiling point of water in Celsius?",
        "Who developed the theory of relativity?",
        "What is the tallest mountain in the world?",
        "What element has the atomic number 1?",
        
        # Technology questions
        "What programming language is known for its use in data science?",
        "What does CPU stand for?",
        "What year was the first iPhone released?",
        "What is the purpose of HTML in web development?",
        "What does API stand for?",
        "What is cloud computing?",
        "What is machine learning?",
        "What is blockchain technology?",
        "What is the difference between RAM and ROM?",
        "What is an operating system?",
        
        # History questions
        "When did World War II end?",
        "Who was the first president of the United States?",
        "What ancient civilization built the pyramids in Egypt?",
        "When did the Roman Empire fall?",
        "What was the Renaissance?",
        "Who was Alexander the Great?",
        "What was the Industrial Revolution?",
        "When did Christopher Columbus reach the Americas?",
        "What was the Cold War?",
        "Who was Mahatma Gandhi?",
        
        # Science questions
        "What is photosynthesis?",
        "What is Newton's first law of motion?",
        "What is the scientific name for humans?",
        "What is DNA?",
        "What is the periodic table?",
        "What causes a rainbow?",
        "What is the greenhouse effect?",
        "What are the states of matter?",
        "What is the speed of light?",
        "What is the difference between weather and climate?",
        
        # Personal reflection questions
        "What is your greatest achievement?",
        "What skill would you like to master?",
        "What is your favorite book and why?",
        "Where do you see yourself in five years?",
        "What is your biggest fear?",
        "What makes you happy?",
        "What would you change about your past if you could?",
        "What are your career goals?",
        "What is your favorite childhood memory?",
        "What advice would you give to your younger self?",
    ]
    
    # Generate additional generic prompts to reach the desired number
    while len(prompts) < num_prompts:
        n = len(prompts) + 1
        prompts.append(f"What is the answer to question {n}?")
    
    # Write prompts to file
    with open(filename, "w") as f:
        for prompt in prompts[:num_prompts]:
            f.write(prompt + "\n")
    
    print(f"Successfully generated {num_prompts} prompts in '{filename}'")

if __name__ == "__main__":
    generate_prompts_file()