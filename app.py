from flask import Flask, render_template, request, redirect, url_for, session
from transformers import GPT2LMHeadModel, GPT2Tokenizer

app = Flask(__name__)
app.secret_key = "your_secret_key" 

# Load GPT-2 model and tokenizer
model_name = "gpt2"
model = GPT2LMHeadModel.from_pretrained(model_name)
tokenizer = GPT2Tokenizer.from_pretrained(model_name, pad_token_id=model.config.eos_token_id)

# Function to generate song lyrics
def generate_song(input_prompts):
    # Combine input prompts into a single string
    input_text = "\n".join(input_prompts)

    # Tokenize and generate lyrics
    input_ids = tokenizer.encode(input_text, return_tensors="pt", max_length=150, truncation=True)
    output = model.generate(input_ids, max_length=200, num_return_sequences=1, no_repeat_ngram_size=2, pad_token_id=model.config.eos_token_id)

    generated_lyrics = tokenizer.decode(output[0], skip_special_tokens=True)

    # Format the generated song with line breaks
    formatted_song = "\n".join(generated_lyrics.split(". "))
    
    return formatted_song


# Function to format the generated song
def format_song(song_text):
    lines = song_text.split("\n")
    formatted_lines = [line.strip() for line in lines if line.strip() != ""]
    formatted_song = "\n".join(formatted_lines)
    return formatted_song

# Route for the home page
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        input_prompts = [
            request.form["prompt1"],
            request.form["prompt2"],
            request.form["prompt3"]
        ]
        generated_song = generate_song(input_prompts)
        formatted_song = format_song(generated_song)

        if "lyrics" not in session:
            session["lyrics"] = []

        session["lyrics"].append(formatted_song)

    return render_template("index.html", lyrics=session.get("lyrics", []))

if __name__ == "__main__":
    app.run(debug=True)
