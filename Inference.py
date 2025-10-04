import torch # pyright: ignore[reportMissingImports]

from transformers import ( # pyright: ignore[reportMissingImports]
    AutoTokenizer,
    MarianMTModel,
    pipeline
)

# path of model
path = "./opus-mt-en-hi-4-bit-quantized-finetuned-model"

def load_pipeline(path):
    tokenizer = AutoTokenizer.from_pretrained(path)
    print("Tokenizer Loaded Successfully")
    inference_model = MarianMTModel.from_pretrained(path, device_map='auto')
    print("Model Loaded Successfully")
    inference_model.eval() # set model for evaluation
    # set up pipeline
    translator = pipeline(
            "translation",
            tokenizer=tokenizer,
            model=inference_model,
        )
    return translator

def inference_fn(text: str, translator):
    if text=="":
        text ="Please Enter English Sentence..."
    translated = translator(text)
    print(f"Original Text: {text}")
    print(f"Translation Text: {translated[0]['translation_text']}\n")
    return translated[0]['translation_text']

# translator = load_pipeline(path)
# text = "Hello! How are you?"

# while text!='q':
#     inference_fn(text, translator)
#     text = input("Enter Text: ")
