# main.py
from transformers import pipeline

def translate_text(text_to_translate, source_lang='en', target_lang='sw'):
    """
    Translates text from a source language to a target language using a pre-trained Hugging Face model.
    This example focuses on English to Swahili, a key language mentioned in the article.
    """
    # The article discusses AI translation in Kenyan languages like Swahili.
    # We use a pre-trained model for English-Swahili translation.
    # This model (Helsinki-NLP/opus-mt-en-sw) is specifically designed for this language pair.
    model_name = f"Helsinki-NLP/opus-mt-{source_lang}-{target_lang}"
    
    try:
        # Initialize the translation pipeline. The model will be downloaded on first run.
        translator = pipeline("translation", model=model_name)
        
        # Perform the translation
        # This step demonstrates the core AI translation concept discussed in the article.
        result = translator(text_to_translate)
        
        return result[0]['translation_text']
    except Exception as e:
        return f"Error during translation: {e}. Make sure you have 'transformers' and 'torch' installed."

if __name__ == "__main__":
    print("--- AI Translation Demo for Kenyan Languages (Swahili) ---")
    print("This script demonstrates AI translation, a core concept of the article.")
    print("It uses a pre-trained model to translate English to Swahili.")
    print("-" * 50)

    # Sample text in English
    english_text = "Artificial intelligence translation is becoming increasingly important."
    print(f"Original English Text: {english_text}")

    # Translate to Swahili
    swahili_translation = translate_text(english_text, source_lang='en', target_lang='sw')
    print(f"Translated Swahili Text: {swahili_translation}")
    print("-" * 50)

    # Another example
    english_text_2 = "Language barriers can still pose a significant obstacle."
    print(f"Original English Text: {english_text_2}")
    swahili_translation_2 = translate_text(english_text_2, source_lang='en', target_lang='sw')
    print(f"Translated Swahili Text: {swahili_translation_2}")
    print("-" * 50)

    # Note: For other Kenyan languages mentioned (Kikuyu, Luo, Kamba, Maasai),
    # specific models would be needed, and their availability/performance
    # can vary, which is precisely the challenge discussed in the article.
    print("Note: The article discusses challenges for diverse languages.")
    print("While this demo uses Swahili (which has good model support),")
    print("other Kenyan languages may have fewer readily available models,")
    print("highlighting the research areas mentioned in the article.")
