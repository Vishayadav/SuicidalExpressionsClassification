"""
Text preprocessing utilities for suicide expression classification
Handles emojis, special characters, and text normalization
"""
import re
import unicodedata

def remove_emojis(text: str) -> str:
    """
    Remove emoji characters from text while preserving meaning.
    
    Args:
        text: Input text with potential emojis
        
    Returns:
        Text with emojis removed
    """
    # Range of emoji characters in Unicode
    emoji_pattern = re.compile(
        "["
        "\U0001F600-\U0001F64F"  # Emoticons
        "\U0001F300-\U0001F5FF"  # Symbols & pictographs
        "\U0001F680-\U0001F6FF"  # Transport & map
        "\U0001F1E0-\U0001F1FF"  # Flags
        "\U00002702-\U000027B0"  # Dingbats
        "\U000024C2-\U0001F251"  # Enclosed characters
        "\U0001f926-\U0001f937"  # People
        "\U00010000-\U0010ffff"  # Supplementary Multilingual Plane
        "\u2640-\u2642"          # Gender symbols
        "\u2600-\u2B55"          # Miscellaneous symbols
        "\u200d"                 # Zero width joiner
        "\u23cf"                 # Play symbol
        "\u23e9"                 # Fast forward
        "\u231a"                 # Watch
        "\ufe0f"                 # Dingbats
        "\u3030"                 # Wavy dash
        "]+",
        flags=re.UNICODE
    )
    
    text = emoji_pattern.sub(r'', text)
    return text


def clean_text(text: str) -> str:
    """
    Clean and preprocess text for model prediction.
    
    Args:
        text: Raw input text
        
    Returns:
        Cleaned text suitable for model input
    """
    if not isinstance(text, str):
        text = str(text)
    
    # Step 1: Remove emojis
    text = remove_emojis(text)
    
    # Step 2: Remove extra whitespace
    text = ' '.join(text.split())
    
    # Step 3: Remove URL (if any)
    text = re.sub(r'http\S+|www.\S+', '', text)
    
    # Step 4: Handle multiple punctuation
    text = re.sub(r'([!?])\1+', r'\1', text)
    
    return text.strip()


def preprocess_for_model(texts: list) -> tuple:
    """
    Preprocess a list of texts for model prediction.
    
    Args:
        texts: List of raw texts
        
    Returns:
        Tuple of (cleaned_texts, emoji_map) where emoji_map shows original vs cleaned
    """
    cleaned = []
    emoji_info = []
    
    for text in texts:
        original = text
        cleaned_text = clean_text(text)
        cleaned.append(cleaned_text)
        
        emoji_info.append({
            'original': original,
            'cleaned': cleaned_text,
            'had_emoji': original != cleaned_text
        })
    
    return cleaned, emoji_info


if __name__ == "__main__":
    # Test the preprocessing
    test_texts = [
        "We love you NASA 💙🌊",
        "Hello so beautiful 💙💙",
        "I want to die 😢",
        "Great day! 😊😊",
        "Visit https://example.com now!!!",
    ]
    
    print("Testing Text Preprocessing\n" + "="*70)
    
    for text in test_texts:
        cleaned = clean_text(text)
        had_emoji = text != cleaned
        
        print(f"\nOriginal: {text}")
        print(f"Cleaned:  {cleaned}")
        print(f"Emoji removed: {had_emoji}")
