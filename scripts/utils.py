import re

def to_screaming_snake_case(text: str) -> str:
    """
    Converts a given string to SCREAMING_SNAKE_CASE.
    """
    if not text:
        return ""
    
    # Handle CamelCase / camelCase
    # Insert underscore between lower or number to upper
    s1 = re.sub(r'([a-z0-9])([A-Z])', r'\1_\2', text)
    
    # Insert underscore before a sequence of uppercase letters followed by a lowercase letter
    s2 = re.sub(r'([A-Z]+)([A-Z][a-z])', r'\1_\2', s1)
    
    # Replace non-alphanumeric characters with underscores
    s3 = re.sub(r'[^a-zA-Z0-9]', '_', s2)
    
    # Replace multiple underscores with a single underscore and strip leading/trailing
    s4 = re.sub(r'_+', '_', s3).strip('_')
    
    return s4.upper()
