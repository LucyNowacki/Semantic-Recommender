import pandas as pd
import re
from html import unescape

class AbstractProcessor:
    def __init__(self):
        pass

    def clean_text(self, text):
        # Replace newline characters with spaces and unescape HTML entities
        text = text.replace('\n', ' ')
        text = unescape(text)
        # Replace math expressions and reduce whitespace
        text = self.replace_math_expressions(text)
        text = re.sub(r'\s+', ' ', text).strip()
        # Remove specific unwanted patterns excluding [MATH_EXPR]
        #text = re.sub(r'\[[^\]]+(?<!\[MATH_EXPR\)])(?!\[MATH_EXPR\])', '', text)  # Remove patterns like '[0,1)' except within [MATH_EXPR]
        return text

    def replace_math_expressions(self, text):
        # Patterns for various math expressions, each could potentially have a unique placeholder
        patterns = [
            r'\$.*?\$',  # Inline math expressions
            r'\\[a-zA-Z]+\{.*?\}',  # LaTeX commands
            r'\w+_{[^}]+}',  # Subscripts
            r'\w+\'',  # Primes
            r'\w+_{[^}]+}\'?',  # Subscripts followed by a prime
            r'\w+^{-?\d+}',  # Superscripts
            r'\{[^}]+\}',  # Expressions in braces
            r'\[[^\]]+\]',  # Expressions in square brackets
            r'\^',  # Caret symbol, often used for superscripts outside LaTeX
            r'\[[^\]]+(?<!\[MATH_EXPR\)])(?!\[MATH_EXPR\])'
        ]
        replacement = ' [MATH_EXPR] '  # Unified placeholder
        for pattern in patterns:
            text = re.sub(pattern, replacement, text)
        # Consolidate repeated placeholders into a single instance
        text = re.sub(r'(\s*\[MATH_EXPR\]\s*)+', ' [MATH_EXPR] ', text)
        return text

    def process(self, series, func=None):
        """
        Applies a specified cleaning function to each entry in a pandas Series.
        If no function is specified, the clean_text method is used by default.
        """
        if func:
            return series.apply(func)
        else:
            return series.apply(self.clean_text)
        
    # Define the function to find special signs
    def find_special_signs(self, col, patterns):

        special_signs = set()  # Set to hold all unique special signs
        for text in col:
            for pattern in patterns:
                matches = re.findall(pattern, text)
                special_signs.update(matches)
        return special_signs
    

from sklearn.preprocessing import MultiLabelBinarizer

def multi_bin(df):
    # Flatten the list of lists in 'terms' column to get all labels in a single list
    all_labels = [label for sublist in df['terms'] for label in sublist]
    
    # # Count the frequency of each unique label and sort them by frequency in descending order
    # label_counts = Counter(all_labels)
    # sorted_label_counts = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
    # print("Frequency of each label in 'terms' (sorted by frequency):")
    # for label, count in sorted_label_counts:
    #     print(f"{label}: {count}")
    
    # Extract unique labels (optional, as MultiLabelBinarizer does this internally)
    unique_labels = set(all_labels)
    print(f"\nUnique labels in 'terms': {unique_labels}")
    print(f"\nNumber of unique labels: {len(unique_labels)}")
    
    mlb = MultiLabelBinarizer()
    encoded_terms = mlb.fit_transform(df['terms'])
    terms_df = pd.DataFrame(encoded_terms, columns=mlb.classes_)
    
    df_ready_encoded = pd.concat([df.reset_index(drop=True), terms_df.reset_index(drop=True)], axis=1)
    
    print("\n", df_ready_encoded.head())
    print("Number of unique columns (including original ones):", df_ready_encoded.shape[1])

    return df_ready_encoded
