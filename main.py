import openai
from typing import List, Dict, Optional

qa_data = [
    {
        "question": "How often should I feed my Golden Retriever?",
        "answer_openai": "Golden Retrievers should be fed according to their age and activity level..."
    },
    {
        "question": "Are there specific vitamins I should make sure are in my Golden Retriever’s diet?",
        "answer_openai": "For Golden Retrievers of all ages, it is important to ensure they are receiving the necessary vitamins..."
    },
    {
        "question": "What is the best dog food for a Golden Retriever?",
        "answer_openai": "For a Golden Retriever, it's important to consider their age when choosing the best dog food..."
    },
    {
        "question": "What is the best puppy food for Golden Retriever puppies?",
        "answer_openai": "For Golden Retriever puppies, it is important to choose a high-quality puppy food..."
    },
    {
        "question": "How do I calculate the proper portion size for my Golden Retriever?",
        "answer_openai": "When determining the proper portion size for your Golden Retriever, it is important to consider their age..."
    }
]

def calculate_word_match(query: str, text: str) -> int:
    """Calculates the number of matching words between the query and the text."""
    query_words = set(query.lower().split())
    text_words = set(text.lower().split())
    common_words = query_words.intersection(text_words)
    return len(common_words)

def find_best_match(query: str, qa_data: List[Dict[str, str]], threshold: int = 3) -> Optional[Dict[str, str]]:
    """Finds the best matching question based on word match count, with a relevance threshold."""
    best_match = None
    best_word_count = 0
    for qa in qa_data:
        word_count = calculate_word_match(query, qa['question'])
        if word_count > best_word_count and word_count >= threshold:
            best_word_count = word_count
            best_match = qa
    return best_match

def get_answer_from_openai(query: str) -> Dict[str, str]:
    """Fetches an answer from OpenAI if no sufficient match is found."""
    
    response = openai.ChatCompletion.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": query},
        ],
        max_tokens=150,
        n=1,
        stop=None,
        temperature=0.7,
    )
    
    return {
        "response_openai": response.choices[0].message['content']
    }

def handle_user_query(query: str, qa_data: List[Dict[str, str]], threshold: int = 5) -> Dict[str, str]:
    """Handles the user query by finding and returning the best matching answer."""
    best_match = find_best_match(query, qa_data, threshold)
    
    if best_match:
        return {"answer": best_match['answer_openai']}
    else:
        # If no sufficient match found, get the answer from OpenAI
        return get_answer_from_openai(query)

# Example usage
if __name__ == '__main__':
    user_query = "I feed my Golden Retriever?"
    answer = handle_user_query(user_query, qa_data)
    print(answer)
