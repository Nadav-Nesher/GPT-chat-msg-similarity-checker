"""
main.py

This module serves as the entry point for executing the NLP similarity analysis functionality.
It demonstrates the usage of the ChatMessagesSimilarity class by initializing it with a user
message and a chat history dataset, and then applying various similarity measurement approaches.

The module imports necessary components from other modules, creates an instance of
ChatMessagesSimilarity, and executes different methods to find the most similar sentences
in the chat history compared to the given user message. It covers lenient custom, professional,
and state-of-the-art approaches to similarity measurement.

Functions:
    flow(): Initializes and returns an instance of ChatMessagesSimilarity.

The script executes the flow function and then demonstrates the usage of each similarity
measurement method provided by the ChatMessagesSimilarity class, printing the results.

Example Usage:
    Run this script directly to see the output of different similarity measurement approaches
    on the predefined user message and chat history.
"""


from chat_messages_similarity import ChatMessagesSimilarity
from user_message import user_message
from chat_history import chat_message_history
from message_role_filter import MessageRoleFilter


def flow():
    chat_similarity_obj = ChatMessagesSimilarity(user_message=user_message,
                                             chat_history=chat_message_history,
                                             is_distinct=True,
                                             ignore_stopwords=True,
                                             is_case_sensitive=False,
                                             lemmatize=True,
                                             ignore_emojis=False
                                             )
    return chat_similarity_obj


if __name__ == '__main__':
    chat_similarity_obj = flow()

    overlap_most_similar_sentence = chat_similarity_obj.find_most_similar_sentence_using_common_words_overlap(
        role_filter=MessageRoleFilter.USER_ROLE)

    # If multiple sentences have the same num of max overlapping words (tokens) out of the entire chat history,
    # they are all saved to allow for a clearer understanding of common-word overlap. Hence, the for-loop.
    for sentence in overlap_most_similar_sentence:
        print(sentence)


    spacy_most_similar_sentence = chat_similarity_obj.find_most_similar_sentence_vector_using_spacy()
    print(spacy_most_similar_sentence)

    sentence_transformers_most_similar_sentence = chat_similarity_obj.find_most_similar_sentence_vector_using_sentence_transformers()
    print(sentence_transformers_most_similar_sentence)

    combined_similarity_results = chat_similarity_obj.find_most_similar_sentence_combined_approaches(
        role_filter=MessageRoleFilter.USER_ROLE)
    print(combined_similarity_results)