"""
chat_messages_similarity.py

This module encompasses the core functionality for Natural Language Processing (NLP) tasks
related to analyzing chat messages. It provides the `ChatMessagesSimilarity` class, which
offers several methods for computing similarity between a user message and a chat history.
The class leverages various NLP techniques ranging from simple word overlap to advanced
sentence embeddings for assessing message similarity.

The module includes methods for preprocessing text, extracting emojis and words, and
calculating similarity using different approaches: a lenient custom approach based on
common words overlap, a professional approach using spaCy's vector model, and a state-of-the-art
(SOTA) professional approach using sentence transformers.

Classes:
    ChatMessagesSimilarity: A class that encapsulates the logic for comparing a user message
                            against a chat history dataset using various NLP methods.

Dependencies:
    - re
    - typing (Dict, List, Set, Union, Optional)
    - collections (Counter)
    - emoji
    - spacy
    - sentence_transformers (SentenceTransformer, util)
    - message_role_filter (MessageRoleFilter)
"""


import re
from typing import Dict, List, Set, Union, Optional
from collections import Counter
import emoji
import spacy
from sentence_transformers import SentenceTransformer, util

from message_role_filter import MessageRoleFilter


model = SentenceTransformer('all-MiniLM-L6-v2')
nlp = spacy.load('en_core_web_lg')


class ChatMessagesSimilarity:
    """
        A class that encapsulates various Natural Language Processing (NLP) methods
        for comparing a user message with messages in a chat history dataset.

        The class provides functionality for preprocessing text, extracting significant
        elements like emojis and words, and applying different strategies to compute
        the similarity between messages. These strategies range from simple word overlap
        to advanced techniques using spaCy's vector model and sentence-transformers.

        Attributes:
            user_message (Dict[str, str]): A dictionary containing the user's message with 'role' and 'content'.
            chat_history (List[Dict[str, str]]): A list of dictionaries, each representing a chat message.
            is_distinct (bool): Flag to determine if repeated words/emojis should be considered distinct.
            ignore_stopwords (bool): Flag to specify whether to ignore stopwords in text processing.
            is_case_sensitive (bool): Flag to indicate if the text processing should be case-sensitive.
            lemmatize (bool): Flag to specify whether to apply lemmatization in text processing.
            ignore_emojis (bool): Flag to specify whether to ignore emojis in text processing.

        Methods:
            handle_surrogate_pairs(text): Handles encoding errors due to surrogate pairs in text.
            replace_multiple_spaces_and_trim_end(text): Replaces multiple spaces with a single space and trims ends.
            preprocess_text(text): Combines preprocessing techniques of surrogate pairs and spaces to the given text.
            extract_emojis(message): Extracts emojis from a given message.
            extract_words(message): Extracts words from a given message, considering various flags.
            extract_emojis_and_words(message): Combines the extraction of emojis and words.
            find_most_similar_sentence_using_common_words_overlap(role_filter): Finds the most similar sentence based on common word overlap.
            find_most_similar_sentence_vector_using_spacy(role_filter): Utilizes spaCy's vector model for sentence similarity.
            find_most_similar_sentence_vector_using_sentence_transformers(role_filter): Applies sentence transformers for sentence similarity.
            find_most_similar_sentence_combined_approaches(role_filter): Combines all approaches to find the most similar sentence.
        """

    def __init__(self,
                 user_message,
                 chat_history,
                 is_distinct=True,
                 ignore_stopwords=True,
                 is_case_sensitive=False,
                 lemmatize=True,
                 ignore_emojis=False):

        self.user_message: Dict[str, str] = user_message
        self.chat_history: List[Dict[str, str]] = chat_history
        self.is_distinct: bool = is_distinct
        self.ignore_stopwords: bool = ignore_stopwords
        self.is_case_sensitive: bool = is_case_sensitive
        self.lemmatize: bool = lemmatize
        self.ignore_emojis: bool = ignore_emojis


    @staticmethod
    def handle_surrogate_pairs(text: str) -> str:
        """
            Handles surrogate pairs in a text string by encoding and then decoding the text.

            This method is useful for processing text that contains surrogate pairs,
            which can cause encoding errors in certain environments. By encoding to UTF-16
            and decoding back to UTF-8, surrogate pairs are correctly handled.

            Parameters:
                text (str): The text string that may contain surrogate pairs.

            Returns:
                str: The processed text with surrogate pairs correctly encoded.
            """

        return text.encode(encoding='utf-16', errors='surrogatepass').decode('utf-16')

    @staticmethod
    def replace_multiple_spaces_and_trim_end(text: str) -> str:
        """
            Processes a given text string by replacing multiple consecutive spaces
            with a single space and trimming spaces at the end of sentences.

            This method enhances text readability and consistency, which is crucial for
            subsequent NLP tasks. It first reduces any instances of multiple spaces to
            just one space, and then trims trailing spaces at the ends of sentences.

            Parameters:
                text (str): The text string to be processed.

            Returns:
                str: The text after replacing multiple spaces with a single space and
                     trimming spaces at the end.
            """

        text = re.sub(pattern='\s+', repl=' ', string=text)
        text = re.sub(pattern=r'\s+([?!]+)', repl=r'\1', string=text)
        return text.strip()

    def preprocess_text(self, text: str) -> str:
        """
            Applies a series of preprocessing steps to the given text string.

            This method is a pipeline of text preprocessing that includes handling surrogate pairs
            and replacing multiple spaces with a single space. The purpose is to standardize the text
            for better performance in NLP tasks by ensuring consistent encoding and whitespace usage.

            Parameters:
                text (str): The text string to be preprocessed.

            Returns:
                str: The preprocessed text after applying all preprocessing steps.
            """

        text = self.handle_surrogate_pairs(text)
        text = self.replace_multiple_spaces_and_trim_end(text)
        return text

    def extract_emojis(self, message: str) -> List[str]:
        """
            Extracts emojis from a given text message.

            This method identifies and extracts all emojis present in the input text.
            If the 'is_distinct' attribute is set to True, only distinct emojis are extracted.
            It is useful for analyzing the embedded emotional or expressive content of the messages.

            Parameters:
                message (str): The text message from which emojis are to be extracted.

            Returns:
                List[str]: A list of extracted emojis from the message. If 'is_distinct' is True,
                           only unique emojis are returned.
            """

        distinct_emojis = None

        # Conditionally find distinct emojis
        if self.is_distinct:
            distinct_emojis = emoji.distinct_emoji_list(message)

        # Choose input for `emoji.demojize()` func based on the `is_distinct` param
        input_text = distinct_emojis if self.is_distinct and distinct_emojis else message

        # Demojize the text
        demojized_text = emoji.demojize(string=input_text, delimiters=(" [emoji:", "] "))

        # Extract emojis from demojized text
        demojized_emojis = [token for token in demojized_text.split() if token.startswith('[') and token.endswith(']')]

        return demojized_emojis

    def extract_words(self, message: str) -> Union[List[str], Set[str]]:
        """
            Extracts words from a given text message, considering various text processing flags.

            This method processes the input message through spaCy's NLP pipeline to extract words.
            It respects the class attributes like 'ignore_stopwords', 'is_case_sensitive', and 'lemmatize'
            to filter and process the words accordingly. The method can return either a list of all words
            or a set of distinct words based on the 'is_distinct' flag.

            Parameters:
                message (str): The text message from which words are to be extracted.

            Returns:
                Union[List[str], Set[str]]: A list or set of extracted words from the message.
                                            The return type depends on the 'is_distinct' flag.
            """

        # Process message through spaCy's NLP pipeline to produce a spaCy doc (document) object
        doc = nlp(message)

        words = [(token.lemma_ if self.lemmatize else token.text).lower() if not self.is_case_sensitive else (token.lemma_ if self.lemmatize else token.text)
                 for token in doc
                 if (not token.is_stop or not self.ignore_stopwords) and not token.is_punct and not emoji.is_emoji(token.text)]

        if self.is_distinct:
            return set(words)
        return words

    def extract_emojis_and_words(self, message: str) -> List[str]:
        """
            Combines the extraction of emojis and words from a given text message.

            This method first checks for and extracts any emojis in the message, considering the
            'ignore_emojis' flag. It then proceeds to extract words, ensuring that purely emoji
            messages are not processed for word extraction. The result is a combined list of
            both emojis and words extracted from the message.

            Parameters:
                message (str): The text message from which emojis and words are to be extracted.

            Returns:
                List[str]: A combined list of extracted emojis and words from the message.
            """

        combined_list = []

        # Check for emojis and extract if needed
        if not self.ignore_emojis and emoji.emoji_count(message) > 0:
            combined_list.extend(self.extract_emojis(message))

        # Check if the message is not purely emojis before extracting words
        if not emoji.purely_emoji(message):
            combined_list.extend(self.extract_words(message))

        return combined_list

    def find_most_similar_sentence_using_common_words_overlap(self, role_filter: Optional[MessageRoleFilter] = None) -> List[Dict[str, Union[str, int, List[str]]]]:
        """
            Finds the most similar sentence(s) in the chat history to the user's message based on
            the overlap of common words and emojis.

            This method preprocesses both the user's message and each message in the chat history
            for a fair comparison. It then calculates the number of common elements (words and emojis)
            between the user's message and each chat history message. The method returns all messages
            that share the highest number of common elements, considering the 'is_distinct' flag for
            uniqueness of words and emojis.

            Parameters:
                role_filter (Optional[MessageRoleFilter]): A filter to select messages based on the role
                                                           ('user', 'assistant', etc.). If None, all messages are considered.

            Returns:
                List[Dict[str, Union[str, int, List[str]]]]: A list of dictionaries, each representing a chat message
                                                             with the highest overlap. Each dictionary contains details
                                                             like the role, content, number of common elements, and the
                                                             actual common elements.
            """

        # Preprocess the message for whitespaces and encoding errors
        preprocessed_user_message = self.preprocess_text(text=self.user_message.get('content', ''))

        user_message_processed_for_emoji_and_word_extraction = self.extract_emojis_and_words(message=preprocessed_user_message)
        user_message_counter = Counter(user_message_processed_for_emoji_and_word_extraction) if not self.is_distinct else Counter(set(user_message_processed_for_emoji_and_word_extraction))
        max_similarity = 0
        similar_sentences = []

        for message in self.chat_history:
            if role_filter and message.get('role', '') != role_filter.value:
                continue

            # Preprocess the message for whitespaces and encoding errors
            preprocessed_chat_history_message = self.preprocess_text(text=message.get('content', ''))

            chat_history_message_processed_for_emoji_and_word_extraction = self.extract_emojis_and_words(message=preprocessed_chat_history_message)
            chat_message_counter = Counter(chat_history_message_processed_for_emoji_and_word_extraction) if not self.is_distinct else Counter(set(chat_history_message_processed_for_emoji_and_word_extraction))
            common_elements = user_message_counter & chat_message_counter  # Intersection
            common_count = sum(common_elements.values())

            if common_count > max_similarity:
                max_similarity = common_count
                similar_sentences = [{
                    'scoring_method': 'overlapping_words',
                    'role': message.get('role', ''),
                    'content': preprocessed_chat_history_message,
                    'num_of_common_elements': common_count,
                    'common_elements': list(common_elements)
                }]
            elif common_count == max_similarity:
                similar_sentences.append({
                    'role': message.get('role', ''),
                    'content': preprocessed_chat_history_message,
                    'num_of_common_elements': common_count,
                    'common_elements': list(common_elements)
                })

        return similar_sentences

    def find_most_similar_sentence_vector_using_spacy(self, role_filter: Optional[MessageRoleFilter] = None) -> Dict[str, Union[str, float]]:
        """
            Determines the most similar sentence in the chat history to the user's message
            using spaCy's vector model for semantic similarity.

            This method preprocesses the user's message and each message in the chat history,
            then uses spaCy's NLP model to convert them into semantic vectors. It calculates
            the cosine similarity between the user's message vector and each chat message vector.
            The method returns the chat message with the highest cosine similarity score.

            Parameters:
                role_filter (Optional[MessageRoleFilter]): A filter to select messages based on the role
                                                           ('user', 'assistant', etc.). If None, all messages are considered.

            Returns:
                Dict[str, Union[str, float]]: A dictionary containing details of the most similar chat message.
                                              Includes the role, content, and the cosine similarity score.
            """

        # Preprocess the message for whitespaces and encoding errors
        preprocessed_user_message = self.preprocess_text(text=self.user_message.get('content', ''))

        user_doc = nlp(preprocessed_user_message)
        max_cosine_similarity = 0
        most_similar_sentence = {}

        for message in self.chat_history:
            if role_filter and message.get('role', '') != role_filter.value:
                continue

            # Preprocess the message for whitespaces and encoding errors
            preprocessed_chat_history_message = self.preprocess_text(text=message.get('content', ''))

            chat_doc = nlp(preprocessed_chat_history_message)
            chat_and_user_cosine_similarity = chat_doc.similarity(user_doc)

            if chat_and_user_cosine_similarity > max_cosine_similarity:
                max_cosine_similarity = chat_and_user_cosine_similarity
                most_similar_sentence = {
                    'scoring_method': 'spacy_model',
                    'role': message.get('role', ''),
                    'content': preprocessed_chat_history_message,
                    'cosine_similarity_score': chat_and_user_cosine_similarity,
                }

        return most_similar_sentence

    def find_most_similar_sentence_vector_using_sentence_transformers(self, role_filter: Optional[MessageRoleFilter] = None) -> Dict[str, Union[str, float]]:
        """
            Identifies the most similar sentence in the chat history to the user's message
            using state-of-the-art sentence-transformers library for semantic similarity.

            This method applies sentence transformers to both the user's message and each
            chat history message, generating embeddings that capture deep semantic meanings.
            It computes the cosine similarity between these embeddings to find the chat message
            that is semantically closest to the user's message. The method returns the chat
            message with the highest cosine similarity score.

            Parameters:
                role_filter (Optional[MessageRoleFilter]): A filter to select messages based on the role
                                                           ('user', 'assistant', etc.). If None, all messages are considered.

            Returns:
                Dict[str, Union[str, float]]: A dictionary containing the details of the most similar chat message.
                                              Includes the role, content, and the cosine similarity score.
            """

        # Preprocess the message for whitespaces and encoding errors
        preprocessed_user_message = self.preprocess_text(text=self.user_message.get('content', ''))

        user_input_raw_text = [preprocessed_user_message]
        user_input_embedding = model.encode(sentences=user_input_raw_text, convert_to_tensor=True)
        max_cosine_similarity = 0
        most_similar_sentence = {}

        for message in self.chat_history:
            if role_filter and message.get('role', '') != role_filter.value:
                continue

            # Preprocess the message for whitespaces and encoding errors
            preprocessed_chat_history_message = self.preprocess_text(text=message.get('content', ''))

            chat_raw_text = [preprocessed_chat_history_message]
            chat_embeddings = model.encode(sentences=chat_raw_text, convert_to_tensor=True)

            # Compute cosine-similarity
            chat_and_user_cosine_similarity_score_tensor = util.cos_sim(user_input_embedding, chat_embeddings)
            chat_and_user_cosine_similarity_score = chat_and_user_cosine_similarity_score_tensor[0, 0].item()

            if chat_and_user_cosine_similarity_score > max_cosine_similarity:
                max_cosine_similarity = chat_and_user_cosine_similarity_score
                most_similar_sentence = {
                    'scoring_method': 'sentence-transformers',
                    'role': message.get('role', ''),
                    'content': preprocessed_chat_history_message,
                    'cosine_similarity_score': chat_and_user_cosine_similarity_score,
                }

        return most_similar_sentence

    def find_most_similar_sentence_combined_approaches(self, role_filter: Optional[MessageRoleFilter] = None) -> Dict[
        str, Union[List[Dict[str, Union[str, int, List[str]]]], Dict[str, Union[str, float]]]]:
        """
            Combines multiple approaches to find the most similar sentence in the chat history
            to the user's message. This method provides a comprehensive analysis by utilizing
            three different similarity measurement approaches.

            The method applies the following approaches:
            1. A lenient custom approach based on common words overlap.
            2. A professional approach using spaCy's vector model.
            3. A state-of-the-art (SOTA) professional approach using sentence-transformers.

            Each approach contributes a unique perspective on text similarity, offering a
            robust analysis of which chat message is most similar to the user's message.

            Parameters:
                role_filter (Optional[MessageRoleFilter]): A filter to select messages based on the role
                                                           ('user', 'assistant', etc.). If None, all messages are considered.

            Returns:
                Dict[str, Union[List[Dict[str, Union[str, int, List[str]]]], Dict[str, Union[str, float]]]]:
                A dictionary containing the results from all three approaches. Each key in the dictionary
                corresponds to an approach, with the value being the result from that specific method.
            """

        result = {
            'lenient_custom_approach': self.find_most_similar_sentence_using_common_words_overlap(role_filter),
            'professional_approach': self.find_most_similar_sentence_vector_using_spacy(role_filter),
            'SOTA_professional_approach': self.find_most_similar_sentence_vector_using_sentence_transformers(role_filter)
        }
        return result





