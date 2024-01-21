"""
user_message.py

This module contains a single dictionary named 'user_message'. It holds the details
of a message from a user, which is intended to be compared against a dataset of chat
history messages in similarity NLP tasks.

Attributes:
    user_message (Dict[str, str]): A dictionary representing a single user message.
                                   It contains two key-value pairs:
                                   'role' (str): Indicates the role of the message sender.
                                                 In this case, it's set to 'user'.
                                   'content' (str): The actual text content of the message.
                                                    This is the message text that will
                                                    be used for NLP similarity comparisons.
"""


user_message = {
    'role': 'user', 'content': "A total of 2364? Does it include the Librot (bonuses) I've accumulated?"
}