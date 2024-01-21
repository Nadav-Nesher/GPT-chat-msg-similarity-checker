"""
message_role_filter.py

This module defines the MessageRoleFilter Enum class, used for specifying the role
of a message sender within the chat history. The enum facilitates filtering messages
based on the sender's role, ensuring consistency and type safety in role-related operations.

Classes:
    MessageRoleFilter (Enum): An enumeration that represents different roles associated
                              with chat messages.

Enum Members:
    SYSTEM_ROLE: Indicates a message sent by the system.
    USER_ROLE: Indicates a message sent by a user.
    ASSISTANT_ROLE: Indicates a message sent by an assistant or chatbot.

Example Usage:
    from message_role_filter import MessageRoleFilter

    # Example of using the enum for filtering
    role = MessageRoleFilter.USER_ROLE
    if message['role'] == role.value:
        # Process user messages
"""


from enum import Enum


class MessageRoleFilter(Enum):
    SYSTEM_ROLE = 'system'
    USER_ROLE = 'user'
    ASSISTANT_ROLE = 'assistant'
