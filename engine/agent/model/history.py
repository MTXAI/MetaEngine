from typing import Union, List, Tuple, Dict

class History():

    def __init__(self, role: str, content: str):
        """
        Initialize a History instance with a role and content.

        Args:
            role (str): The role of the message (e.g., "user", "assistant").
            content (str): The content of the message.
        """
        self.role = role
        self.content = content

    def to_msg_str(self) -> str:
        """
        Convert the history entry to a string representation.

        Returns:
            str: A string formatted as "role: content".
        """
        return f"{self.role}: {self.content}"

    @classmethod
    def convert_histories_to_msg_str(cls, histories) -> str:
        """
        Convert a list of History instances to a single string.

        Args:
            histories (List[History]): A list of History instances.

        Returns:
            str: A string containing all history entries, each on a new line.
        """
        return "\n".join([h.to_msg_str() for h in histories])