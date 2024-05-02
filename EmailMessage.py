from typing import List
from datetime import datetime


class EmailMessage:
    def __init__(self, message_id: str, sender: str, recipients: List[int], time: datetime) -> None:
        self.id = message_id
        self.sender = sender
        self.recipients = recipients
        self.time = time

    def __str__(self) -> str:
        return f"Email {self.id} from {self.sender} to {self.recipients} at {self.time}"

    def __hash__(self) -> int:
        return hash(self.id)
