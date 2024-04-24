class Person:
    def __init__(self, email: str) -> None:
        self.email = email
        self.received = dict()
        self.sent = dict()
        self.recipients = set()

    def __hash__(self) -> int:
        return hash(self.email)