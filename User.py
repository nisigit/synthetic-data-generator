class User:
    def __init__(self, uuid: int) -> None:
        self.uuid = uuid
        self.received = dict()
        self.sent = dict()
        self.recipients = set()

    def __hash__(self) -> int:
        return hash(self.uuid)