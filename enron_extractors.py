import email


def extract_sender(email_str: str) -> str:
    email_msg = email.message_from_string(email_str)
    sender = email_msg.get("From")
    if not sender:
        raise ValueError("No sender found")

    return sender


def extract_recipients(email_str: str) -> list[str]:
    email_msg = email.message_from_string(email_str)
    all_recipients = email_msg.get("To", "") \
        .replace("\n\t", "") \
        .replace("\n", "") \
        .split(", ")
    enron_recipients = {str(recipient) for recipient in all_recipients if recipient.endswith("@enron.com")}
    return list(enron_recipients)


def extract_date(email_str: str) -> str:
    email_msg = email.message_from_string(email_str)
    date_str = "-".join(email_msg.get("Date").split("-")[:-1])
    return email.utils.parsedate_to_datetime(date_str)


def extract_subject(email_str: str) -> str:
    email_msg = email.message_from_string(email_str)
    return email_msg.get("Subject")


def extract_message_id(email_str: str) -> str:
    email_msg = email.message_from_string(email_str)
    return email_msg.get("Message-ID")
