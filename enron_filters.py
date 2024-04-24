import email

def is_valid_folder(filepath: str) -> bool:
    comp_generated_folders = {'all_documents', 'discussion_threads', 'sent_mail'}
    for gen_folder in comp_generated_folders:
        if gen_folder in filepath:
            return False
    return True


def is_enron_msg(email_str: str) -> bool:
    email_msg = email.message_from_string(email_str)
    if not email_msg.get("To"):
        return False

    if not email_msg.get("From").endswith("@enron.com"):
        return False

    all_recipients = email_msg.get("To").replace("\n\t", "").replace("\n", "").split(", ")
    enron_recipients = [recipient for recipient in all_recipients if recipient.endswith("@enron.com")]
    if not enron_recipients:
        return False

    return True
