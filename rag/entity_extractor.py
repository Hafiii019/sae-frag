def extract_entities(text):

    medical_terms = [
        "pneumothorax",
        "effusion",
        "cardiomegaly",
        "atelectasis",
        "nodule",
        "opacity",
        "consolidation"
    ]

    found = []

    text = text.lower()

    for term in medical_terms:
        if term in text:
            found.append(term)

    return set(found)