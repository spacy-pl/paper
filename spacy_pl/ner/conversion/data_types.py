class Token:
    def __init__(self, orth, attribs, id):
        self.orth = orth
        self.attribs = attribs
        self.id = id

    def is_NE(self):
        return self.get_NE() is not None and self.get_NE() != "O"

    def get_NE(self):
        for attrib in self.attribs:
            for k in attrib:
                if attrib[k] != "0":
                    return k

        return None

    def __str__(self):
        return self.orth + ":" + str(self.attribs)

    @classmethod
    def from_json(cls, json):
        # attribs = [{'converted': [{json['ner': '1']}]}]
        return cls(json['orth'], json['ner'], json['id'])

    def to_json(self):
        return {
            'orth': self.orth,
            'id': self.id,
            'ner': self.attribs
        }


class Sentence:
    def __init__(self, tokens=None):
        self.tokens = tokens if tokens is not None else []

    def add(self, token):
        self.tokens.append(token)

    def to_json(self):
        return {'tokens': [t.to_json()
                           for t in self.tokens
                           ], 'brackets': []
                }

    @classmethod
    def from_json(cls, json):
        if 'tokens' not in json:
            return cls()
        tokens = [Token.from_json(tok) for tok in json['tokens']]
        return cls(tokens)


class Paragraph:
    def __init__(self, sentences=None):
        self.sentences = sentences if sentences is not None else []

    def add(self, sentence):
        self.sentences.append(sentence)

    def to_json(self):
        return {'sentences': [sentence.to_json() for sentence in self.sentences]}

    @classmethod
    def from_json(cls, json):
        if 'sentences' not in json:
            return cls()
        sentences = [Sentence.from_json(sent_json) for sent_json in json['sentences']]
        return cls(sentences)


class Document:
    def __init__(self, id, paragraphs=None):
        self.id = id
        self.paragraphs = paragraphs if paragraphs is not None else []

    def add(self, paragraph):
        self.paragraphs.append(paragraph)

    def to_json(self):
        return {'id': self.id,
                'paragraphs': [p.to_json() for p in self.paragraphs]}

    @classmethod
    def from_json(cls, json):
        if 'id' not in json or 'paragraphs' not in json:
            return cls()

        paragraphs = [Paragraph.from_json(paragraph_json) for paragraph_json in json['paragraphs']]
        return cls(json['id'], paragraphs)


class Corpus:
    def __init__(self, documents=None):
        self.documents = documents if documents is not None else []

    def add(self, document):
        self.documents.append(document)

    def to_json(self):
        return [doc.to_json() for doc in self.documents]

    @classmethod
    def from_json(cls, json):
        documents = [Document.from_json(doc_json) for doc_json in json]
        return cls(documents)


class TokenSpacy(Token):
    def to_json(self):
        return {
            'orth': self.orth,
            'id': self.id,
            'ner': self.get_NE()
        }
