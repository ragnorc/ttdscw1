#%%
from collections import defaultdict
from lark import Lark, tree, Transformer, v_args
import xmltodict 
import re
import nltk


def tokenize(str):
    return re.findall("[A-Z]{2,}(?![a-z])|[A-Z][a-z]+(?=[A-Z])|[\w][\w\'\-]*", str)

def remove_stop_words(words):
    stop_words = set(line.strip() for line in open('stop_words.txt'))
    return [word for word in words if word not in stop_words]


@v_args(inline=True)
class QueryParser(Transformer):
    def __init__(self, index):
        super().__init__(visit_tokens=True)
        self.index = index

    def __default_token__(self, token):
        if token.type == 'TERM':
            return set(self.index.search(token.value).keys())

        if token.type == 'PHRASE':
            matches = set()
            terms = token.value.replace('"','').split()
            results = [self.index.search(terms[0]), self.index.search(terms[1])]
            intersection = set(results[0].keys()).intersection(results[1].keys())
            for docID in intersection:
                for pos in results[0][docID]:
                    if pos+1 in results[1][docID]:
                        matches.add(docID)
                            
                #print("matches:")
                #print(matches)
            return matches
        else:
            return token.value

    def negation(self, results):
        return self.index.docIDs.difference(set(results))


    def boolean(self, left, operator, right):
        print(operator)
        if operator == "OR":
            return left.union(right)
        elif operator == "AND":
            return left.intersection(right)
        
    def query(self, value):
        return value



class Index():

        def __init__(self, source='collections/trec.sample.xml'):

            self.index = defaultdict(lambda: defaultdict(list))
            self.docIDs = set()
            with open(source, 'r') as file:
                xml = file.read()

            documents = xmltodict.parse(xml)["document"]["DOC"]
            print(len(documents))
            for document in documents:
                self.docIDs.add(document["DOCNO"])
                # Tokenize the document
                tokens = [token.lower() for section in ["HEADLINE","TEXT","PUB","PAGE"] for token in tokenize(document[section])]
                #print(tokens)
                # Remove stop words
                filtered_tokens = remove_stop_words(tokens)

                stems = [nltk.stem.PorterStemmer().stem(token) for token in filtered_tokens ]

                for i, stem in enumerate(stems):
                    self.index[stem][document["DOCNO"]].append(i+1) 
        

        def search(self, term):
            print(nltk.stem.PorterStemmer().stem(term))
            return self.index[nltk.stem.PorterStemmer().stem(term)]


        def write_to_file(self, file="index.txt"):

            fout = open(file, "w")

            for term, documents in sorted(self.index.items()):
                fout.write(term + ':'+ str(len(documents)) + '\n')
                for docID, positions in documents.items():
                    fout.write('\t' + docID + ': '+ ','.join(map(str, positions))  + '\n')

            fout.close()





# %%
with open('queries.boolean.txt', 'r') as file:
        queries = file.readlines()
        for query in queries:
            print(query)
            print(re.findall('\"[\s\w]+\"|[\w]+', query))

# %%






class QueryEngine():

    def __init__(self):
        grammar = """
        query: PHRASE | TERM | boolean | negation
        boolean: query OPERATOR query
        PHRASE: /\"[\s\w]+\"/
        TERM: /[\w]+/
        OPERATOR: "OR" | "AND"
        negation: "NOT" query
        %import common.WS
        %ignore WS
        """
        self.parser = Lark(grammar, start='query')
    
    
    def query(self, index, q):
        processor = QueryParser(index)
        return processor.transform(self.parser.parse(q))


# %%
len(QueryEngine().query(index, "NOT dencora"))
# %%
