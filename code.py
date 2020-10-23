#%%
from collections import defaultdict
from lark import Lark, tree, Transformer, v_args
import xmltodict 
import re
import nltk
import math


def tokenize(str):
    return [token.lstrip("'").rstrip("'") for token in re.findall("[\d]+[\.]?[\d]+|[\w][\w\'\-]*", str)]

def remove_stop_words(words):
    stop_words = set(line.strip() for line in open('stop_words.txt'))
    return [word for word in words if word not in stop_words]


@v_args(inline=True)
class QueryEvaluator(Transformer):
    """
    This class defines the query evaluator. It receives a syntax tree of the query and 
    evaluates the leafs in a bottom up fashion.
    """

    def __init__(self, index):
        super().__init__(visit_tokens=True)
        self.index = index

    def __default_token__(self, token):
        if token.type == 'TERM':
            return self.index.search(token.value)

        if token.type == 'PHRASE':
            matches = set()
            terms = token.value.replace('"','').split()
            results = [self.index.search(term) for term in terms]
            intersection =  set.intersection(*[set(result.keys()) for result in results])
            for docID in intersection:
                print(docID)
                for pos in results[0][docID]:
                    if all([pos+i in results[i][docID] for i in range(1, len(terms))]):
                        matches.add(docID)
                        break
            return matches
        else:
            return token.value

    def negation(self, results):
        return self.index.docIDs.difference(set(results))

    def proximity(self, num, results1, results2):
        matches = set()
        intersection =  set(results1.keys()).intersection(set(results2.keys()))
        for docID in intersection:
            if len([abs(int(i)-int(j)) <= int(num) for i in results1[docID] for j in results2[docID]]) > 0:
                matches.add(docID)
        return matches

    def boolean(self, left, operator, right):
        if operator == "OR":
            return left.union(right)
        elif operator == "AND":
            return left.intersection(right)
        
    def query(self, results):
        if isinstance(results, dict):
            return set(results.keys())
        return results



class Index():

        def __init__(self, source='tre.5000.xml'):

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

class QueryEngine():
    """
    Class that defines the grammar of the possible queries and instantiates a parser. 
    The query methods pass the query string to the parser and then evaluates the returned syntax tree.
    """

    def __init__(self):
        grammar = """
        query: PHRASE | TERM | boolean | negation | proximity
        proximity: "#"NUM"("TERM","TERM")"
        boolean: query OPERATOR query
        NUM: /[\d]+/
        PHRASE: /\"[\s\w]+\"/
        TERM: /[\w][\w\-\']*/
        OPERATOR: "OR" | "AND"
        negation: "NOT" query
        %import common.WS
        %ignore WS
        """
        self.parser = Lark(grammar, start='query')
    
    
    def boolean_query(self, index, q):
        processor = QueryEvaluator(index)
        return processor.transform(self.parser.parse(q))

    def ranked_query(self, index, q):
        df = dict()
        tf = defaultdict(dict)
        terms = tokenize(q)
        for term in terms:
            results = index.search(term)
            df[term] = len(results.keys())
            for docID,positions in results.items():
                tf[docID][term] = len(positions)
           
        scores = {
            docID: sum([(1 + math.log10(tf[docID][t])) * math.log10(len(index.docIDs)/df[t]) for t in terms if t in tf[docID]])

            for docID in tf.keys()
        }
        return scores

    
    def write_ranked_to_file(self, results, query_id, file="results.ranked.txt"):

        fout = open(file, "a")
        for docID, score in sorted(results.items(), reverse=True, key=lambda item: item[1]):
            fout.write( str(query_id) + ',' + docID + ','+ str(score) + '\n')
        fout.close()


    def write_boolean_to_file(self, results, query_id, file="results.boolean.txt"):

        fout = open(file, "a")
        for docID in sorted(results,  key=lambda item: int(item)):
            fout.write( str(query_id) + ',' + docID + '\n')
        fout.close()


# %%
index = Index(source='trec.5000.xml')
q = QueryEngine()

# %%
index.write_to_file("index.txt")

with open('queries.ranked.txt') as f:
    for i, line in enumerate(f):
        q.write_ranked_to_file(q.ranked_query(index, line.strip().split(" ",1)[1]), i+1, "results.ranked.txt")

with open('queries.boolean.txt') as f:
    for i, line in enumerate(open('queries.boolean.txt')):
            q.write_boolean_to_file(q.boolean_query(index, line.strip().split(" ",1)[1]), i+1, "results.boolean.txt")
