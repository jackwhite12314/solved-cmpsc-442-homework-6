Download Link: https://assignmentchef.com/product/solved-cmpsc-442-homework-6
<br>
<h1>Instructions</h1>

In this assignment, you will gain experience working with hidden Markov models for part-of-speech tagging.

A skeleton file homework6-cmpsc442.py containing empty definitions for each question has been provided. Since portions of this assignment will be graded automatically, none of the names or function signatures in this file should be modified. However, you are free to introduce additional variables or functions if needed.

You may import definitions from any standard Python library, and are encouraged to do so in case you find yourself reinventing the wheel.

You will find that in addition to a problem specification, most programming questions also include a pair of examples from the Python interpreter. These are meant to illustrate typical use cases to clarify the assignment, and are not comprehensive test suites. In addition to performing your own testing, you are strongly encouraged to verify that your code gives the expected output for these examples before submitting.

You are strongly encouraged to follow the Python style guidelines set forth in <a href="https://legacy.python.org/dev/peps/pep-0008/">PEP 8</a>, which was written in part by the creator of Python. However, your code will not be graded for style.

<h1>1.   Hidden Markov Models [95 points]</h1>

In this section, you will develop a hidden Markov model for part-of-speech (POS) tagging, using the Brown corpus as training data. The tag set used in this assignment will be the <a href="http://www.petrovi.de/data/universal.pdf">universal POS ta</a><a href="http://www.petrovi.de/data/universal.pdf">g</a><a href="http://www.petrovi.de/data/universal.pdf"> set</a>, which is composed of the twelve POS tags NOUN (noun), VERB (verb), ADJ (adjective), ADV (adverb), PRON (pronoun), DET (determiner or article), ADP (preposition or postposition), NUM (numeral), CONJ (conjunction), PRT (particle), ‘.’ (punctuation mark), and X (other).

As in previous assignments, your use of external code should be limited to built-in Python modules, which excludes packages such as NumPy and NLTK.

<ol>

 <li><strong>[10 points] </strong>Write a function load_corpus(path) that loads the corpus at the given path and returns it as a list of POS-tagged sentences. Each line in the file should be treated as a separate sentence, where sentences consist of sequences of whitespace-separated strings of the form “token=POS”. Your function should return a list of lists, with individual entries being 2-tuples of the form (token, POS).</li>

</ol>

&gt;&gt;&gt; c = load_corpus(“brown_corpus.txt”)                                          &gt;&gt;&gt; c = load_corpus(“brown_corpus.txt”)

&gt;&gt;&gt; c[1402]                                                                                                  &gt;&gt;&gt; c[1799]

[(‘It’, ‘PRON’), (‘made’, ‘VERB’),                                                                [(‘The’, ‘DET’), (‘prospects’, ‘NOUN’),

(‘him’, ‘PRON’), (‘human’, ‘NOUN’),                                                          (‘look’, ‘VERB’), (‘great’, ‘ADJ’),

(‘.’, ‘.’)]                                                                                                          (‘.’, ‘.’)]

<ol start="2">

 <li><strong>[20 points] </strong>In the Tagger class, write an initialization method</li>

</ol>

__init__(self, sentences) which takes a list of sentences in the form produced by load_corpus(path) as input and initializes the internal variables needed for the POS tagger. In particular, if <em>{ t</em><em><sub>1</sub>, t</em><em><sub>2</sub>, . . . , t</em><em><sub>n</sub> }</em> denotes the set of tags and <em>{ w</em><em><sub>1</sub>, w</em><em><sub>2</sub>, . . . , w</em><em><sub>m</sub> }</em> denotes the set of tokens found in the input sentences, you should at minimum compute:

The initial tag probabilities <em>π(t</em><em><sub>i</sub>)</em> for <em>1 ≤ i ≤ n</em>, where <em>π(t</em><em><sub>i</sub>)</em> is the probability that a sentence begins with tag <em>t</em><em><sub>i</sub></em>.

The transition probabilities <em>a(t</em><em><sub>i</sub> </em>→<em> t</em><em><sub>j</sub>)</em> for <em>1 ≤ i, j ≤ n</em>, where <em>a(t</em><em><sub>i</sub> </em>→<em> t</em><em><sub>j</sub>)</em> is the probability that tag <em>t</em><em><sub>j</sub></em> occurs after tag <em>t</em><em><sub>i</sub></em>.

The emission probabilities <em>b(t</em><em><sub>i</sub> </em>→<em> w</em><em><sub>j</sub>)</em> for <em>1 ≤ i ≤ n</em> and <em>1 ≤ j ≤ m</em>, where <em>b(t</em><em><sub>i</sub> </em>→<em> w</em><em><sub>j</sub>)</em> is the probability that token <em>w</em><em><sub>j</sub></em> is generated given tag <em>t</em><em><sub>i</sub></em>.

It is imperative that you use Laplace smoothing where appropriate to ensure that your system can handle novel inputs, but the exact manner in which this is done is left up to you as a design decision. Your initialization method should take no more than a few seconds to complete when given the full Brown corpus as input.

<ol start="3">

 <li><strong>[25 points] </strong>In the Tagger class, write a method most_probable_tags(self, tokens) which returns the list of the most probable tags corresponding to each input token. In particular, the most probable tag for a token <em>w</em><em><sub>j</sub></em> is defined to be the tag with index <em>i</em><em><sup>*</sup> = argmax</em><em><sub>i</sub> b(t</em><em><sub>i</sub> </em>→<em> w</em><em><sub>j</sub>)</em>.</li>

</ol>

&gt;&gt;&gt; c = load_corpus(“brown_corpus.txt”)                                                &gt;&gt;&gt; c = load_corpus(“brown_corpus.txt”)

&gt;&gt;&gt; t = Tagger(c)                                                                                        &gt;&gt;&gt; t = Tagger(c)

&gt;&gt;&gt; t.most_probable_tags(                                                                     &gt;&gt;&gt; t.most_probable_tags(

…   [“The”, “man”, “walks”, “.”])                                                                           …   [“The”, “blue”, “bird”, “sings”])

[‘DET’, ‘NOUN’, ‘VERB’, ‘.’]                                                                       [‘DET’, ‘ADJ’, ‘NOUN’, ‘VERB’]

<ol start="4">

 <li><strong>[40 points] </strong>In the Tagger class, write a method viterbi_tags(self, tokens) which returns the most probable tag sequence as found by Viterbi decoding. Recall from lecture that Viterbi decoding is a modification of the Forward algorithm, adapted to find the path of highest probability through the trellis graph containing all possible tag sequences.</li>

</ol>

Computation will likely proceed in two stages: you will first compute the probability of the most likely tag sequence, and will then reconstruct the sequence which achieves that probability from end to beginning by tracing backpointers.

&gt;&gt;&gt; c = load_corpus(“brown_corpus.txt”)                                                &gt;&gt;&gt; c = load_corpus(“brown_corpus.txt”)

&gt;&gt;&gt; t = Tagger(c)                                                                                        &gt;&gt;&gt; t = Tagger(c)

&gt;&gt;&gt; s = “I am waiting to reply”.split()                                               &gt;&gt;&gt; s = “I saw the play”.split()

&gt;&gt;&gt; t.most_probable_tags(s)                                                                 &gt;&gt;&gt; t.most_probable_tags(s)

[‘PRON’, ‘VERB’, ‘VERB’, ‘PRT’, ‘NOUN’]                                            [‘PRON’, ‘VERB’, ‘DET’, ‘VERB’]

&gt;&gt;&gt; t.viterbi_tags(s)                                                                                  &gt;&gt;&gt; t.viterbi_tags(s)

[‘PRON’, ‘VERB’, ‘VERB’, ‘PRT’, ‘VERB’]                                            [‘PRON’, ‘VERB’, ‘DET’, ‘NOUN’]

<h1>2.   Feedback [5 points]</h1>

<ol>

 <li><strong>[1 point] </strong>Approximately how long did you spend on this assignment?</li>

 <li><strong>[2 points] </strong>Which aspects of this assignment did you find most challenging? Were there any significant stumbling blocks?</li>

 <li><strong>[2 points] </strong>Which aspects of this assignment did you like? Is there anything you would have changed?</li>

</ol>