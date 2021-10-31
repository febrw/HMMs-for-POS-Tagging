import inspect, sys, hashlib
import itertools, math

# Hack around a warning message deep inside scikit learn, loaded by nltk
#  Modelled on https://stackoverflow.com/a/25067818
import warnings
with warnings.catch_warnings(record=True) as w:
    save_filters=warnings.filters
    warnings.resetwarnings()
    warnings.simplefilter('ignore')
    import nltk
    warnings.filters=save_filters
try:
    nltk
except NameError:
    # didn't load, produce the warning
    import nltk

from nltk.corpus import brown
from nltk.tag import map_tag, tagset_mapping

if map_tag('brown', 'universal', 'NR-TL') != 'NOUN':
    # Out-of-date tagset, we add a few that we need
    tm=tagset_mapping('en-brown','universal')
    tm['NR-TL']=tm['NR-TL-HL']='NOUN'

class HMM:
    def __init__(self, train_data, test_data):
        """
        Initialise a new instance of the HMM.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :param test_data: the test/evaluation dataset, a list of sentence with tags
        :type test_data: list(list(tuple(str,str)))
        """
        self.train_data = train_data
        self.test_data = test_data

        # Emission and transition probability distributions
        self.emission_PD = None
        self.transition_PD = None
        self.states = []

        self.viterbi = []
        self.backpointer = []

    def lidstone(freqDist):
        return nltk.LidstoneProbDist(freqDist,0.01,freqDist.B() + 1)

    # Compute emission model using ConditionalProbDist with a LidstoneProbDist estimator.
    #   To achieve the latter, pass a function
    #    as the probdist_factory argument to ConditionalProbDist.
    #   This function should take 3 arguments
    #    and return a LidstoneProbDist initialised with +0.01 as gamma and an extra bin.
    #   See the documentation/help for ConditionalProbDist to see what arguments the
    #    probdist_factory function is called with.
    def emission_model(self, train_data):
        """
        Compute an emission model using a ConditionalProbDist.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :return: The emission probability distribution and a list of the states
        :rtype: Tuple[ConditionalProbDist, list(str)]
        """
        #raise NotImplementedError('HMM.emission_model')
        # TODO prepare data

        # Don't forget to lowercase the observation otherwise it mismatches the test data
        # Do NOT add <s> or </s> to the input sentences

        data = [[(p[1], p[0].lower())  for p in s] for s in train_data]

        emission_FD = nltk.ConditionalFreqDist(list(itertools.chain.from_iterable(data)))

        self.emission_PD = nltk.ConditionalProbDist(emission_FD,HMM.lidstone)
        self.states = emission_FD.conditions()
        
        return self.emission_PD, self.states

    # Access function for testing the emission model
    # For example model.elprob('VERB','is') might be -1.4
    def elprob(self,state,word):
        """
        The log of the estimated probability of emitting a word from a state

        :param state: the state name
        :type state: str
        :param word: the word
        :type word: str
        :return: log base 2 of the estimated emission probability
        :rtype: float
        """
        return self.emission_PD[state].logprob(word)


    # Compute transition model using ConditionalProbDist with a LidstonelprobDist estimator.
    # See comments for emission_model above for details on the estimator.
    def transition_model(self, train_data):
        """
        Compute an transition model using a ConditionalProbDist.

        :param train_data: The training dataset, a list of sentences with tags
        :type train_data: list(list(tuple(str,str)))
        :return: The transition probability distribution
        :rtype: ConditionalProbDist
        """

        data = [[(p[1], p[0].lower()) for p in s] for s in train_data]

        # The data object should be an array of tuples of conditions and observations,
        # in our case the tuples will be of the form (tag_(i),tag_(i+1)).
        # DON'T FORGET TO ADD THE START SYMBOL </s> and the END SYMBOL </s>
        data =  [[("<s>","<s>")] + s + [("</s>","</s>")] for s in data]

        tagGenerators=(((s[i][0],s[i+1][0]) for i in range(len(s)-1)) for s in data)
        data = itertools.chain.from_iterable(tagGenerators)

        transition_FD = nltk.ConditionalFreqDist(data)
        self.transition_PD = nltk.ConditionalProbDist(transition_FD,HMM.lidstone)

        return self.transition_PD

    # Access function for testing the transition model
    # For example model.tlprob('VERB','VERB') might be -2.4
    def tlprob(self,state1,state2):
        """
        The log of the estimated probability of a transition from one state to another

        :param state1: the first state name
        :type state1: str
        :param state2: the second state name
        :type state2: str
        :return: log base 2 of the estimated transition probability
        :rtype: float
        """
        return self.transition_PD[state1].logprob(state2)

    # Train the HMM
    def train(self):
        """
        Trains the HMM from the training data
        """
        self.emission_model(self.train_data)
        self.transition_model(self.train_data)

    # Part B: Implementing the Viterbi algorithm.

    # Initialise data structures for tagging a new sentence.
    # Describe the data structures with comments.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD
    # Input: first word in the sentence to tag

    def initialise(self, observation):
        """
        Initialise data structures for tagging a new sentence.

        :param observation: the first word in the sentence to tag
        :type observation: str
        """

        # Initialise step 0 of viterbi, including
        # transition from <s> to observation
        # use costs (-log-base-2 probabilities)

        # Viterbi and Backpointer matrices are [T]x[N]
        # where N is the number of states and T is the number of observations
        # I set the first row corresponding to the first observation to zero for both structures
        # Viterbi[step][state] = cost (Viterbi Function - pi - b)
        # Backpointer[step][state] stores the index of the state with the least cost
        # As we do not know T in advance, we can iteratively add rows for each observation in our input sentence

        N = len(self.states)
        self.viterbi = [N*[0]]
        self.backpointer = [N*[0]]

        for s in self.states:
            pi = self.tlprob("<s>",s)
            b = self.elprob(s,observation)
            self.viterbi[0][self.states.index(s)] = -(pi + b)
        

    # Tag a new sentence using the trained model and already initialised data structures.
    # Use the models stored in the variables: self.emission_PD and self.transition_PD.
    # Update the self.viterbi and self.backpointer data structures.
    # Describe your implementation with comments.
    def tag(self, observations):
        """
        Tag a new sentence using the trained model and already initialised data structures.

        :param observations: List of words (a sentence) to be tagged
        :type observations: list(str)
        :return: List of tags corresponding to each word of the input
        """

        tags = []
        # Number of states
        N = len(self.states)
        # Number of words in our sentence
        T = len(observations)
        
        # iterate from second till last step
        for t in range(1,T):
        # add a row to each structure for each new observation
            self.viterbi.append(N*[0])
            self.backpointer.append(N*[0])
            
            for s in self.states:  
                min_cost = self.viterbi[t-1][0] - self.tlprob(self.states[0],s) - self.elprob(s,observations[t])
                best_idx = 0
                for s_prime in range(1,N):
                    pi = self.tlprob(self.states[s_prime],s)
                    b = self.elprob(s,observations[t])
                    viterbi_cost = self.viterbi[t-1][s_prime] - pi - b
                    # Establish lowest cost from s_prime, store this for backpointers
                    if viterbi_cost < min_cost:
                        min_cost = viterbi_cost
                        best_idx = s_prime
                state_idx = self.states.index(s)
                # Update Viterbi and backpointer structures
                self.viterbi[t][state_idx] = min_cost
                self.backpointer[t][state_idx] = best_idx

        # Add a termination step with cost based solely on cost of transition to </s> , end of sentence.

        final_idx = 0
        for s in self.states:
            min_cost = self.viterbi[T-1][0] - self.tlprob(self.states[0],"</s>")
            best_idx = 0
            # loop over states, finding lowest cost state transitioning to end of sentence tag
            for s_prime in range(1,N):
                viterbi_cost = self.viterbi[T-1][s_prime] - self.tlprob(self.states[s_prime],"</s>")
                if viterbi_cost < min_cost:
                    min_cost = viterbi_cost
                    best_idx = s_prime
            final_idx = best_idx
        # Reconstruct the tag sequence using the backpointer list.
        # Return the tag sequence corresponding to the best path as a list.
        # The order should match that of the words in the sentence.

        tags = [self.states[final_idx]] + tags
        prev_s = ""

        for i in range(T,1,-1):
            # Iteratively find the previous state, and update tags, then backpointers
            prev_s = self.states[self.backpointer[i-1][final_idx]]
            # prepend this new tag to our list, making a new list
            tags = [prev_s] + tags
            final_idx = self.backpointer[i-1][final_idx]

        return list(tags)

    # Access function for testing the viterbi data structure
    # For example model.get_viterbi_value('VERB',2) might be 6.42
    def get_viterbi_value(self, state, step):
        """
        Return the current value from self.viterbi for
        the state (tag) at a given step

        :param state: A tag name
        :type state: str
        :param step: The (0-origin) number of a step
        :type step: int
        :return: The value (a cost) for state as of step
        :rtype: float
        """
        state_idx = self.states.index(state)
        return self.viterbi[step][state_idx]

    # Access function for testing the backpointer data structure
    # For example model.get_backpointer_value('VERB',2) might be 'NOUN'
    def get_backpointer_value(self, state, step):
        """
        Return the current backpointer from self.backpointer for
        the state (tag) at a given step

        :param state: A tag name
        :type state: str
        :param step: The (0-origin) number of a step
        :type step: str
        :return: The state name to go back to at step-1
        :rtype: str
        """
        state_idx = self.states.index(state)
        return self.states[self.backpointer[step][state_idx]]

def answer_question4b():
    """
    Report a hand-chosen tagged sequence that is incorrect, correct it
    and discuss
    :rtype: list(tuple(str,str)), list(tuple(str,str)), str
    :return: your answer [max 280 chars]
    """

    # One sentence, i.e. a list of word/tag pairs, in two versions
    #  1) As tagged by your HMM
    #  2) With wrong tags corrected by hand
    tagged_sequence = [("i'm", 'X'), ('ruddy', 'X'), ('lazy', 'X'), (',', '.'), ('and', 'CONJ'), ("i'm", 'PRT'), ('getting', 'VERB'), ('on', 'ADP'), ('in', 'ADP'), ('years', 'NOUN'), ('.', '.')] 
    correct_sequence = [("I'm", 'PRT'), ('ruddy', 'ADV'), ('lazy', 'ADJ'), (',', '.'), ('and', 'CONJ'), ("I'm", 'PRT'), ('getting', 'VERB'), ('on', 'PRT'), ('in', 'ADP'), ('years', 'NOUN'), ('.', '.')]
    # Why do you think the tagger tagged this example incorrectly?
    answer =  inspect.cleandoc("""\
    The tagger misclassified 4 tags in this sentence. 3 were mistagged as other, which is a popular choice in low probability scenarios. For each case, the emission proability of the word is fairly low, explaining the mistagging.
    
   
    """)[0:280]
    print(len(answer))

    return tagged_sequence, correct_sequence, answer

def answer_question5():
    """
    Suppose you have a hand-crafted grammar that has 100% coverage on
    constructions but less than 100% lexical coverage.
    How could you use a POS tagger to ensure that the grammar
    produces a parse for any well-formed sentence,
    even when it doesn't recognise the words within that sentence?

    :rtype: str
    :return: your answer [max 500 chars]
    """
    return inspect.cleandoc("""\
    We can first use our POS tagger to predict the tags of unseen words. This is accomplished with smoothing methods (in our case, a Lidstone distribution), and the use of our transition and emission matrices, where we may redistribute probability mass to accont for unseen words in our model. We can assign a predicted tag to each word and feed the predicted tag sequence through a constituency tree comprised from the tags alone, resulting in a parse.
    """)[0:500]

def answer_question6():
    """
    Why else, besides the speedup already mentioned above, do you think we
    converted the original Brown Corpus tagset to the Universal tagset?
    What do you predict would happen if we hadn't done that?  Why?

    :rtype: str
    :return: your answer [max 500 chars]
    """
    return inspect.cleandoc("""\
    The Brown corpus contains over 80 tags, whereas the Universal Tagset uses 17. This vastly reduces the sparse data problem as we see many more occurences of each tag. Many of our emission probabilities would be 0 and need smoothing due to sparse data, though using fewer tag categories counteracts this. For transition probabilities, we have fewer examples of each pairwise state transition, also incurring smaller samples and less reliable results. Again fewer categories here improves our results.
    """)[0:500]

# Useful for testing
def isclose(a, b, rel_tol=1e-09, abs_tol=0.0):
    # http://stackoverflow.com/a/33024979
    return abs(a - b) <= max(rel_tol * max(abs(a), abs(b)), abs_tol)

def answers():
    global tagged_sentences_universal, test_data_universal, \
           train_data_universal, model, test_size, train_size, ttags, \
           correct, incorrect, accuracy, \
           good_tags, bad_tags, answer4b, answer5

    # Load the Brown corpus with the Universal tag set.
    tagged_sentences_universal = brown.tagged_sents(categories='news', tagset='universal')

    # Divide corpus into train and test data.
    test_size = 500
    train_size = len(tagged_sentences_universal) - 500 

    test_data_universal = tagged_sentences_universal[-500:]
    train_data_universal = tagged_sentences_universal[:-500]

    if hashlib.md5(''.join(map(lambda x:x[0],train_data_universal[0]+train_data_universal[-1]+test_data_universal[0]+test_data_universal[-1])).encode('utf-8')).hexdigest()!='164179b8e679e96b2d7ff7d360b75735':
        print('!!!test/train split (%s/%s) incorrect, most of your answers will be wrong hereafter!!!'%(len(train_data_universal),len(test_data_universal)),file=sys.stderr)

    # Create instance of HMM class and initialise the training and test sets.
    model = HMM(train_data_universal, test_data_universal)

    # Train the HMM.
    model.train()

    # Some preliminary sanity checks
    # Use these as a model for other checks
    e_sample=model.elprob('VERB','is')
    if not (type(e_sample)==float and e_sample<=0.0):
        print('elprob value (%s) must be a log probability'%e_sample,file=sys.stderr)

    t_sample=model.tlprob('VERB','VERB')
    if not (type(t_sample)==float and t_sample<=0.0):
           print('tlprob value (%s) must be a log probability'%t_sample,file=sys.stderr)

    if not (type(model.states)==list and \
            len(model.states)>0 and \
            type(model.states[0])==str):
        print('model.states value (%s) must be a non-empty list of strings'%model.states,file=sys.stderr)

    print('states: %s\n'%model.states)

    ######
    # Try the model, and test its accuracy [won't do anything useful
    # until you've filled in the tag method]
    ######
    s='the cat in the hat came back'.split()
    model.initialise(s[0])
    ttags = model.tag(s)
    print("Tagged a trial sentence:\n  %s"%list(zip(s,ttags)))

    v_sample=model.get_viterbi_value('VERB',5)
    if not (type(v_sample)==float and 0.0<=v_sample):
           print('viterbi value (%s) must be a cost'%v_sample,file=sys.stderr)

    b_sample=model.get_backpointer_value('VERB',5)
    if not (type(b_sample)==str and b_sample in model.states):
           print('backpointer value (%s) must be a state name'%b_sample,file=sys.stderr)

    # check the model's accuracy (% correct) using the test set
    correct = 0
    incorrect = 0
    counter = 0
    for sentence in test_data_universal:
        
        tokens_correct = 0
        s = [word.lower() for (word, tag) in sentence]
        model.initialise(s[0])
        tags = model.tag(s)

       
        for ((word,gold),tag) in zip(sentence,tags):
            if tag == gold:
                correct +=1
                tokens_correct+=1
            else:
                incorrect +=1
        if (len(sentence)!= tokens_correct and counter < 10):
            counter+=1
            print("Original Sentence, tagged",'\n')
            print(sentence,'\n')
            print("Output from your model",'\n')
            print(list(zip(s,tags)),'\n')


    accuracy = correct/(incorrect+correct)
    print('Tagging accuracy for test set of %s sentences: %.4f'%(test_size,accuracy))

    # Print answers for 4b, 5 and 6
    bad_tags, good_tags, answer4b = answer_question4b()
    print('\nA tagged-by-your-model version of a sentence:')
    print(bad_tags)
    print('The tagged version of this sentence from the corpus:')
    print(good_tags)
    print('\nDiscussion of the difference:')
    print(answer4b[:280])
    answer5=answer_question5()
    print('\nFor Q5:')
    print(answer5[:500])
    answer6=answer_question6()
    print('\nFor Q6:')
    print(answer6[:500])

if __name__ == '__main__':
    if len(sys.argv)>1 and sys.argv[1] == '--answers':
        import adrive2_embed
        from autodrive_embed import run, carefulBind
        with open("userErrs.txt","w") as errlog:
            run(globals(),answers,adrive2_embed.a2answers,errlog)
    else:
        answers()
