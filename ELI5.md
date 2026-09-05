# The short version, in plain words

## What a language model is doing when it writes

Every time the model writes one word, it does the same thing. It looks at everything
written so far and gives a score to every single word it knows. Qwen, the model I tested,
knows 151,936 word pieces, so it produces 151,936 scores. The highest score is the word it
thinks fits best. Then it turns those scores into percentages and picks one.

Those raw scores are called logits. Everything in this project is about the shape of that
list of 151,936 numbers.

## What temperature actually is

Temperature is the knob that decides how much the model respects its own scores.

At a very low temperature, the model always picks the single highest-scoring word. It is
predictable and repetitive. At a high temperature, the differences between scores stop
mattering, so a word scored 3rd and a word scored 3000th become almost equally likely, and
the text turns to nonsense. The usual setting is 1, which is somewhere in between.

Here is the thing that got me started. The equation that converts scores into
probabilities at temperature T is *character for character* the same equation physicists
use for how a real physical system behaves at a real temperature. It was borrowed from
physics in the first place. Not similar to it, the same.

So if it is the same equation, all the other equations of physics should apply too. That
is what I tested.

## The translation

To use the physics, you need a dictionary between the two worlds. It is short:

- A **word** the model could pick is like a **state** a physical system could be in.
- A word's **score** is its **energy**, flipped in sign. High score means low energy, and
  low energy means the system likes to sit there. The model's favourite word is the
  bottom of the valley.
- **Temperature** is temperature. It decides how much energy is available to climb out of
  that valley.

Now two quantities from physics become measurable for a language model.

**Entropy** measures how spread out the possibilities are. If the model is sure the next
word is "Paris", entropy is near zero. If a thousand words look equally good, entropy is
high. People already use this to guess whether a model is making things up, and it works
reasonably well.

**Heat capacity** measures how much the situation *changes* when you nudge the
temperature. This one is less obvious and it turns out to matter more.

## Why heat capacity is the interesting one

Think about actual ice. If you warm ice from minus 20 degrees to minus 19, almost nothing
happens. Warm it from minus 1 to plus 1 and it becomes water. All the interesting change
is concentrated at one temperature, the melting point. Heat capacity is the instrument
that finds that point, because it spikes exactly where the big change happens.

Language models have a melting point too, and nobody had measured it.

Below it, the model stays committed to its top few words. Above it, the mass of 151,936
irrelevant words wins by sheer weight of numbers and the output collapses into noise. I
worked out the formula for where that happens, and then measured it. For the model I
tested it sits between about 1.5 and 2.6.

That is a satisfying number, because people who use these models found by trial and error
that turning the temperature past roughly 2 produces garbage. They just never knew there
was an equation for it.

## The four things I found

**1. The tools people already use are thermometer readings at one fixed temperature.**
The two standard ways of measuring a model's uncertainty turn out to be exactly the
entropy and exactly the heat capacity, both measured at temperature 1. Nobody had noticed.
It is like discovering that two instruments everyone uses are the same instrument pointed
at the same spot. Which raises the obvious question of what is at the other spots.

**2. There is a formula for the melting point, and it works.** It says the melting point
depends on the gap between the model's favourite word and the general mush of everything
else, divided by how many words the model knows. I tested it by artificially shrinking the
model's vocabulary across a range of 19,000 to 1, and the formula held.

**3. Every question gets read at the wrong temperature.** This is the useful one.

Each question has its own melting point, and they differ by about 13%. But everyone
measures uncertainty at temperature 1 for every question. That is like checking whether
different metals have melted by putting them all in the same oven, when each one melts at
its own temperature.

So instead I measured each question at *its own* melting point. That one change works
better than the entire standard toolkit combined, and it costs nothing extra to compute.

**4. One number beats everything at telling whether the model actually knows the thing.**
The peak height of the heat capacity, which is a measure of how sharply the model commits
to its answer, turns out to be better than any standard measure at spotting questions
where the model is bluffing. On its own it beat all seven standard measures put together.

## The six things I got wrong

I want to be straight about this, because most of my ideas failed.

I expected the model's competing answers to sit in tidy separated groups, like the energy
levels of an atom. They do not. The picture is one smooth smear.

I predicted that a common technique called top-k sampling works by raising the melting
point. It is the opposite, and the data showed me why within minutes.

My main idea was that some of a model's uncertainty is trivial (choosing between "Paris"
and "It is Paris") and some is serious (choosing between "Paris" and "Berlin"), and that
temperature could separate the two. It works perfectly in a simulation I built. It does
nothing at all on a real model.

## The measurement that explained why I was wrong

I could not work out why, so I measured it directly.

I took the model's top 12 candidate next words for each question, forced it to start with
each one, and let it finish the sentence. Then I checked how many of those 12 led to the
same final answer.

The answer was almost none. **96.7% of them led to a completely different answer.**

So the trivial uncertainty I was trying to remove is barely there. The model's choice of
next word essentially *is* its choice of answer. There was nothing to separate.

This also explains something surprising. A well-known technique called semantic entropy
works by generating ten answers and grouping the ones that mean the same thing. On my
test it scored 0.8568, and simply reading the raw uncertainty once scored 0.8607. It did
slightly worse while costing ten times as much, for the same reason: there was nothing for
it to fix. That is not a criticism of the method, which was designed for long free-form
answers where phrasing really does vary. It is a statement about where it earns its cost.

## The honest bottom line

None of this fixes hallucination. A model that never learned a fact cannot be made to know
it by measuring it more cleverly.

What this does is give a better instrument. It makes the measurement independent of
things that should not affect it, it explains where a couple of existing techniques do
and do not earn their keep, and it gives a formula for a limit people had only found by
trial and error. Improving the actual decision of when to shut up rather than guess, which
is what you would want in practice, it barely helps at all.

---

# Part two: what happened after I kept going

## I tested my own prediction and it was right

At the end of part one I said something specific. I claimed the "96.7% of choices lead to
different answers" result was special to short factual questions, and that for longer
open-ended answers it should flip, because there really are many ways to phrase a sentence.

So I ran the same experiment on open-ended questions like "Why is the sky blue?" instead of
"What is the capital of Peru?"

It flipped completely.

For factual questions, 67.8% of the model's candidate next words led somewhere genuinely
different. For open-ended questions, only 9.2% did. Put the other way round: when the model
starts an open-ended answer, **94.7% of the uncertainty among its top 12 candidate tokens is
about wording and only 5.3% is about content**. For a factual question it is the reverse,
13.2% wording. This experiment renormalises those 12 candidates and does not measure the
rest of the vocabulary.

This matters practically. There is a well-known technique that generates ten answers and
groups the ones that mean the same thing, specifically to strip out the wording noise. Now
I can say exactly when it is worth the cost. On open-ended answers there is 94.7% noise to
strip, so it should help a lot. On short factual answers there is 13.2%, which is why in my
test it did slightly worse than just reading the model once.

There is also an uncomfortable consequence for my own earlier results. Everything in part
one was measured at the first word of a short factual answer, which is a case where that
word carries the meaning. At the first word of an open-ended answer the model has already
decided what it will say and is only picking how to start the sentence. So any method that
looks at the first word alone is reading grammar, not facts, and none of my earlier numbers
should be assumed to carry over to long-form writing.

## The melting point formula, properly tested

In part one I derived a formula for the temperature at which a model's output turns to
noise. It says the melting point depends on the gap between the model's favourite word and
the general mush of everything else, divided by a term that grows with how many words the
model knows.

Testing the second half of that needed models with genuinely different vocabulary sizes.
That turned out to be awkward, because almost every real model knows between 49,000 and
250,000 word pieces, which is not enough variation to see the effect.

So I trained five of my own. Same architecture, same books, same amount of reading, and the
only thing changed between them was how many word pieces they were allowed to know: 257,
512, 2048, 8192 and 32768. Then I combined those with seven real models including GPT-2 and
BLOOM.

Twelve models, vocabularies from 257 to 250,880, a range of about a thousand to one. The
formula holds. Including the vocabulary term cuts the disagreement between models by 58%,
and the improvement is statistically solid rather than luck.

## The mistake that looked like a disproof

This one is worth telling because it nearly fooled me.

My first version of the test said the formula was wrong. Badly wrong, with the models
disagreeing by a factor of ten and even trending the wrong way. I almost wrote it up as a
failure.

The problem was one line. To measure "the gap between the favourite word and everything
else", I had used the *average* score of all the words. But some models have a handful of
words with wildly extreme scores, and an average gets dragged around by those. BLOOM was the
worst: using the average, its gap looked like 148 when the honest answer was about 18.

The fix was to use the most common score instead of the average, which is what the physics
actually asks for. With that one change, all twelve models line up.

There is a general lesson in that, and it is not really about my project. Several published
methods for measuring how confident a model is do their sums using averages over these
scores. If a model has a few extreme outliers, that average is not measuring what people
think it is measuring, and it is not safe to compare between models.

## The last idea: the model may be picking the wrong answer on purpose

Here is a consequence I did not expect and am still testing.

When a model writes, it picks the single most likely sequence of words. But that is not the
same as picking the most likely *meaning*. If one meaning can be said in a hundred ways and
another can only be said one way, the second can win on any individual sentence while
losing badly once you add up all hundred versions of the first.

In physics this is exactly the difference between energy and free energy. A state that can
be arranged many ways gets a bonus, and the system settles into the arrangement with the
lowest free energy, not the lowest energy. Since I have now measured that open-ended answers
have enormous wording freedom, that bonus should be large, and standard decoding ignores it
entirely.

## Where all this leaves things

Still true: none of this fixes hallucination. It gives a better instrument and it explains
where two existing techniques do and do not earn their cost.

What is new since part one is that the melting formula now stands up across twelve models,
my prediction about open-ended answers was right and the effect is large, and I found a
measurement error in my own work that has a wider warning attached to it.

---

# Part three: I tested my own idea on a second model and it failed

At the end of part two I had a method I was pleased with. The reasoning went like this. The
model decides what it is going to say in the first two or three words, so instead of
generating sixteen full answers and comparing them, just try the handful of alternative
opening words and see whether they lead somewhere different. Same information, less work.

On the first model it worked well. So I ran it on a second, smaller model.

It fell apart. The score dropped from 0.97 to 0.67, where 0.5 is a coin flip. On that model
the plain old method I had been criticising actually did better than mine. So the method is
withdrawn. It worked on one model and not on the next, which means it did not work.

The measurement it was built on did survive, and cleanly. On the smaller model, 90.6% of
the uncertainty at the start of an open-ended answer is about wording rather than content,
against 94.7% on the bigger one. And for short factual questions it is 4.8% against 13.2%.
The two situations are just as far apart on both models.

## The thing that got stronger

While checking the failure I found something better than what I had lost.

I had been saying that reading a model's uncertainty is a weak way to catch it making things
up in long-form writing. Looking at the two models side by side, it is worse than weak.

On the bigger model, higher uncertainty at the start of an answer meant the model was
somewhat more likely to be fabricating. On the smaller model, it meant the exact opposite,
and strongly so. Higher uncertainty there meant the model actually *knew* the answer.

So the signal does not just have a weak effect. It points in a different direction depending
on which model you use. And on the bigger model it even pointed in different directions for
different comparisons.

That matters practically. A weak signal is still usable if you know which way it points, you
just need more data. A signal whose direction you cannot predict is not usable at all,
because to know which way to read it you would need labelled examples from that exact model
and that exact kind of question. Which is the work you were trying to avoid by using the
signal in the first place.

The expensive method, generating many answers and grouping them by meaning, pointed the right
way on both models every time. It costs about ten times more and it earns it.

## Why I am telling you about the failure

This is the honest shape of the work. I built a measuring instrument out of physics, most of
what I hoped to find with it was not there, and the useful thing turned out to be a plain
measurement the physics only pointed me toward. Then the one method I proposed died on
contact with a second model.

The result I trust most is the one that survived the test that killed my own idea.
