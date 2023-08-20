## Large Language Models or LLMs

Take a look at this Twitter thread by productivity Youtuber Ali Abdaal:

![image](https://github.com/roy-sub/Machine-Learning-Bootcamp/blob/main/LLM%20Tweet.png)

At first glance, nothing seems awry and by the looks of it, it seems to be a really popular thread. Here’s the catch — It was almost entirely written by a large language model 
(LLM) pretending to be him. In fact, it did such a good job that it got over 1 million views and nearly 25k engagements shortly after it was published.

If your immediate thought is “Does this mean AI is coming for our jobs?”, don’t worry. It’s not quite there. Yet.

The model that wrote Ali’s thread is called GPT-3 (Generative Pretrained Transformer). With no context of his personality, other than a few words that he entered to get it 
started, it promptly churned out a pro twitter thread most of us might be proud of.

I can sense a lot of questions in your mind — “Why are these called Large language models?”, “How do they work?” I’ll try to break these down in the next few sections.

#### What are LLMs?

Large language models are neural networks based on a specific breakthrough from 2017, the Transformer (no pun intended). The reason they are large is because 
a) They are several gigabytes in size and b) They are trained on enormous (petabyte scale) datasets. These models have billions of parameters. Think of parameters like 
knobs that a model can tweak to learn things. Therefore, the more knobs it has, the more complex representations it can learn. Over time, these models have become larger and 
larger, and have also become significantly more powerful.

There are several different types of large language models, each with its own unique strengths and weaknesses. Some of the most popular models include OpenAI’s GPT-3.5 (Generative Pre-trained Transformer 3.5), 
Google’s BERT (Bidirectional Encoder Representations from Transformers), and T5 (Text-to-Text Transfer Transformer)

### Why are they large?

So we know why they’re called large language models. Are they really such a big deal? Well, kind of. They can do everything from writing essays, creating charts, and writing code 
(yup, scary), with limited or no supervision. Fundamentally though, all they are trained to do is predict the next word in a sentence — Yes, similar to autocomplete on your phone. 
So why do you need a large model and datasets for something this trivial? Let’s look at that next.

LLMs are remarkable because a single model can be used for several tasks. As the size of the model increases (in parameters), it can learn more complex representations, and, 
therefore can do more tasks better than a smaller model. What’s more, they can learn new tasks from just a handful of examples. These tasks can be any of the things I mentioned 
before and more! That’s why researchers are trying to scale these models up.

The natural question that emerges next is how on earth do these models learn : These models learn in two stages — Pretraining and fine-tuning.

### Generative Pre-training of GPT

As mentioned earlier, we need to pre-train the model on very large text corpus. It then understands the underlying relationship between various words in the given context. 
Let’s look at an example.

Suppose we input a sentence to the model, “Life is a work in progress”. What does it learn? And how? Firstly, we should understand that we push any model to learn by 
providing inputs and outputs (targets). It then learns to map them i.e., for a given input, it learns to predict the output as close to the target that is present in 
the training data. For generative purpose, a brilliant way of creating a target is to shift our input sentence one word to the right and make it as a target! This way, 
we are teaching the model to generate “next word”, given the previous sequence of words.

![image](https://github.com/roy-sub/Machine-Learning-Bootcamp/blob/main/Generative%20Pre-Training.png)

This means, the model is learning the relationships such that it learns:

* For input, “Life is a”, the target word to generate is “work”
* For input, “Life is a work in”, the target word to generate is “progress”
* And so on…

**Note :** And lastly comes the token “<eos>”. This is end of sentence token, it now learns to stop the sentence!

There are some interesting aspects of this training process:

**Self-Supervised:** It is interesting to note that the training is unsupervised or self-supervised, meaning that nobody provided the model with any labelled input. Just the 
raw sentences are fed to the model, the target sequence is shifted by one token and voila! The model starts learning relationships between the words which it can reproduce 
later while predicting the output sequence.

**Auto regressive:** Also, the context of the words is carried from not just the last word but all the previous words. After each token or word is generated, that token 
is added to the sequence of inputs. And the whole sequence then becomes an input to the subsequent step!

**Unidirectional:** The context is carried and learnt from left to right and not vice versa. Unlike BERT, which is bidirectional.

### Supervised Fine Tuning of GPT

The generative pre-training helps GPT understand the nuances of the language. After we have pre-trained the GPT model, we can now fine-tune it for any task in NLP. 
We can use domain specific dataset in the same language to take advantage of the learnings and understanding of the model for that language.

The purpose of fine-tuning is to further optimize the GPT model to perform well on a specific task, by adjusting its parameters to better fit the data for that task. 
For example, a GPT model that has been pre-trained on a large corpus of text data can be fine-tuned on a dataset of specific sports commentary to improve its ability to 
answer the questions for a given sports event. Please note that it can get expensive to fine tune such large models depending on the size of data. There are other smaller 
or specific versions of GPT3 (such as Ada and Davinci) that are available for fine tuning as well.

Fine-tuning a GPT model is a powerful tool for a variety of NLP applications, as it enables the model to be tailored to specific tasks and datasets.

A simplified explanation is that the model builds a probability distribution of the possible words that can come in next! Will it be “work” or “song”? If the word “work” has 
a higher probability (i.e., appearing the greatest number of times) the word “work” is predicted. The model then tries to predict the next word after “Life is a work”. 
And this continues till we reach “End of Sentence” prediction!

## LangChain

LLMs have made it possible to use prompting to develop powerful AI applications much faster than ever before but an application say building a question answering system to 
answer questions about the text documents that might require prompting an LLM multiple times with multiple inputs pausing the LLMs output to then feed it to Downstream LLM 
prompts and so on and so there's a lot of glue code needed to build these applications. LangChain created by Harrison is a powerful open source software package that makes 
this process easier.

In short LangChain is a framework for building applications with language models it started after assesing the situations of folks who were building Cutting Edge and advanced 
applications and saw some common abstractions that can be factored out into a library to make it easier for others to do so as well.

The core abstractions that you can use to build an LLM powered application are as follows : models prompts and parchers these are basic building blocks for interacting with 
inputs and outputs of language models memory which is how you turn the stateless language model into something that you can hold a conversation with so it remembers previous 
interactions chains or how you can string together sequencies of these operations when calling a language model question answering over documents how you can combine your own 
data with a language model so that it knows more than just the data that it was trained on. Evaluation how do you evaluate these applications that are now powered by this 
non-deterministic model and agents which involve using the language model as a reasoning engine when interacting with other sources of data and computation with these building 
blocks there is an incredible amount of applications that you can build often with a relatively few lines of code and that's super exciting. And so an LLM is a very powerful 
platform to develop on but instead of calling LLMs yourself and doing all the work to put the application together kind of from scratch on top of the LLM LangChain provides 
high level abstractions that make building these applications easier.

In this blog I will talk about prompts, which are how you get models to do useful and interesting things and indexes, which are ways of ingesting data so that you can combine 
it with models. And then I'll talk about chains, which are more end-to-end use cases along with agents, which are a very exciting type of end-to-end use case which uses the 
model as a reasoning engine.

### Models, Prompts and Output Parsers

### Memory

![image](https://github.com/roy-sub/Machine-Learning-Bootcamp/blob/main/LLM%20Memory.png)

When you interact with these models, naturally they don't remember what you say before or any of the previous conversation, which is an issue when you're building some 
applications like Chatbot and you want to have a conversation with them. And so, in this section we'll cover memory, which is basically how do you remember previous parts of 
the conversation and feed that into the language model so that they can have this conversational flow as you're interacting with them. So, LangChain offers multiple 
sophisticated options of managing these memories.

When you use a large language model for a chat conversation, the large language model itself is actually stateless. The language model itself does not remember the conversation 
you've had so far. And each transaction, each call to the API endpoint is independent. And chatbots appear to have memory only because there's usually rapid code that provides 
the full conversation that's been had so far as context to the LLM. And this memory storage is used as input or additional context to the LLM so that they can generate an output 
as if it's just having the next conversational turn, knowing what's been said before. And as the conversation becomes long, the amounts of memory needed becomes really, really 
long and does the cost of sending a lot of tokens to the LLM, which usually charges based on the number of tokens it needs to process, will also become more expensive. 
So LangChain provides several convenient kinds of memory to store and accumulate the conversation.

Let's look at a different type of memory called conversation buffer window memory that only keeps a window of memory. If I set memory to conversational buffer window memory with 
k equals one, the variable k equals one specifies that I wanted to remember just one conversational exchange. That is one utterance from me and one utterance from chatbot. So this 
is a nice feature because it lets you keep track of just the most recent few conversational terms. In practice, you probably won't use this with k equals one. You use this with 
k set to a larger number. But still, this prevents the memory from growing without limit as the conversation goes longer.

Few types of memory that LangChain uses includes buffer memories that limits based on number of conversation exchanges or tokens or a memory that can summarize tokens above 
a certain limit. LangChain actually supports additional memory types as well. One of the most powerful is vector data memory, if you're familiar with word embeddings and 
text embeddings, the vector database actually stores such embeddings. If you don't know what that means, don't worry about it I will explain it later. And LangChain also supports 
entity memories, which is applicable when you wanted to remember details about specific people, specific other entities, such as if you talk about a specific friend, you can have 
LangChain remember facts about that friend, which would be an entity in an explicit way.

When you're implementing applications using LangChain, you can also use multiple types of memories such as using one of the types of conversation memory plus additionally entity 
memory to recall individuals. So this way you can remember maybe a summary of the conversation plus an explicit way of storing important facts about important people in the 
conversation. And of course, in addition to using these memory types, it's also not uncommon for developers to store the entire conversation in the conventional database. Some 
sort of key value store or SQL database. So you could refer back to the whole conversation for auditing or for improving the system further. 

### Chains

In this section, we will talk about the most important key building block of LangChain, namely, the chain. The chain usually combines an LLM, large language model, together with 
a prompt, and with this building block you can also put a bunch of other building blocks together to carry out a sequence of operations on your text or on your other data.

```
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains import LLMChain

llm = ChatOpenAI(temperature=0.9)

prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}?"
)

chain = LLMChain(llm=llm, prompt=prompt)

product = "Queen Size Sheet Set"
chain.run(product)

--> 'Royal Bedding'
```
So the first chain we're going to cover is the LLM chain. And this is a simple but really powerful chain that underpins a lot of the chains that we'll go over in the future. 
And so, we're going to import three different things. We're going to import the OpenAI model, so the LLM. We're going to import the chat prompt template. And so this is the 
prompt. And then we're going to import the LLM chain. And so first, what we're going to do is we're going to initialize the language model that we want to use. So we're going 
to initialize the chat OpenAI with a high temperature so that we can get some fun descriptions. Now we're going to initialize a prompt. And this prompt is going to take in a 
variable called product. It's going to ask the LLM to generate what the best name is to describe a company that makes that product. And then finally, we're going to combine 
these two things into a chain. And so, this is what we call an LLM chain. And it's quite simple. It's just the combination of the LLM and the prompt

```
from langchain.chains import SimpleSequentialChain

llm = ChatOpenAI(temperature=0.9)

# prompt template 1
first_prompt = ChatPromptTemplate.from_template(
    "What is the best name to describe \
    a company that makes {product}?"
)

# Chain 1
chain_one = LLMChain(llm=llm, prompt=first_prompt)


# prompt template 2
second_prompt = ChatPromptTemplate.from_template(
    "Write a 20 words description for the following \
    company:{company_name}"
)
# chain 2
chain_two = LLMChain(llm=llm, prompt=second_prompt)

overall_simple_chain = SimpleSequentialChain(chains=[chain_one, chain_two],
                                             verbose=True
                                            )

overall_simple_chain.run(product)
```

The next type of chain, which will be simple sequential chains. And so sequential chains run a sequence of chains one after another. So to start, you're going to import the simple 
sequential chain. And this works well when we have subchains that expect only one input and return only one output. And so here we're going to first create one chain, 
which uses an LLM and a prompt. And this prompt is going to take in the product and will return the best name to describe that company. So that will be the first chain. Then 
we're going to create a second chain. In this second chain, we'll take in the company name and then output a 20-word description of that company. And so you can imagine how 
these chains might want to be run one after another, where the output of the first chain, the company name, is then passed into the second chain. We can easily do this by 
creating a simple sequential chain where we have the two chains described there. And we'll call this overall simple chain. Now, what you can do is run this chain over any 
product description. And so if we use it with the product above, the queen size sheet set, we can run it over and we can see that it first outputs royal betting. And then it 
passes it into the second chain and it comes up with this description of what that company could be about. The simple sequential chain works well when there's only a single 
input and a single output. But what about when there are multiple inputs or multiple outputs? And so we can do this by using just the regular sequential chain. 

```
from langchain.chains import SequentialChain

llm = ChatOpenAI(temperature=0.9)

# prompt template 1: translate to english
first_prompt = ChatPromptTemplate.from_template(
    "Translate the following review to english:"
    "\n\n{Review}"
)
# chain 1: input= Review and output= English_Review
chain_one = LLMChain(llm=llm, prompt=first_prompt, 
                     output_key="English_Review"
                    )

second_prompt = ChatPromptTemplate.from_template(
    "Can you summarize the following review in 1 sentence:"
    "\n\n{English_Review}"
)
# chain 2: input= English_Review and output= summary
chain_two = LLMChain(llm=llm, prompt=second_prompt, 
                     output_key="summary"
                    )

# prompt template 3: translate to english
third_prompt = ChatPromptTemplate.from_template(
    "What language is the following review:\n\n{Review}"
)
# chain 3: input= Review and output= language
chain_three = LLMChain(llm=llm, prompt=third_prompt,
                       output_key="language"
                      )


# prompt template 4: follow up message
fourth_prompt = ChatPromptTemplate.from_template(
    "Write a follow up response to the following "
    "summary in the specified language:"
    "\n\nSummary: {summary}\n\nLanguage: {language}"
)
# chain 4: input= summary, language and output= followup_message
chain_four = LLMChain(llm=llm, prompt=fourth_prompt,
                      output_key="followup_message"
                     )

# overall_chain: input= Review 
# and output= English_Review,summary, followup_message
overall_chain = SequentialChain(
    chains=[chain_one, chain_two, chain_three, chain_four],
    input_variables=["Review"],
    output_variables=["English_Review", "summary","followup_message"],
    verbose=True
)

```

Now let's talk about sequential chain The main difference between a simple sequential chain and a sequential chain is that a simple sequential chain can only take a single input 
and produce a single output, while a sequential chain can take multiple inputs and produce multiple outputs. In a simple sequential chain, the output of each step is the input 
to the next step. This means that the output of the chain is determined by the input to the first step and the outputs of all the intermediate steps. Whereas in a sequential 
chain, the output of each step can be used as the input to multiple other steps. This means that the output of the chain can be determined by the inputs to any of the steps and 
the outputs of all the other steps.

![image](https://github.com/roy-sub/Machine-Learning-Bootcamp/blob/main/LLM%20Route%20Chain.png)

So far we've covered the LLM chain, simple sequential chain and sequential cahin. But what if you want to do something more complicated? A pretty common but basic operation is 
to route an input to a chain depending on what exactly that input is. A good way to imagine this is if you have multiple sub chains, each of which specialized for a particular 
type of input, you could have a router chain which first decides which subchain to pass it to and then passes it to that chain. 

### Evaluating LLM Applications :

When building a complex application using an LLM, one of the important but sometimes tricky steps is how do you evaluate how well your application is doing? Is it meeting some 
accuracy criteria? And also, if you decide to change your implementation, maybe swap in a different LLM, or change the strategy of how you use a vector database or something 
else to retrieve chunks, or change some other parameters of your system, how do you know if you're making it better or worse? In this section, we will dive into some frameworks 
on how to think about evaluating a LLM-based application, as well as some tools to help you do that. These applications are really chains and sequences of a lot of different steps. 
And so honestly, part of the first thing that you should do is just understand what exactly is going in and coming out of each step. And so some of the tools can really just be 
thought of as visualizers or debuggers in that vein. But it's often really useful to get a more holistic picture on a lot of different data points of how the model is doing. 
And one way to do that is by looking at things by eye. But there's also this really cool idea of using language models themselves and chains themselves to evaluate other 
language models, and other chains, and other applications. And we'll dive a bunch into that as well. So, lots of cool topics, and I find that with a lot of development shifting 
towards prompting-based development, developing applications using LLMs, this whole workflow evaluation process is being rethought. So, lots of exciting concepts in this section.

**Automated Evaluation with Language Models:** Language models can automate example creation, streamlining evaluation. Utilizing a question answering (QA) generation chain, you 
can generate query-answer pairs from documents. This approach saves time and provides diverse examples for evaluation. The QA evaluation chain plays a pivotal role in automated 
evaluation. By providing examples and predictions, you can gauge performance effectively. Language models help in grading responses by comparing predicted and real answers.

**Debugging Using langchain.debug:** Debugging complex applications requires insights into each step. The langchain.debug tool offers detailed information about the process. 
By turning on debug mode, you can trace the chain's execution, understand inputs/outputs, and identify potential issues. This debugging approach is especially valuable for 
complex chains with multiple steps.

**LangChain Evaluation Platform:** The LangChain Evaluation Platform enhances evaluation by providing a UI for tracking and visualizing inputs and outputs. This platform allows 
you to view the execution steps, delve into chain details, and add examples to datasets effortlessly. Creating datasets for evaluation is fundamental. The platform's capability 
to add examples to datasets simplifies the process, enabling you to accumulate examples gradually for a robust evaluation cycle.

Evaluating LLM-based applications requires a comprehensive approach that encompasses manual and automated techniques. With tools like langchain.debug and the LangChain Evaluation Platform, developers can gain insights into chain execution and efficiently create datasets for evaluation. As the landscape of language model application development evolves, embracing these evaluation practices becomes essential for building reliable and accurate applications.

### Agents

Sometimes people think of a large language model as a knowledge store, as if it's learned to memorize a lot of information, maybe off the internet, so when you ask it a 
question, it can answer the question. But I think a even more useful way to think of a large language model is sometimes as a reasoning engine, in which you can give it 
chunks of text or other sources of information. And then the large language model, LLM, will maybe use this background knowledge that's learned off the internet, but to 
use the new information you give it to help you answer questions or reason through content or decide even what to do next. And that's what LangChain's Agents framework helps 
you to do. 
 
They're also one of the most powerful parts, but they're also one of the newer parts. So we're seeing a lot of stuff emerge here that's really new to everyone in the field. 
And so this should be a very exciting lesson as we dive into what agents are, how to create and how to use agents, how to equip them with different types of tools, like search 
engines that come built into LangChain, and then also how to create your own tools, so that you can let agents interact with any data stores, any APIs, any functions that you 
might want them to.

### Conclusion

In the world of AI, Large Language Models (LLMs) like GPT-3 are transformative tools, capable of understanding and generating human-like text. This blog introduced the core 
workings of LLMs, explained LangChain's role in simplifying their application, and highlighted the importance of evaluating and utilizing these models effectively. With 
LangChain's tools and the concept of Agents, developers can create powerful applications that reason, respond, and interact with data, ushering in a new era of AI-driven 
solutions. As AI continues to evolve, understanding LLMs and their capabilities empowers us to build innovative applications that can shape the future.











