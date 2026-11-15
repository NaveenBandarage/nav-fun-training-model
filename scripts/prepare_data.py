#!/usr/bin/env python3
"""Write small fictional training and validation sets for a three-label classifier."""
import json
from pathlib import Path
ROOT = Path(__file__).resolve().parent.parent
SYSTEM = 'Classify the message. Reply with exactly one label: greeting, question, or request. Greetings include thanks. Use question for general questions. Use request when the person asks you to do or suggest something.'
TRAIN = {
 'greeting': ['hello there', 'hey friend', 'good morning', 'thanks so much', 'nice to meet you', 'good evening', 'thank you for your help', 'hi there', 'good afternoon', 'cheers', 'hello again', 'many thanks'],
 'question': ['why is the sky blue?', 'what is a comet?', 'how do rainbows form?', 'where do penguins live?', 'what is a haiku?', 'how many days are in a week?', 'why do leaves change color?', 'what does melody mean?', 'where is the moon?', 'how do plants grow?', 'what is a constellation?', 'why is the ocean salty?'],
 'request': ['suggest a fun board game', 'write a short poem about cats', 'give me an idea for lunch', 'please recommend a book', 'make up a silly name', 'tell me a story about a frog', 'create a list of picnic snacks', 'suggest a weekend activity', 'write a friendly birthday message', 'give me a drawing idea', 'make a riddle about the moon', 'recommend a rainy day activity'],
}
VALID = {
 'greeting': ['hello my friend', 'much appreciated', 'good night'],
 'question': ['what makes thunder?', 'where do otters live?', 'why do stars shine?'],
 'request': ['suggest a simple craft', 'write me a short story', 'give me a recipe idea'],
}
def main():
 out=ROOT/'data'; out.mkdir(exist_ok=True)
 for split, groups in [('train', TRAIN), ('valid', VALID)]:
  rows=[{'messages':[{'role':'system','content':SYSTEM},{'role':'user','content':message},{'role':'assistant','content':label}]} for label, messages in groups.items() for message in messages]
  (out/f'{split}.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rows))
  print(split, len(rows))
if __name__ == '__main__': main()
