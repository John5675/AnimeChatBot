import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

import requests

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

with open("intents.json", "r") as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()


def top_manga():
    request = requests.get(f"https://api.jikan.moe/v4/top/manga")
    request = request.json()
    request = request["data"]
    list = ""
    for manga in request:
        list = list + manga["title"] + "\n"
    print("\nHere are the top manga of all time: \n" + list)


def recent_manga():
    request = requests.get(f"https://api.jikan.moe/v4/recommendations/manga?page=1")
    request = request.json()
    request = request["data"]
    list = ""
    for manga in request:
        list = list + manga["entry"][0]["title"] + "\n"
    print("\nHere are the recent manga: \n" + list)


bot_name = "Mangaka"
print("Let's chat! (type 'quit' to exit)")
while True:
    sentence = input("You: ")
    if sentence == "quit":
        break

    sentence = tokenize(sentence)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == "topmanga" and tag == intent["tag"]:
                top_manga()
            elif tag == "recentmanga" and tag == intent["tag"]:
                recent_manga()
            elif tag == intent["tag"]:
                print(f"{bot_name}: {random.choice(intent['responses'])}")
    else:
        print(f"{bot_name}: I do not understand...")
