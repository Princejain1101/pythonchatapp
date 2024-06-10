from queryModel import queryOpenai
chats = {}

def processRequest(request):
    if request['id'] not in chats:
        chats[request['id']] = []
    chats[request['id']].append(
        {
            "role":"user",
            "content": request['request']
        }
        )
    messages = packageChat(request['id'])
    response = queryOpenai(messages)
    return {
        'id': request['id'],
        'request': response,
    }
def packageChat(id):
    systemInstructions = "You are an Ai assistant that lives in a wix chat widget on a wix website."
    systemMessage = {
        "role": "system",
        "content": systemInstructions
    }
    lastMessages = chats[id][-10:]
    lastMessages.insert(0, systemMessage)
    print(lastMessages)
    return lastMessages
