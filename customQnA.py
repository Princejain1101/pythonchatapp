import numpy as np
from embeddingUtils import distances_from_embeddings
import pandas as pd
from openai import OpenAI
import os

def read_embeddings(process_dir="processed/"):
    df=pd.read_csv(process_dir + 'embeddings.csv', index_col=0)
    df['embeddings'] = df['embeddings'].apply(eval).apply(np.array)

    df.head()
    return df

client = OpenAI(
    api_key=os.environ.get("OPENAI_API_KEY"),
)

def create_context(
    question, df, max_len=1800, size="ada"
):
    """
    Create a context for a question by finding the most similar context from the dataframe
    """

    # Get the embeddings for the question
    q_embeddings = client.embeddings.create(input=question, model='text-embedding-ada-002').data[0].embedding

    # Get the distances from the embeddings
    df['distances'] = distances_from_embeddings(q_embeddings, df['embeddings'].values, distance_metric='cosine')


    returns = []
    cur_len = 0

    urls = []
    # Sort by distance and add the text to the context until the context is too long
    for i, row in df.sort_values('distances', ascending=True).iterrows():

        # Add the length of the text to the current length
        cur_len += row['n_tokens'] + 4

        # If the context is too long, break
        if cur_len > max_len:
            break

        # Else add it to the text that is being returned
        returns.append(row["text"])
        urls.append(row["url"])
    # Return the context
    urls = set(urls)
    return "\n\n###\n\n".join(returns), ", ".join(urls)

def answer_question(
    df,
    model="gpt-3.5-turbo",
    question="Am I allowed to publish model outputs to Twitter, without a human review?",
    max_len=1800,
    size="ada",
    debug=False,
    max_tokens=150,
    stop_sequence=None
):
    """
    Answer a question based on the most similar context from the dataframe texts
    """
    context, urls = create_context(
        question,
        df,
        max_len=max_len,
        size=size,
    )

    # If debug, print the raw model response
    if debug:
        print("Context:\n" + context)
        print("\n\n")
# completion = openai.ChatCompletion.create(
#        model="gpt-3.5-turbo",
#        messages=[{"role": "system", "content": f"Answer the question as truthfully as possible, and if you're unsure of the answer, say \"Sorry, I don't know\".\\n{context}\n\n{conversation_history}"},
#                           {"role": "user", "content": f"{question}"}]
# )

    try:
        # Create a chat completion using the question and context
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": f"Answer the question as truthfully as possible, and if you're unsure of the answer, say \"Sorry, I don't know\".\\n{context}\n\n"},
                {"role": "user", "content": f"{question}"}
            ],
            # messages=[
            #     {"role": "system", "content": "Answer the question based on the context below\n\n"},
            #     {"role": "user", f"content": "Context: {context}\n\n---\n\nQuestion: {question}\nAnswer:"}
            # ],
            temperature=1,
            max_tokens=max_tokens,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
            stop=stop_sequence,
        )
        return response.choices[0].message.content + " (Sources: " + urls + ")"
    except Exception as e:
        print(e)
        return ""

# print(answer_question(df, question="What day is it?", debug=False))

# print(answer_question(df, question="Who is harshal patil?"))

# print(answer_question(df, question="Learning and Tips from Reconnecting with social and work?"))
