"""Welcome to Pynecone! This file outlines the steps to create a basic app."""

import os
from datetime import datetime

# Import pynecone.
import pynecone as pc
from langchain.chains import LLMChain, ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.document_loaders import (
    TextLoader
)
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.prompts.chat import ChatPromptTemplate
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import Chroma
from pynecone.base import Base

# openai.api_key = "<YOUR_OPENAI_API_KEY>"
f = open("./chatbot/apiKey.txt", 'r')
line = f.readline()
f.close()
os.environ["OPENAI_API_KEY"] = line

LOADER_DICT = {
    "txt": TextLoader,
}
CUR_DIR = os.path.dirname(os.path.join(os.path.abspath("./chatbot")))
DATA_DIR = os.path.join(CUR_DIR, "data/")
CHROMA_PERSIST_DIR = os.path.join(DATA_DIR, "upload/chroma-persist")
CHROMA_COLLECTION_NAME = "kakao-bot"

PROMPT_DIR = os.path.join(CUR_DIR, "prompt/")
PARSE_INTENT_PROMPT_TEMPLATE = os.path.join(PROMPT_DIR, "parse_intent.txt")
INTENT_LIST = os.path.join(PROMPT_DIR, "intent.txt")
FIND_FROM_INFO_PROMPT_TEMPLATE = os.path.join(PROMPT_DIR, "find_from_info.txt")

llmModel = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")


def read_prompt_template(file_path: str) -> str:
    with open(file_path, "r") as file:
        prompt_template = file.read()

    return prompt_template


def create_chain(template_path, output_key):
    return LLMChain(
        llm=llmModel,
        prompt=ChatPromptTemplate.from_template(
            template=read_prompt_template(template_path)
        ),
        output_key=output_key,
        verbose=True,
    )


parse_intent_chain = create_chain(
    template_path=PARSE_INTENT_PROMPT_TEMPLATE,
    output_key="intent"
)

find_answer_chain = create_chain(
    template_path=FIND_FROM_INFO_PROMPT_TEMPLATE,
    output_key="answer"
)

default_chain = ConversationChain(llm=llmModel, output_key="text")

_db = Chroma(
    persist_directory=CHROMA_PERSIST_DIR,
    embedding_function=OpenAIEmbeddings(),
    collection_name=CHROMA_COLLECTION_NAME,
)
_retriever = _db.as_retriever()


def upload_embedding_from_file(file_path):
    loader = LOADER_DICT.get(file_path.split(".")[-1])
    if loader is None:
        raise ValueError("Not supported file type")
    documents = loader(file_path).load()

    text_splitter = CharacterTextSplitter(chunk_size=500,
                                          chunk_overlap=100)  # 잘라서 주고, 그 사이에 overlap 줘서 자른다 (의미 손실을 막기 위해서).
    docs = text_splitter.split_documents(documents)

    Chroma.from_documents(  # ChromaDB에 저장함.
        docs,
        OpenAIEmbeddings(),  # openai의 임베딩 사용
        collection_name=CHROMA_COLLECTION_NAME,
        persist_directory=CHROMA_PERSIST_DIR,
    )
    print('db success')


def upload_embeddings_from_dir(dir_path):
    failed_upload_files = []

    for root, dirs, files in os.walk(dir_path):
        for file in files:
            if file.endswith(".txt"):
                file_path = os.path.join(root, file)

                try:
                    upload_embedding_from_file(file_path)
                    print("SUCCESS: ", file_path)
                except Exception as e:
                    print("FAILED: ", file_path + f"by({e})")
                    failed_upload_files.append(file_path)


upload_embeddings_from_dir(os.path.relpath('./data/'))


def query_db(query: str, use_retriever: bool = False) -> list[str]:
    if use_retriever:
        docs = _retriever.get_relevant_documents(query)
    else:
        docs = _db.similarity_search(query)

    str_docs = [doc.page_content for doc in docs]
    return str_docs


def generate_answer(user_message) -> dict[str, str]:
    context = dict(user_message=user_message)
    context["input"] = context["user_message"]
    context["intent_list"] = read_prompt_template(INTENT_LIST)

    intent = parse_intent_chain.run(context)

    if intent == "kakao_channel" or intent == "kakao_sync" or intent == "kakao_social":
        context["related_document"] = query_db(context["user_message"])
        answer = find_answer_chain.run(context)
    else:
        answer = default_chain.run(context["user_message"])

    return {"answer": answer}


def ask_to_chatbot(text):
    answer = generate_answer(text)
    return answer['answer']


#
# def make_prompt_template() -> PromptTemplate:
#     return PromptTemplate(
#         input_variables=["text"],
#         template="사용자의 질문에 대한 적절한 답변을 주어진 정보에서 찾아서 반환해야 한다. \n답변은 두 문단 내로 간략하게 제공한다. \n<정보>: \n"
#                  + kakao_sync_data + "\n <질문>: {text}"
#     )

#
# def ask_about_kakao_sync(text) -> str:
#     prompt_template = make_prompt_template()
#     chain = LLMChain(llm=llm, verbose=True, prompt=prompt_template)
#     return chain.run(text)


class Message(Base):
    original_text: str
    text: str
    created_at: str


class State(pc.State):
    """The app state."""

    text: str = ""
    messages: list[Message] = []
    answer = "Answer will appear here."

    def output(self):
        if not self.text.strip():
            return "Answer will appear here."
        self.answer = ask_to_chatbot(self.text)

    def post(self):
        self.output()
        self.messages = [
                            Message(
                                original_text=self.text,
                                text=self.answer,
                                created_at=datetime.now().strftime("%B %d, %Y %I:%M %p"),
                            )
                        ] + self.messages


# Define views.


def header():
    """Basic instructions to get started."""
    return pc.box(
        pc.text("Chat Bot 🗿", font_size="2rem"),
        pc.text(
            "Chat with ChatGPT!",
            margin_top="0.5rem",
            color="#666",
        ),
    )


def down_arrow():
    return pc.vstack(
        pc.icon(
            tag="arrow_down",
            color="#666",
        )
    )


def text_box(text):
    return pc.text(
        text,
        background_color="#fff",
        padding="1rem",
        border_radius="8px",
    )


def message(message):
    return pc.box(
        pc.vstack(
            text_box(message.original_text),
            down_arrow(),
            text_box(message.text),
            pc.box(
                pc.text(" · ", margin_x="0.3rem"),
                pc.text(message.created_at),
                display="flex",
                font_size="0.8rem",
                color="#666",
            ),
            spacing="0.3rem",
            align_items="left",
        ),
        background_color="#f5f5f5",
        padding="1rem",
        border_radius="8px",
    )


def smallcaps(text, **kwargs):
    return pc.text(
        text,
        font_size="0.7rem",
        font_weight="bold",
        text_transform="uppercase",
        letter_spacing="0.05rem",
        **kwargs,
    )


def output():
    return pc.box(
        pc.box(
            smallcaps(
                "ANSWER",
                color="#aeaeaf",
                background_color="white",
                padding_x="0.1rem",
            ),
            position="absolute",
            top="-0.5rem",
        ),
        pc.text(State.answer),
        padding="1rem",
        border="1px solid #eaeaef",
        margin_top="1rem",
        border_radius="8px",
        position="relative",
    )


def index():
    """The main view."""
    return pc.container(
        header(),
        pc.input(
            placeholder="Chat with GPT",
            on_blur=State.set_text,
            margin_top="1rem",
            border_color="#eaeaef"
        ),
        output(),
        pc.button("Post", on_click=State.post, margin_top="1rem"),
        pc.vstack(
            pc.foreach(State.messages, message),
            margin_top="2rem",
            spacing="1rem",
            align_items="left"
        ),
        padding="2rem",
        max_width="600px"
    )


# Add state and page to the app.
app = pc.App(state=State)
app.add_page(index, title="ChatBot")
app.compile()
