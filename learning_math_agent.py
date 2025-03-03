import os
import operator
from typing import Literal
from langchain_anthropic import ChatAnthropic
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
import uuid
from getpass import getpass
from typing import Annotated, Dict, List, Optional, Tuple, TypedDict
from typing import List
import ipywidgets as widgets
from IPython.display import display
from pydantic import BaseModel
from IPython.display import Image, display
from langchain_community.utils.math import cosine_similarity
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.memory import InMemoryStore
from langgraph.graph import END, START, StateGraph
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import semantic_chunkers
from semantic_chunkers import StatisticalChunker
#from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai.embeddings import OpenAIEmbeddings
from semantic_router.encoders import OpenAIEncoder
load_dotenv()
from openai import OpenAI
OpenAI

os.environ["OPENAI_API_KEY"]='sk-'
TAVILY_API_KEY = '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'
os.environ["TAVILY_API_KEY"] = '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'

tavily_search = TavilySearchResults(max_results=3, api_key = TAVILY_API_KEY)
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0)
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


class Goals(BaseModel):
    """Structure for defining learning goals"""

    goals: str = Field(None, description="Learning goals")

class LearningCheckpoint(BaseModel):
    """Structure for a single checkpoint"""
    description: str = Field(..., description="Main checkpoint description")
    criteria: List[str] = Field(..., description="List of success criteria")
    verification: str = Field(..., description="How to verify this checkpoint")

class Checkpoints(BaseModel):
    """Main checkpoints container with index tracking"""
    checkpoints: List[LearningCheckpoint] = Field(
        ...,
        description="List of checkpoints covering foundation, application, and mastery levels"
    )

class SearchQuery(BaseModel):
    """Structure for search query collection"""
    search_queries: list = Field(None, description="Search queries for retrieval.")

class LearningVerification(BaseModel):
    """Structure for verification results"""
    understanding_level: bool #float = Field(..., ge=0, le=1)
    feedback: str
    suggestions: List[str]
    context_alignment: bool


class MathTeaching(BaseModel):
    """Structure for math teaching method"""
    simplified_explanation: str
    key_concepts: List[str]
    #analogies: List[str]

class QuestionOutput(BaseModel):
    """Structure for question generation output"""
    question: str

class InContext(BaseModel):
    """Structure for context verification"""
    is_in_context: str = Field(..., description="Yes or No")


class RouteQuery(BaseModel):
    """Route a user query to the most relevant datasource."""
    route: Literal["search", "ordinary"] = Field(
        ...,
        description="Given a user question choose to route it to a tool or a ordinary question.",
    )
class LearningtState(TypedDict):
    topic: str
    goals: List[Goals]
    context: str
    context_chunks: Annotated[list, operator.add]
    context_key: str
    search_queries: SearchQuery
    checkpoints: Checkpoints
    verifications: LearningVerification
    teachings: MathTeaching
    current_checkpoint: int
    current_question: QuestionOutput
    current_answer: str


def extract_content_from_chunks(chunks):
    """Extract and combine content from chunks with splits attribute.

    Args:
        chunks: List of chunk objects that may contain splits attribute

    Returns:
        str: Combined content from all chunks joined with newlines
    """
    content = []

    for chunk in chunks:
        if hasattr(chunk, 'splits') and chunk.splits:
            chunk_content = ' '.join(chunk.splits)
            content.append(chunk_content)

    return '\n'.join(content)


def format_checkpoints_as_message(checkpoints: Checkpoints) -> str:
    """Convert Checkpoints object to a formatted string for the message.

    Args:
        checkpoints (Checkpoints): Checkpoints object containing learning checkpoints

    Returns:
        str: Formatted string containing numbered checkpoints with descriptions and criteria
    """
    message = "Here are the learning checkpoints:\n\n"
    for i, checkpoint in enumerate(checkpoints.checkpoints, 3):
        message += f"Checkpoint {i}:\n"
        message += f"Description: {checkpoint.description}\n"
        message += "Success Criteria:\n"
        for criterion in checkpoint.criteria:
            message += f"- {criterion}\n"
    return message


def generate_checkpoint_message(checks: List[LearningCheckpoint]) -> HumanMessage:
    """Generate a formatted message for learning checkpoints that need context.

    Args:
        checks (List[LearningCheckpoint]): List of learning checkpoint objects

    Returns:
        HumanMessage: Formatted message containing checkpoint descriptions, criteria and
                     verification methods, ready for context search
    """
    formatted_checks = []
    for check in checks:
        checkpoint_text = f"""
        Description: {check.description}
        Success Criteria:
        {chr(10).join(f'- {criterion}' for criterion in check.criteria)}
        Verification Method: {check.verification}
        """
        formatted_checks.append(checkpoint_text)

    all_checks = "\n---\n".join(formatted_checks)

    checkpoints_message = HumanMessage(content=f"""The following learning checkpoints need additional context:
        {all_checks}

        Please generate search queries to find relevant information.""")

    return checkpoints_message

#dont ask for Verification by substitution or definitions, only final results
learning_checkpoints_generator = SystemMessage(content="""You will be given a learning topic title and learning objectives in hebrew
Your goal is to generate clear learning checkpoints that will help verify understanding and progress through the topic.
The output should be in the following dictionary structure:
checkpoint 
-> description (level checkpoint description)    
-> criteria 
-> verification 
Requirements for each checkpoint:
- Description should be clear and concise, expecting to solve exercises according to the topic ,No ask for explanations, definitions or full solutions 
- Criteria should be technical understanding, solving calculative exercises, No need for full solution and placing  
- Verification method should be practical and appropriate for the level, No need for full solution and placing.  
- Verification will be double checked  by language model and math websites 
- All elements should align with the learning objectives
- Use action verbs and clear language
Ensure all checkpoints progress logically from foundation to mastery.
IMPORTANT - ANSWER ONLY 10 CHECKPOINTS""")

checkpoint_based_query_generator = SystemMessage(content="""You will be given learning checkpoints for a topic.
Your goal is to generate search queries that will retrieve content matching each checkpoint's requirements from retrieval systems or web search.
Follow these steps:
1. Analyze each learning checkpoint carefully.
2. For each checkpoint, generate ONE targeted search query that will retrieve:
   - Content for checkpoint verification""")

validate_context = SystemMessage(content="""You will be given a learning criteria and context.
Check if the the criteria could be answered using the context.
Always answer YES or NO""")

question_generator = SystemMessage(content="""You will be given a checkpoint description, success criteria, and verification method.
Your goal is to generate an appropriate question that aligns with the checkpoint's verification requirements.the difficulty of the questions should be incremental
The question should:
1. Follow the specified verification method
2. Cover all success criteria
3. Be clear and specific
4. No graph requests
5. no need for full solution
Output should be a single, strait forward  question that effectively tests the checkpoint's level requiring only end solution .""")

answer_verifier = SystemMessage(content="""You will be given a student's answer, question, checkpoint details, and relevant context.
Your goal is to verify that the result is correct.


Output should include:

- feedback: detailed explanation after double check yourself if the feedback is correct 
- suggestions: list of specific improvements
- context_alignment: boolean indicating if the answer aligns with provided context""")

Math_teacher = SystemMessage(content="""You will be given verification results, checkpoint criteria, and learning context.
Your mission is to generate  exercises to master a subject gradually and reveal the full solution after the student answered the question.
The explanation should include:
1. Simplified explanation 
2. Key concepts to remember
Output should follow :
- simplified_explanation: clear
- key_concepts: list of essential points
Focus on making complex ideas accessible and memorable.""")

class ContextStore:
    """Store for managing context chunks and their embeddings in memory.

    A class that provides storage and retrieval of context data using an in-memory store.
    Each context entry consists of context chunks and their corresponding embeddings.
    """

    def __init__(self):
        """Initialize ContextStore with an empty in-memory store."""
        self.store = InMemoryStore()

    def save_context(self, context_chunks: list, embeddings: list, key: str = None):
        """Save context chunks and their embeddings to the store.

        Args:
            context_chunks (list): List of context chunk objects
            embeddings (list): List of corresponding embeddings for the chunks
            key (str, optional): Custom key for storing the context. Defaults to None,
                               in which case a UUID is generated.

        Returns:
            str: The key used to store the context
        """
        namespace = ("context",)

        if key is None:
            key = str(uuid.uuid4())

        value = {
            "chunks": context_chunks,
            "embeddings": embeddings
        }

        self.store.put(namespace, key, value)
        return key

    def get_context(self, context_key: str):
        """Retrieve context data from the store using a key.

        Args:
            context_key (str): The key used to store the context

        Returns:
            dict: The stored context value containing chunks and embeddings
        """
        namespace = ("context",)
        memory = self.store.get(namespace, context_key)
        return memory.value


def generate_query(state: LearningtState):
    """Generates search queries based on learning checkpoints from current state."""
    structured_llm = llm.with_structured_output(SearchQuery,method="function_calling")
    checkpoints_message = HumanMessage(content=format_checkpoints_as_message(state['checkpoints']))
    messages = [checkpoint_based_query_generator, checkpoints_message]
    search_queries = structured_llm.invoke(messages)
    return {"search_queries": search_queries}


def search_web(state: LearningtState):
    """Retrieves and processes web search results based on search queries."""
    search_queries = state["search_queries"].search_queries

    all_search_docs = []
    for query in search_queries:
        search_docs = tavily_search.invoke(query)
        all_search_docs.extend(search_docs)

    formatted_search_docs = [
        f'Context: {doc["content"]}\n Source: {doc["url"]}\n'
        for doc in all_search_docs
    ]

    chunk_embeddings = embeddings.embed_documents(formatted_search_docs)
    context_key = context_store.save_context(
        formatted_search_docs,
        chunk_embeddings,
        key=state.get('context_key'),

    )

    return {"context_chunks": formatted_search_docs}

#,method="function_calling"
def generate_checkpoints(state: LearningtState):
    """Creates learning checkpoints based on given topic and goals."""
    structured_llm = llm.with_structured_output(Checkpoints,method="function_calling")
    messages = [
        learning_checkpoints_generator,
        SystemMessage(content=f"Topic: {state['topic']}"),
        SystemMessage(content=f"Goals: {', '.join(str(goal) for goal in state['goals'])}")
    ]
    print('messages', messages)
    checkpoints = structured_llm.invoke(messages)
    return {"checkpoints": checkpoints}


def chunk_context(state: LearningtState):
    """Splits context into manageable chunks and generates their embeddings."""
    encoder = OpenAIEncoder(name="text-embedding-3-large")
    chunker = StatisticalChunker(
        encoder=encoder,
        min_split_tokens=128,
        max_split_tokens=512
    )

    chunks = chunker([state['context']])
    content = []
    for chunk in chunks:
        content.append(extract_content_from_chunks(chunk))

    chunk_embeddings = embeddings.embed_documents(content)
    context_key = context_store.save_context(
        content,
        chunk_embeddings,
        key=state.get('context_key')
    )
    return {"context_chunks": content, "context_key": context_key}


def context_validation(state: LearningtState):
    """Validates context coverage against checkpoint criteria using stored embeddings."""
    context = context_store.get_context(state['context_key'])
    chunks = context['chunks']
    chunk_embeddings = context['embeddings']

    checks = []
    #llm.with_structured_output(RouteQuery, method="function_calling")
    structured_llm = llm.with_structured_output(InContext,method="function_calling")

    for checkpoint in state['checkpoints'].checkpoints:
        query = embeddings.embed_query(checkpoint.verification)

        similarities = cosine_similarity([query], chunk_embeddings)[0]
        top_3_indices = sorted(range(len(similarities)),
                               key=lambda i: similarities[i],
                               reverse=True)[:3]
        relevant_chunks = [chunks[i] for i in top_3_indices]


        messages = [
            validate_context,
            HumanMessage(content=f"""
            Criteria:
            {chr(10).join(f"- {c}" for c in checkpoint.criteria)}

            Context:
            {chr(10).join(relevant_chunks)}
            """)
        ]

        response = structured_llm.invoke(messages)
        if response.is_in_context.lower() == "no":
            checks.append(checkpoint)


    if checks:
        structured_llm = llm.with_structured_output(SearchQuery,method="function_calling" )
        checkpoints_message = generate_checkpoint_message(checks)

        messages = [checkpoint_based_query_generator, checkpoints_message]
        search_queries = structured_llm.invoke(messages)
        return {"search_queries": search_queries}
    return {"search_queries": None}


def generate_question(state: LearningtState):
    """Generates assessment questions based on current checkpoint verification requirements."""
    structured_llm = llm.with_structured_output(QuestionOutput,method="function_calling")
    current_checkpoint = state['current_checkpoint']
    checkpoint_info = state['checkpoints'].checkpoints[current_checkpoint]


    messages = [
        question_generator,
        HumanMessage(content=f"""
        Checkpoint Description: {checkpoint_info.description}
        Success Criteria:
        {chr(10).join(f"- {c}" for c in checkpoint_info.criteria)}
        Verification Method: {checkpoint_info.verification}
        Generate an appropriate verification question.""")
    ]

    question_output = structured_llm.invoke(messages)
    return {"current_question": question_output.question}


def verify_answer(state: LearningtState):
    """Evaluates user answers against checkpoint criteria using relevant context chunks."""
    structured_llm = llm.with_structured_output(LearningVerification,method="function_calling")

    current_checkpoint = state['current_checkpoint']
    checkpoint_info = state['checkpoints'].checkpoints[current_checkpoint]
    context = context_store.get_context(state['context_key'])
    chunks = context['chunks']
    chunk_embeddings = context['embeddings']
    query = embeddings.embed_query(checkpoint_info.verification)
    similarities = cosine_similarity([query], chunk_embeddings)[0]
    top_3_indices = sorted(range(len(similarities)),
                           key=lambda i: similarities[i],
                           reverse=True)[:3]

    relevant_chunks = [chunks[i] for i in top_3_indices]
    messages = [
        answer_verifier,
        HumanMessage(content=f"""
        Question: {state['current_question']}
        Answer: {state['current_answer']}

        Checkpoint Description: {checkpoint_info.description}
        Success Criteria:
        {chr(10).join(f"- {c}" for c in checkpoint_info.criteria)}
        Verification Method: {checkpoint_info.verification}

        Context:
        {chr(10).join(relevant_chunks)}

        Assess the answer.""")
    ]

    verification = structured_llm.invoke(messages)
    return {"verifications": verification}


def teach_concept(state: LearningtState):
    """Creates simplified Math teaching explanations for concepts that need reinforcement."""
    structured_llm = llm.with_structured_output(MathTeaching,method="function_calling")
    current_checkpoint = state['current_checkpoint']
    checkpoint_info = state['checkpoints'].checkpoints[current_checkpoint]

    messages = [
        Math_teacher,
        HumanMessage(content=f"""
        Criteria: {checkpoint_info.criteria}
        Verification: {state['verifications']}

        Context:
        {state['context_chunks']}

        Create a step by step  exercises.""")
    ]


    teaching = structured_llm.invoke(messages)
    return {"teachings": teaching}

def user_answer(state: LearningtState):
    """Placeholder for handling user's answer input."""
    pass

def next_checkpoint(state: LearningtState):
    """Advances to the next checkpoint in the learning sequence."""
    current_checkpoint = state['current_checkpoint'] + 1
    return {'current_checkpoint': current_checkpoint}


def route_context(state: LearningtState):
    """Determines whether to process existing context or generate new search queries."""
    if state.get("context"):
        return 'chunk_context'
    return 'generate_query'


def route_verification(state: LearningtState):
    """Determines next step based on verification results and checkpoint progress."""
    current_checkpoint = state['current_checkpoint']
    if state['verifications'].understanding_level == False:
        return 'teach_concept'

    if current_checkpoint + 1 < len(state['checkpoints'].checkpoints):
        return 'next_checkpoint'

    return END


def route_teaching(state: LearningtState):
    """Routes to next checkpoint or ends session after teaching intervention."""
    current_checkpoint = state['current_checkpoint']
    if current_checkpoint + 1 < len(state['checkpoints'].checkpoints):
        return 'next_checkpoint'
    return END


def route_search(state: LearningtState):
    """Directs flow between question generation and web search based on query status."""
    if state['search_queries'] is None:
        return "generate_question"
    return "search_web"

searcher = StateGraph(LearningtState)
memory = MemorySaver()
context_store = ContextStore()

searcher.add_node("generate_query", generate_query)
searcher.add_node("search_web", search_web)
searcher.add_node("chunk_context", chunk_context)
searcher.add_node("context_validation", context_validation)
searcher.add_node("generate_checkpoints", generate_checkpoints)
searcher.add_node("generate_question", generate_question)
searcher.add_node("next_checkpoint", next_checkpoint)
searcher.add_node("user_answer", user_answer)
searcher.add_node("verify_answer", verify_answer)
searcher.add_node("teach_concept", teach_concept)

# Flow
searcher.add_edge(START, "generate_checkpoints")
searcher.add_conditional_edges('generate_checkpoints', route_context,['chunk_context', 'generate_query'])
searcher.add_edge("generate_query", "search_web")
searcher.add_edge("search_web", "generate_question")
searcher.add_edge("chunk_context", 'context_validation')
searcher.add_conditional_edges('context_validation', route_search,['search_web', 'generate_question'])

searcher.add_edge("generate_question", "user_answer")
searcher.add_edge("user_answer", "verify_answer")
searcher.add_conditional_edges(
    "verify_answer",
    route_verification,
    {
        "next_checkpoint": "next_checkpoint",
        "teach_concept": "teach_concept",
        END: END
    }
)

searcher.add_conditional_edges(
    "teach_concept",
    route_teaching,
    {
        "next_checkpoint": "next_checkpoint",
        END: END
    }
)
searcher.add_edge("next_checkpoint", "generate_question")



graph = searcher.compile(interrupt_after=["generate_checkpoints"], interrupt_before=["user_answer"], checkpointer=memory)

#display(Image(graph.get_graph(xray=1).draw_mermaid_png()))
Image(graph.get_graph().draw_mermaid_png())

def print_checkpoints(event):
    """Pretty print checkpoints information with improved visual formatting"""
    checkpoints = event.get('checkpoints', '')
    if checkpoints:
        print("\n" + "=" * 80)
        print("🎯 LEARNING CHECKPOINTS OVERVIEW".center(80))
        print("=" * 80 + "\n")

        for i, checkpoint in enumerate(checkpoints.checkpoints, 10):
            # Checkpoint header with number
            print(f"📍 CHECKPOINT #{i}".center(80))
            print("─" * 80 + "\n")

            # Description section with text wrapping
            print("📝 Description:")
            print("─" * 40)
            words = checkpoint.description.split()
            current_line = []
            current_length = 0

            for word in words:
                if current_length + len(word) + 1 <= 70:
                    current_line.append(word)
                    current_length += len(word) + 1
                else:
                    print(f"  {' '.join(current_line)}")
                    current_line = [word]
                    current_length = len(word)

            if current_line:
                print(f"  {' '.join(current_line)}")
            print()

            # Success Criteria section
            print("✅ Success Criteria:")
            print("─" * 40)
            for j, criterion in enumerate(checkpoint.criteria, 1):
                print('criterion', criterion)
                # Wrap each criterion text
                words = criterion.split()
                current_line = []
                current_length = 0
                first_line = True

                for word in words:
                    if current_length + len(word) + 1 <= 66:  # Shorter width to account for numbering
                        current_line.append(word)
                        current_length += len(word) + 1
                    else:
                        if first_line:
                            print(f"  {j}. {' '.join(current_line)}")
                            first_line = False
                        else:
                            print(f"     {' '.join(current_line)}")
                        current_line = [word]
                        current_length = len(word)

                if current_line:
                    if first_line:
                        print(f"  {j}. {' '.join(current_line)}")
                    else:
                        print(f"     {' '.join(current_line)}")
            print()

            # Verification Method section
            print("🔍 Verification Method:")
            print("─" * 40)
            words = checkpoint.verification.split()
            print('words',words)
            current_line = []
            current_length = 0

            for word in words:
                if current_length + len(word) + 1 <= 70:
                    current_line.append(word)
                    current_length += len(word) + 1
                else:
                    print(f"  {' '.join(current_line)}")
                    current_line = [word]
                    current_length = len(word)

            if current_line:
                print(f"  {' '.join(current_line)}")
            print()

            # Separator between checkpoints
            if i < len(checkpoints.checkpoints):
                print("~" * 80 + "\n")

        print("=" * 80 + "\n")


def print_verification_results(event):
    """Pretty print verification results with improved formatting"""
    verifications = event.get('verifications', '')
    if verifications:
        print("\n" + "=" * 50)
        print("📊 VERIFICATION RESULTS".center(50))
        print("=" * 50 + "\n")

        # Understanding Level with visual bar
        understanding = verifications.understanding_level
        # bar_length = 20
        # filled_length = int(understanding * bar_length)
        # bar = "█" * filled_length + "░" * (bar_length - filled_length)
        print(f"📈 Understanding Level: {understanding}")
        #print(f"📈 Understanding Level: [{bar}] {understanding * 100:.1f}%\n")

        # Feedback section
        print("💡 Feedback:")
        print(f"{verifications.feedback}\n")

        # Suggestions section
        print("🎯 Suggestions:")
        for i, suggestion in enumerate(verifications.suggestions, 1):
            print(f"  {i}. {suggestion}")
        print()

        #Context Alignment
        print("🔍 Context Alignment:")
        print(f"{verifications.context_alignment}\n")

        print("-" * 50 + "\n")


def print_teaching_results(event):
    """Pretty print Math teaching results with improved formatting"""
    teachings = event.get('teachings', '')
    if teachings:
        print("\n" + "=" * 70)
        print("🎓 MATH TEACHING EXPLANATION".center(70))
        print("=" * 70 + "\n")

        # Simplified Explanation section
        print("📚 SIMPLIFIED EXPLANATION:")
        print("─" * 30)
        # Split explanation into paragraphs for better readability
        paragraphs = teachings.simplified_explanation.split('\n')
        for paragraph in paragraphs:
            # Wrap text at 60 characters for better readability
            words = paragraph.split()
            lines = []
            current_line = []
            current_length = 0

            for word in words:
                if current_length + len(word) + 1 <= 60:
                    current_line.append(word)
                    current_length += len(word) + 1
                else:
                    lines.append(' '.join(current_line))
                    current_line = [word]
                    current_length = len(word)

            if current_line:
                lines.append(' '.join(current_line))

            for line in lines:
                print(f"{line}")
            print()

        # Key Concepts section
        print("💡 KEY CONCEPTS:")
        print("─" * 30)
        for i, concept in enumerate(teachings.key_concepts, 1):
            print(f"  {i}. {concept}")
        print()

        # Analogies section
        # print("🔄 ANALOGIES & EXAMPLES:")
        # print("─" * 30)
        # for i, analogy in enumerate(teachings.analogies, 1):
        #     print(f"  {i}. {analogy}")
        # print()

        print("=" * 70 + "\n")

context = '''search in web what are linear functions'''



initial_input = {
    "topic": "linear equations  ",
    'goals': ['to generate exercises for 8 grade   '],
    'context': context,
    'current_checkpoint': 0}

thread = {"configurable": {"thread_id": "20"}}

for event in graph.stream(initial_input, thread, stream_mode="values"):
        print_checkpoints(event)



def create_checkpoint_editor(checkpoints_model: Checkpoints):
    """
    Creates an interactive checkpoint editor using a Pydantic model.

    Args:
        checkpoints_model: Pydantic model of Checkpoints class
    """
    # Convert to list of dictionaries for easier editing
    checkpoints = [cp.model_dump() for cp in checkpoints_model.checkpoints]
    checkpoints_widgets = []
    accepted_checkpoints = []

    def create_criterion_widget(checkpoint_index: int, criterion_value: str = "", criterion_index: int = None):
        """Creates a widget for a single criterion with a delete button"""
        criterion_container = widgets.HBox([
            widgets.Text(
                value=criterion_value,
                description=f'{criterion_index + 1}.' if criterion_index is not None else 'New',
                layout=widgets.Layout(width='85%')
            ),
            widgets.Button(
                description='Delete',
                button_style='danger',
                layout=widgets.Layout(width='15%')
            )
        ])

        def on_criterion_change(change):
            nonlocal criterion_index
            if criterion_index is not None:
                checkpoints[checkpoint_index]['criteria'][criterion_index] = change['new']

        def remove_criterion(b):
            if criterion_index is not None:
                checkpoints[checkpoint_index]['criteria'].pop(criterion_index)
                update_checkpoint_widget(checkpoint_index)

        criterion_container.children[0].observe(on_criterion_change, names='value')
        criterion_container.children[1].on_click(remove_criterion)

        return criterion_container

    def create_checkpoint_widget(checkpoint: dict, index: int):
        """Creates a widget for a single checkpoint"""

        def on_accept_change(change):
            if change['new']:
                accepted_checkpoints.append(index)
            else:
                if index in accepted_checkpoints:
                    accepted_checkpoints.remove(index)

        def on_description_change(change):
            checkpoints[index]['description'] = change['new']

        def on_verification_change(change):
            checkpoints[index]['verification'] = change['new']

        def add_criterion(b):
            checkpoints[index]['criteria'].append("")
            print(checkpoints[index]['criteria'])
            update_checkpoint_widget(index)

        def remove_checkpoint(b):
            checkpoints.pop(index)
            update_all_checkpoints()

        # Header with checkbox and delete button
        header = widgets.HBox([
            widgets.HTML(f'<h3 style="margin: 0;">Checkpoint {index + 1}</h3>'),
            widgets.Checkbox(
                value=False,
                description='Accept',
                indent=False,
                layout=widgets.Layout(margin='5px 0 0 20px')
            ),
            widgets.Button(
                description='Delete checkpoint',
                button_style='danger',
                layout=widgets.Layout(margin='0 0 0 20px')
            )
        ])

        # Description
        description = widgets.Textarea(
            value=checkpoint['description'],
            description='Description:',
            layout=widgets.Layout(width='95%', height='60px')
        )

        # Criteria
        criteria_label = widgets.HTML('<b>Criteria:</b>')
        criteria_container = widgets.VBox([
            create_criterion_widget(index, criterion, i)
            for i, criterion in enumerate(checkpoint['criteria'])
        ])

        # Add criterion button
        add_criterion_btn = widgets.Button(
            description='Add criterion',
            button_style='success',
            layout=widgets.Layout(margin='10px 0')
        )

        # Verification
        verification = widgets.Textarea(
            value=checkpoint['verification'],
            description='Verification:',
            layout=widgets.Layout(width='95%', height='60px', margin='10px 0')
        )

        separator = widgets.HTML('<hr style="margin: 20px 0;">')

        # Combine all elements
        checkpoint_widget = widgets.VBox([
            header,
            description,
            criteria_label,
            criteria_container,
            add_criterion_btn,
            verification,
            separator
        ])

        # Add observers and handlers
        header.children[1].observe(on_accept_change, names='value')
        header.children[2].on_click(remove_checkpoint)
        description.observe(on_description_change, names='value')
        verification.observe(on_verification_change, names='value')
        add_criterion_btn.on_click(add_criterion)

        return checkpoint_widget

    def update_checkpoint_widget(index: int):
        """Updates a single checkpoint widget"""
        if 0 <= index < len(checkpoints):
            checkpoints_widgets[index] = create_checkpoint_widget(checkpoints[index], index)
            update_main_container()

    def update_all_checkpoints():
        """Updates all checkpoint widgets"""
        nonlocal checkpoints_widgets
        checkpoints_widgets = [
            create_checkpoint_widget(checkpoint, i)
            for i, checkpoint in enumerate(checkpoints)
        ]
        update_main_container()

    def add_new_checkpoint(b):
        """Adds a new checkpoint"""
        checkpoints.append({
            'description': '',
            'criteria': [],
            'verification': ''
        })
        update_all_checkpoints()

    def get_pydantic_model() -> Checkpoints:
        """Converts the current editor state back to a Pydantic model"""
        return Checkpoints(checkpoints=[
            LearningCheckpoint(**checkpoint)
            for checkpoint in checkpoints
        ])

    # Create initial checkpoint widgets
    checkpoints_widgets = [
        create_checkpoint_widget(checkpoint, i)
        for i, checkpoint in enumerate(checkpoints)
    ]

    # Add new checkpoint button
    add_checkpoint_btn = widgets.Button(
        description='Add checkpoint',
        button_style='success',
        layout=widgets.Layout(margin='20px 0')
    )
    add_checkpoint_btn.on_click(add_new_checkpoint)

    # Main container
    main_container = widgets.VBox(
        checkpoints_widgets + [add_checkpoint_btn],
        layout=widgets.Layout(
            padding='20px',
            border='1px solid #ddd',
            border_radius='5px'
        )
    )

    def update_main_container():
        """Updates the main container"""
        main_container.children = tuple(checkpoints_widgets + [add_checkpoint_btn])

    # Add method to container to retrieve data later
    main_container.get_model = get_pydantic_model

    return main_container


checkpoints = event['checkpoints']
# print('event',event)
editor = create_checkpoint_editor(checkpoints)
display(editor)


updated_model = editor.get_model()
graph.update_state(thread, {"checkpoints": updated_model}, as_node="generate_checkpoints")
print(graph.stream(None, thread, stream_mode="values"))

for event in graph.stream(None, thread, stream_mode="values"):

    # Review
    current_question = event.get('current_question', '')
    if current_question:
        print(current_question)

answer_question = input("Answer the question above: ")
graph.update_state(thread, {"current_answer": answer_question}, as_node="user_answer")
for event in graph.stream(None, thread, stream_mode="values"):
    print(graph.get_state(thread).next)

print_verification_results(event)
print_teaching_results(event)
context_store.get_context(event['context_key'])['chunks'][:3]
#print(event['current_question'])

for i in range (2,10):
   print(event['current_question'])
   answer_question = input(f"Answer the {i} question ")
   graph.update_state(thread, {"current_answer": answer_question}, as_node="user_answer")
   for event in graph.stream(None, thread, stream_mode="values"):
       graph.get_state(thread).next
   print(print_verification_results(event))
   print(print_teaching_results(event))
   context_store.get_context(event['context_key'])['chunks'][:3]
