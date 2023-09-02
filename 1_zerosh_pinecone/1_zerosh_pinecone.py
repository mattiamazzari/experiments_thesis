from dotenv import load_dotenv,find_dotenv
from langchain.document_loaders import UnstructuredPDFLoader, OnlinePDFLoader, DirectoryLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.chains import LLMChain
from langchain import HuggingFacePipeline
from transformers import AutoTokenizer
from langchain.chains import SimpleSequentialChain
from instruct_pipeline import InstructionTextGenerationPipeline
import transformers
import torch
import pinecone
import os
import sys

def main():
    load_dotenv(find_dotenv())
    pdf_path = sys.argv[1]
    loader = PyPDFLoader(pdf_path)

    ## Other options for loaders
    # loader = UnstructuredPDFLoader("../data/field-guide-to-data-science.pdf")
    ### This one is used to load an online pdf:
    #loader = OnlinePDFLoader("https://wolfpaulus.com/wp-content/uploads/2017/05/field-guide-to-data-science.pdf")

    data = loader.load()

    # Note: If you're using PyPDFLoader then it will split by page for you already
    print (f'You have {len(data)} document(s) in your data')
    #print (f'There are {len(data[30].page_content)} characters in your document')
    
    #directory = '/content/data'
    #documents = load_docs(directory)
    #len(documents)
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=0)
    texts = text_splitter.split_documents(data)
    print (f'Now you have {len(texts)} documents')
    
    # Check to see if there is an environment variable with your API keys, if not, use what you put below
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY', '9fa8ba9d-344d-4466-8e7e-78f825ad7caf')
    PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV', 'gcp-starter') # You may need to switch with your env

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')

    # initialize pinecone
    pinecone.init(
        api_key=PINECONE_API_KEY,  # find at app.pinecone.io
        environment=PINECONE_API_ENV  # next to api key in console
    )

    index_name = "example" # put in the name of your pinecone index here
    docsearch = Pinecone.from_texts([t.page_content for t in texts], embeddings, index_name=index_name)

    query = "What category does this insurance claim belong to?"
    query_2 = "What subcategory does this insurance claim belong to?"
    docs_for_category = docsearch.similarity_search(query)
    docs_for_subcategory = docsearch.similarity_search(query_2)
    
    name = 'mosaicml/mpt-7b'

    config = transformers.AutoConfig.from_pretrained(name, trust_remote_code=True)
    #config.max_seq_len = 8192
    #config.attn_config['attn_impl'] = 'triton'  # change this to use triton-based FlashAttention
    #config.init_device = 'cuda:0'  # For fast initialization directly on GPU!

    load_8bit = True
    tokenizer = AutoTokenizer.from_pretrained(name)  # , padding_side="left")
    model = transformers.AutoModelForCausalLM.from_pretrained(
        name,
        config=config,
        torch_dtype=torch.bfloat16,  # Load model weights in bfloat16
        trust_remote_code=True,
        load_in_8bit=load_8bit,
        device_map="auto",
    )

    model.eval()
    if torch.__version__ >= "2":
        model = torch.compile(model)

    print("--PIPELINE INIT--")
    pipeline = InstructionTextGenerationPipeline(model=model, tokenizer=tokenizer)

    llm = HuggingFacePipeline(pipeline = pipeline, model_kwargs = {'temperature':0})
    
    first_context = """You are a text classifier of insurance claims. Given the text of the insurance claim delimited by triple backquotes, classify and label
    the insurance claim with one of these three classes: Polizze, Sinistri and Area Commerciale.
    Here is a detailed description of the meaning of each class:
    The 'Polizze' class groups all the insurance claims sent by customers to report issues related to the contract established between the customer and the referring insurance company are included.
    This includes any requested insurance type: life insurance, auto insurance, health insurance, and more.
    The 'Sinistri' class encompasses all cases where customers contest issues related to incidents, which may include accidental damages and road accidents.
    he 'Area Commerciale' class groups all the insurance claims are reviewed where customers contest the lack of assistance from the insurance company despite repeated requests.
    Additionally, this section includes customer inquiries, such as requests for information about insurance types, quotation requests, issues related to the website and application
    """

    first_prompt = PromptTemplate(
        template=f"{first_context}\n\n```{{insurance_claim}}```\n\nQuery: {query}\nAnswer in Italian:"
    )
    chain = load_qa_chain(llm, prompt=first_prompt, output_key="result_macrocategory_classification")
    
    # Run the first chain to classify the macrocategory
    macrocategory_result = chain({"input_documents": docs_for_category, "first_query": query}, return_only_outputs=True)
    
    # Define the second context and query based on the macrocategory result
    if macrocategory_result == "Polizze":
        second_context = """
        You are again a text classifier. Given your previous answer containing the class assigned to the insurance claim,
        classify the subcategory for the Polizze category. The possible subcategories for Polizze are: Subcategory_A, Subcategory_B, Subcategory_C.
        """
    elif macrocategory_result == "Sinistri":
        second_context = """
        You are again a text classifier. Given your previous answer containing the class assigned to the insurance claim,
        classify the subcategory for the Sinistri category. The possible subcategories for Sinistri are: Subcategory_X, Subcategory_Y, Subcategory_Z.
        """
    elif macrocategory_result == "Area Commerciale":
        second_context = """
        You are again a text classifier. Given your previous answer containing the class assigned to the insurance claim,
        classify the subcategory for the Area Commerciale category. The possible subcategories for Area Commerciale are: Subcategory_P, Subcategory_Q, Subcategory_R.
        """
    else:
        # Handle the case where macrocategory classification is unknown
        second_context = "Unable to determine subcategory without macrocategory classification."

    second_prompt = PromptTemplate(
        template=f"{second_context}\n\nQuery: {query_2}\nAnswer in Italian:"
    )

    chain_two = load_qa_chain(pipeline=pipeline, prompt=second_prompt, output_key="result_subcategory_classification")

    # Run the second chain to classify the subcategory
    subcategory_result = chain_two({"input_documents": docs_for_subcategory, "second_query": query_2}, return_only_outputs=True)

    print(f"Macrocategory Classification: {macrocategory_result}")
    print(f"Subcategory Classification: {subcategory_result}")
    
    # Define a sequential chain using the two chains above: the second chain takes the output of the first chain as input
    overall_chain = SimpleSequentialChain(chains=[chain, chain_two], verbose=True)

    """
    overall_chain = SequentialChain(
        chains=[synopsis_chain, review_chain],
        input_variables=["era", "title"],
        # Here we return multiple variables
        output_variables=["synopsis", "review"],
        verbose=True)
    """

    # Run the chain specifying only the input variable for the first chain.
    overall_chain({"input_documents": docs_for_category, "first_query": query_1, "second_query": query_2})

def load_docs(directory):
  loader = DirectoryLoader(directory)
  documents = loader.load()
  return documents

if __name__ == "__main__":
    main()