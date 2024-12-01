This project is an AI-powered application designed to autonomously read and parse multiple PDF documents from a specified website. The application leverages two key Large Language Models (LLMs) to provide efficient information retrieval and comprehension for various sectors such as education, research, and corporate training.

The **Embeddings Model** (HuggingFaceInstructEmbeddings) is responsible for creating meaningful vector representations of the document text. This enables efficient storage and retrieval of the content, forming the foundation for the system's ability to search and access specific information within the PDFs.

The **Conversational Model** (google/flan-t5-xxl) plays a crucial role by providing intelligent, context-aware responses. It retrieves relevant chunks from the vector store and generates precise answers, allowing for an interactive, document-based Q&A experience. Together, these models facilitate advanced conversational capabilities that enable users to query the content of the documents seamlessly. 

This integration of both LLMs allows the system to autonomously generate relevant questions and answers based on the extracted content, enhancing information comprehension and retrieval for users across different sectors.
