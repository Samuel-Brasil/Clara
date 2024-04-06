# Step 1: PDF Upload
uploaded_file = st.file_uploader("Escolha um arquivo em pdf", type="pdf")
if uploaded_file is not None:
    # Step 2: Read Data from the PDF
    reader = PdfReader(uploaded_file)
    pdf_text = ''
    for page in reader.pages:
        text = page.extract_text()
        if text:
            pdf_text += text

    # Step 3: Define and Apply Text Splitter
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,  # thousand characters
        chunk_overlap=200,
        length_function=len,
    )
    text_chunks = text_splitter.split_text(pdf_text)

    # Step 4: Convert Text to Embeddings
    embeddings = OpenAIEmbeddings()
    pdf_embeddings = FAISS.from_texts(text_chunks, embeddings)

    # Step 5: Load the QA Chain
    chain = load_qa_chain(OpenAI(), chain_type="stuff")

    # Store answers to display after UI elements
    answers = []

    # Step 6: Predefined Query Examples
    query_examples = [
        ("Explicação com Linguagem Simples", "Faça uma explicação detalhada, sem deixar nada de fora, explicando a decisão em linguagem simples e clara, de modo que uma pessoa leiga consiga entender"),
        ("Resultado do Julgamento", "Explique o resultado do julgamento, usando linguagem simples e clara"),
    ]
    for button_label, query in query_examples:
        if st.button(button_label):
            docs = pdf_embeddings.similarity_search(query)
            answer = chain.run(input_documents=docs, question=query)
            answers.append(answer)

    # Step 7: Interface for Question-Answering
    text_input = st.text_input("Pergunte CLARA sobre o caso:", )
    if text_input:
        docs = pdf_embeddings.similarity_search(text_input)
        answer = chain.run(input_documents=docs, question=text_input)
        answers.append(answer)

    # Display answers
    for answer in answers:
        st.write(answer)
