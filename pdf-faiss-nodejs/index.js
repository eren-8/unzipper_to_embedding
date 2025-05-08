const fs = require("fs");
const pdfParse = require("pdf-parse");
const axios = require("axios");

// Step 1: Read PDF
async function extractTextFromPDF(filePath) {
    const dataBuffer = fs.readFileSync(filePath);
    const data = await pdfParse(dataBuffer);
    return data.text;
}

// Step 2: Chunk text
function chunkText(text, chunkSize = 1000, chunkOverlap = 200) {
    const chunks = [];
    let start = 0;
    while (start < text.length) {
        const end = Math.min(start + chunkSize, text.length);
        chunks.push(text.slice(start, end));
        start += chunkSize - chunkOverlap;
    }
    return chunks;
}

// Step 3: Call Python embed server
async function embedChunks(chunks) {
    const res = await axios.post("http://127.0.0.1:3000/embed", {
        texts: chunks,
    });
    return res.data.embeddings;
}

// Step 4: Save to JSON
function saveEmbeddingsWithChunks(chunks, embeddings, outPath = "embedding_store.json") {
    const data = chunks.map((chunk, idx) => ({
        text: chunk,
        embedding: embeddings[idx]
    }));
    fs.writeFileSync(outPath, JSON.stringify(data, null, 2));
    console.log(`🧾 Saved ${data.length} entries to ${outPath}`);
    console.log(data.text);
}

// Run everything
async function run(pdfPath) {
    console.log("📄 Reading PDF...");
    const text = await extractTextFromPDF(pdfPath);

    console.log("✂️ Chunking...");
    const chunks = chunkText(text);
    console.log(`✅ Created ${chunks.length} chunks.`);

    console.log("🧠 Embedding...");
    const embeddings = await embedChunks(chunks);
    console.log(`✅ Got ${embeddings.length} embeddings.`);

    saveEmbeddingsWithChunks(chunks, embeddings);
}

run("sample.pdf"); // <-- make sure this is your actual file name
