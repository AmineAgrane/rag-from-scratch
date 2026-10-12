# RAG From Scratch

LLMs are trained on a large but fixed corpus of data, limiting their ability to reason about private or recent information. Fine-tuning is one way to mitigate this, but is often [not well-suited for factual recall](https://www.anyscale.com/blog/fine-tuning-is-for-form-not-facts) and [can be costly](https://www.glean.com/blog/how-to-build-an-ai-assistant-for-the-enterprise).
Retrieval augmented generation (RAG) has emerged as a popular and powerful mechanism to expand an LLM's knowledge base, using documents retrieved from an external data source to ground the LLM generation via in-context learning. 
These notebooks accompany a [video playlist](https://youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x&feature=shared) that builds up an understanding of RAG from scratch, starting with the basics of indexing, retrieval, and generation. 
![rag_detail_v2](https://github.com/langchain-ai/rag-from-scratch/assets/122662504/54a2d76c-b07e-49e7-b4ce-fc45667360a1)
 
[Video playlist](https://www.youtube.com/playlist?list=PLfaIDFEXuae2LXbO1_PKyVJiQ23ZztA0x)


The fastest fix is to bypass the SDK’s method signature and send the request yourself to the same Azure endpoint, with include_blocks in the JSON body. The TypeError is purely client-side: the Azure SDK’s generated process() method doesn’t declare the parameter. That says nothing yet about whether the Azure backend supports it.

1. Raw HTTP call with include_blocks
python
import httpx, base64

ENDPOINT = "https://<resource>.services.ai.azure.com"
URL = f"{ENDPOINT}/providers/mistral/azure/ocr"   # use the exact path the SDK hits (see note)
API_KEY = "..."

with open("sample.pdf", "rb") as f:
    b64 = base64.b64encode(f.read()).decode()

payload = {
    "model": "mistral-ocr-4-0",
    "document": {
        "type": "document_url",
        "document_url": f"data:application/pdf;base64,{b64}",
    },
    "include_blocks": True,
    "confidence_scores_granularity": "word",
}

r = httpx.post(
    URL,
    json=payload,
    headers={"Authorization": f"Bearer {API_KEY}", "Content-Type": "application/json"},
    timeout=120,
)
r.raise_for_status()
data = r.json()
print(data["pages"][0].get("blocks"))
print(data["pages"][0].get("confidence_scores"))