# BÁO CÁO MÔN HỌC
# XỬ LÝ NGÔN NGỮ TỰ NHIÊN

**Đề tài:** Xây dựng hệ thống chatbot hỗ trợ truy vấn thông tin

---

## MỞ ĐẦU

### Lý do chọn đề tài

Xử lý ngôn ngữ tự nhiên (NLP) là một trong những lĩnh vực phát triển nhanh nhất của trí tuệ nhân tạo hiện đại. Từ các trợ lý ảo như Siri, Google Assistant đến các hệ thống chatbot doanh nghiệp, NLP đã trở thành công nghệ nền tảng trong nhiều ứng dụng thực tế: phân tích cảm xúc khách hàng, hỗ trợ tìm kiếm thông tin, dịch máy, phân loại tài liệu và hỏi đáp tự động.

Trong bối cảnh lượng thông tin trên internet tăng trưởng theo cấp số nhân, người dùng ngày càng cần các công cụ giúp họ tra cứu thông tin nhanh chóng mà không phải tự lọc qua hàng chục trang web. Chatbot thông minh ứng dụng NLP chính là giải pháp cho vấn đề này: người dùng đặt câu hỏi bằng ngôn ngữ tự nhiên, hệ thống tự động tìm kiếm, tổng hợp và trả lời.

### Mục tiêu của bài báo cáo

Mục tiêu của đề tài là xây dựng một sản phẩm demo ứng dụng NLP — cụ thể là hệ thống chatbot có khả năng:
- Hiểu câu hỏi ngôn ngữ tự nhiên (tiếng Việt và tiếng Anh)
- Tự động truy vấn thông tin từ website và REST API
- Tổng hợp thông tin và trả lời có trích dẫn nguồn

Báo cáo phân tích thuật toán NLP cốt lõi được áp dụng là kiến trúc **Transformer** và cơ chế **Self-Attention**, nền tảng của mô hình ngôn ngữ lớn (LLM) **Llama 3.3 70B** được sử dụng trong hệ thống.

### Phạm vi và cấu trúc báo cáo

Báo cáo được chia thành 3 chương:
- **Chương 1**: Cơ sở lý thuyết về NLP, biểu diễn văn bản và thuật toán Transformer
- **Chương 2**: Phân tích bài toán, dữ liệu sử dụng và thiết kế kiến trúc hệ thống
- **Chương 3**: Xây dựng mô hình, cài đặt, demo và đánh giá kết quả

---

## CHƯƠNG 1. CƠ SỞ LÝ THUYẾT

### 1.1. Tổng quan về Xử lý ngôn ngữ tự nhiên

**Định nghĩa:** Xử lý ngôn ngữ tự nhiên (Natural Language Processing — NLP) là lĩnh vực giao thoa giữa khoa học máy tính, trí tuệ nhân tạo và ngôn ngữ học, nghiên cứu cách máy tính hiểu, phân tích, tổng hợp và tạo ra ngôn ngữ của con người.

**Các bài toán NLP phổ biến:**

| Bài toán | Mô tả | Ví dụ ứng dụng |
|----------|-------|----------------|
| Text Classification | Phân loại văn bản vào các nhãn | Lọc spam, phân loại tin tức |
| Sentiment Analysis | Phân tích cảm xúc tích cực/tiêu cực | Đánh giá sản phẩm |
| Named Entity Recognition (NER) | Nhận dạng thực thể có tên | Tên người, địa điểm, tổ chức |
| Machine Translation | Dịch máy tự động | Google Translate |
| **Question Answering** | **Trả lời câu hỏi tự nhiên** | **Chatbot — trọng tâm đề tài** |
| Text Summarization | Tóm tắt văn bản | Tóm tắt bài báo |

**Bài toán Question Answering** là trọng tâm của đề tài: hệ thống nhận đầu vào là câu hỏi ngôn ngữ tự nhiên, xử lý và truy xuất thông tin liên quan, sau đó sinh ra câu trả lời phù hợp. Trong đề tài này, nguồn thông tin là các trang web và API bên ngoài được truy vấn động theo thời gian thực.

---

### 1.2. Biểu diễn văn bản

Để máy tính xử lý được văn bản, cần chuyển đổi ngôn ngữ tự nhiên thành dạng số học. Các phương pháp biểu diễn văn bản phát triển từ đơn giản đến phức tạp:

#### 1.2.1. Bag of Words (BoW)

BoW biểu diễn một văn bản bằng tần suất xuất hiện của từng từ trong từ điển, bỏ qua thứ tự từ.

**Ví dụ:**
- Câu: *"Hà Nội thời tiết đẹp"*
- Vector: `[0, 0, 1, 0, 1, 1, 0, ...]` (1 nếu từ xuất hiện)

**Hạn chế:** Không nắm được ngữ nghĩa ("ngân hàng" — tổ chức tài chính hay bờ sông?), không quan tâm thứ tự từ, vector rất thưa (sparse).

#### 1.2.2. TF-IDF (Term Frequency — Inverse Document Frequency)

TF-IDF cải tiến BoW bằng cách gán trọng số cho từng từ dựa trên tần suất xuất hiện trong văn bản và trong toàn bộ corpus:

$$\text{TF-IDF}(t, d) = \text{TF}(t, d) \times \log\frac{N}{\text{DF}(t)}$$

Trong đó:
- TF(t, d): tần suất từ *t* trong tài liệu *d*
- N: tổng số tài liệu
- DF(t): số tài liệu chứa từ *t*

**Ưu điểm:** Giảm trọng số các từ phổ biến như "là", "và", "của".  
**Hạn chế:** Vẫn không nắm được ngữ nghĩa và ngữ cảnh.

#### 1.2.3. Word Embedding — Word2Vec

Word2Vec (Mikolov et al., 2013) biểu diễn mỗi từ bằng một vector dày đặc (dense vector) trong không gian nhiều chiều. Các từ có nghĩa tương tự sẽ có vector gần nhau.

**Ví dụ:** `vector("vua") - vector("đàn ông") + vector("phụ nữ") ≈ vector("nữ hoàng")`

**Hạn chế:** Mỗi từ chỉ có một vector cố định, không phụ thuộc ngữ cảnh ("ngân hàng" trong mọi câu đều có cùng vector).

#### 1.2.4. Contextual Embedding — Transformer (áp dụng trong đề tài)

Các mô hình dựa trên Transformer (BERT, GPT, Llama) tạo ra **contextual embedding**: vector của một từ thay đổi tùy theo ngữ cảnh xung quanh.

- "Tôi đến **ngân hàng** rút tiền" → vector "ngân hàng" khác
- "Con thuyền cập **ngân hàng** sông" → vector "ngân hàng" khác

Đây là cách **Llama 3.3 70B** — mô hình được sử dụng trong đề tài — biểu diễn và hiểu văn bản. Mô hình sử dụng **Byte-Pair Encoding (BPE)** để tokenize văn bản trước khi tạo embedding.

---

### 1.3. Thuật toán sử dụng trong đề tài: Transformer và Attention

#### 1.3.1. Kiến trúc Transformer

Transformer (Vaswani et al., "Attention Is All You Need", 2017) là kiến trúc deep learning nền tảng của hầu hết các mô hình NLP hiện đại. Transformer gồm hai thành phần chính:

- **Encoder**: mã hóa chuỗi đầu vào thành biểu diễn ẩn (hidden representation)
- **Decoder**: giải mã biểu diễn ẩn thành chuỗi đầu ra

Cấu trúc mỗi tầng (layer):
1. Multi-Head Self-Attention
2. Feed-Forward Network
3. Layer Normalization + Residual Connection

#### 1.3.2. Cơ chế Self-Attention

Self-Attention cho phép mỗi từ trong câu "chú ý" đến tất cả các từ khác để hiểu ngữ nghĩa. Đây là cơ chế cốt lõi phân biệt Transformer với các mô hình trước đó (RNN, LSTM).

**Công thức Self-Attention:**

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$

Trong đó:
- **Q (Query)**: ma trận truy vấn — "từ này đang hỏi gì?"
- **K (Key)**: ma trận khóa — "từ kia có liên quan không?"
- **V (Value)**: ma trận giá trị — "nếu liên quan thì lấy thông tin gì?"
- **d_k**: chiều của vector Key (hệ số scale tránh gradient vanishing)
- **softmax**: chuẩn hóa điểm attention thành xác suất (tổng = 1)

**Diễn giải:** Với câu *"Con mèo ngồi trên thảm vì **nó** mệt"*:
- Q của từ "nó" → K của "con mèo" cho điểm cao → V của "con mèo" được lấy nhiều hơn
- Kết quả: mô hình biết "nó" chỉ "con mèo"

#### 1.3.3. Multi-Head Attention

Thay vì chạy một attention duy nhất, Multi-Head Attention chạy h attention song song, mỗi head nắm một loại quan hệ khác nhau:

$$\text{MultiHead}(Q,K,V) = \text{Concat}(\text{head}_1, ..., \text{head}_h)W^O$$

$$\text{head}_i = \text{Attention}(QW_i^Q, KW_i^K, VW_i^V)$$

Llama 3.3 70B sử dụng **Grouped Query Attention (GQA)** — biến thể tối ưu của Multi-Head Attention, giảm bộ nhớ trong khi duy trì hiệu năng.

#### 1.3.4. Large Language Model (LLM) và Llama 3.3 70B

**LLM** là mô hình Transformer được huấn luyện trên hàng nghìn tỷ token văn bản với hàng tỷ tham số. Kết quả là mô hình có khả năng:
- Hiểu và sinh ngôn ngữ tự nhiên
- Reasoning (suy luận) phức tạp
- Zero-shot learning: thực hiện nhiệm vụ chưa từng thấy

**Llama 3.3 70B** (Meta AI, 2024):
- 70 tỷ tham số
- Context window: 128.000 token
- Hỗ trợ đa ngôn ngữ bao gồm tiếng Việt
- Hỗ trợ **Function Calling / Tool Use**

#### 1.3.5. Tool Use (Function Calling)

Tool Use là khả năng LLM nhận dạng khi nào cần gọi công cụ bên ngoài (hàm, API) để lấy thông tin, thay vì chỉ dựa vào kiến thức huấn luyện. Cơ chế:

1. Cung cấp cho LLM danh sách tools với mô tả (tên, chức năng, tham số)
2. LLM phân tích câu hỏi → quyết định cần tool nào
3. LLM trả về `finish_reason = "tool_calls"` kèm tên tool và tham số
4. Ứng dụng thực thi tool → trả kết quả về LLM
5. LLM tổng hợp kết quả → sinh câu trả lời cuối

**Lý do chọn Transformer/LLM:**
- Xử lý tốt văn bản dài và ngữ cảnh phức tạp
- Hỗ trợ tiếng Việt (đa ngôn ngữ)
- Tool Use cho phép truy vấn thông tin động
- Zero-shot: không cần dữ liệu training riêng

**Ưu điểm:**
- Hiểu ngữ nghĩa sâu nhờ contextual embedding
- Sinh ngôn ngữ tự nhiên, trôi chảy
- Linh hoạt với nhiều loại câu hỏi

**Nhược điểm:**
- Cần tài nguyên tính toán lớn → sử dụng qua API
- Phụ thuộc kết nối internet
- Giới hạn rate limit ở tài khoản miễn phí

---

## CHƯƠNG 2. PHÂN TÍCH BÀI TOÁN VÀ THIẾT KẾ HỆ THỐNG

### 2.1. Mô tả bài toán

**Đầu vào (Input):**
- Câu hỏi ngôn ngữ tự nhiên của người dùng (tiếng Việt hoặc tiếng Anh)
- Ví dụ: *"Giá Bitcoin hiện tại là bao nhiêu?"*, *"Thời tiết Hà Nội hôm nay?"*

**Đầu ra (Output):**
- Câu trả lời tổng hợp, đầy đủ, có trích dẫn nguồn URL
- Hiển thị streaming (từng token, không chờ toàn bộ câu trả lời)

**Yêu cầu chức năng của hệ thống NLP:**

| # | Yêu cầu | Mô tả |
|---|---------|-------|
| 1 | Hiểu ngôn ngữ tự nhiên | Phân tích câu hỏi, xác định ý định (intent) |
| 2 | Truy vấn web | Tự động fetch và trích xuất nội dung từ URL |
| 3 | Gọi REST API | Gửi HTTP request, xử lý JSON response |
| 4 | Tổng hợp thông tin | Kết hợp dữ liệu từ nhiều nguồn, sinh câu trả lời |
| 5 | Streaming | Truyền phản hồi theo từng token về giao diện |
| 6 | Quản lý hội thoại | Lưu lịch sử theo session, hỗ trợ nhiều cuộc trò chuyện |
| 7 | Đa ngôn ngữ | Hỗ trợ tiếng Việt và tiếng Anh |

### 2.2. Dữ liệu sử dụng

**Nguồn thu thập:**
Đề tài không sử dụng bộ dataset cố định. Thay vào đó, dữ liệu được thu thập **động theo thời gian thực** từ:
- Các trang web công khai trên internet
- Các REST API công khai (tỷ giá, thời tiết, tin tức...)

**Quy mô:** Không giới hạn — hệ thống có thể truy vấn bất kỳ trang web nào có thể truy cập công khai.

**Các bước tiền xử lý dữ liệu** (thực hiện trong `app/services/scraper.py`):

**Bước 1 — Chuẩn hóa HTML (HTML Normalization):**
Loại bỏ các thẻ HTML không chứa nội dung hữu ích:
```
Xóa: <script>, <style>, <nav>, <footer>, <header>, <aside>, <form>
```
Mục đích: loại bỏ JavaScript, CSS, menu điều hướng, quảng cáo — giữ lại nội dung chính.

**Bước 2 — Trích xuất văn bản (Text Extraction):**
```python
text = soup.get_text(separator="\n", strip=True)
```
Chuyển đổi HTML còn lại thành văn bản thuần túy, dùng ký tự xuống dòng làm phân cách giữa các phần.

**Bước 3 — Loại bỏ dòng trống (Blank Line Removal):**
```python
lines = [line for line in text.splitlines() if line.strip()]
```
Lọc các dòng rỗng để giảm nhiễu và tiết kiệm token khi gửi lên LLM.

**Bước 4 — Tokenization:**
Được thực hiện nội bộ bởi LLM Llama 3.3. Mô hình sử dụng **Byte-Pair Encoding (BPE)**: phân tách văn bản thành các subword token, xử lý tốt từ lạ và tiếng Việt.

**Bước 5 — Cắt ngắn (Truncation):**
```
Giới hạn: 5.000 ký tự
Nếu vượt quá → cắt bớt + thêm dấu hiệu "[nội dung bị cắt bớt]"
```
Mục đích: tránh vượt context window của LLM và tiết kiệm token API.

**Bước 6 — Xử lý trang JavaScript (Fallback):**
Nếu nội dung sau bước 2 rỗng (trang render bằng JavaScript), hệ thống tự động chuyển sang dùng **Playwright** — trình duyệt headless — để render đầy đủ trước khi trích xuất.

### 2.3. Kiến trúc hệ thống — Sơ đồ Pipeline NLP

**Kiến trúc tổng thể:**

```
┌─────────────────────────────────────────────────────────────┐
│                     BROWSER (Giao diện Web)                  │
│   [Sidebar sessions] ←→ [Vùng chat + Streaming render]       │
└──────────────────────────────┬──────────────────────────────┘
                               │ HTTP POST /api/chat
                               ▼
┌─────────────────────────────────────────────────────────────┐
│                   FastAPI Backend (Python)                    │
│  routes/chat.py → StreamingResponse → chat_stream()          │
└──────────────────────────────┬──────────────────────────────┘
                               │ messages + system_prompt + tools
                               ▼
┌─────────────────────────────────────────────────────────────┐
│              Groq API — Llama 3.3 70B (LLM)                  │
│         Self-Attention → Phân tích ngữ nghĩa                 │
│         Tool Use Decision: cần tool nào?                     │
└──────────┬──────────────────────────────────────────────────┘
           │ finish_reason = "tool_calls"
           ▼
┌─────────────────────────────────────────────────────────────┐
│                      Tool Executor                           │
│   ┌──────────────────┐     ┌──────────────────────────┐     │
│   │  fetch_webpage   │     │        call_api           │     │
│   │ requests + BS4   │     │  requests JSON response   │     │
│   │ HTML → text      │     │  Gọi REST API công khai   │     │
│   │ (≤ 5000 ký tự)   │     │  (≤ 5000 ký tự)          │     │
│   └──────────────────┘     └──────────────────────────┘     │
└──────────┬──────────────────────────────────────────────────┘
           │ tool_result → append vào messages
           ▼
      [Gửi lại Groq API — Agentic Loop]
           │ finish_reason = "stop"
           ▼
      [Stream từng token về Frontend]
```

**Mô tả luồng xử lý dữ liệu:**

1. **Bước 1 — Nhận đầu vào:** Người dùng nhập câu hỏi → trình duyệt gửi HTTP POST đến `/api/chat` với `{session_id, message}`

2. **Bước 2 — Chuẩn bị ngữ cảnh:** FastAPI ghép `system_prompt` + lịch sử hội thoại + câu hỏi mới thành chuỗi `messages`

3. **Bước 3 — Phân tích NLP:** Gửi lên Groq API với 2 tool definitions. LLM dùng Self-Attention phân tích ý nghĩa câu hỏi, quyết định có cần tool không

4. **Bước 4 — Thực thi Tool (nếu cần):**
   - `finish_reason = "tool_calls"` → backend nhận tên tool + tham số
   - Gọi `fetch_webpage(url)` hoặc `call_api(url, method, params)`
   - Tiền xử lý kết quả (clean HTML, truncate) → thêm vào `messages`
   - Lặp lại bước 3 (Agentic Loop)

5. **Bước 5 — Sinh câu trả lời:** `finish_reason = "stop"` → LLM tổng hợp thông tin, sinh câu trả lời với `stream=True`

6. **Bước 6 — Hiển thị:** `StreamingResponse` gửi từng token về browser → JavaScript render real-time

---

## CHƯƠNG 3. XÂY DỰNG MÔ HÌNH VÀ CÀI ĐẶT

### 3.1. Quy trình huấn luyện mô hình

Đề tài sử dụng phương pháp **Transfer Learning** với mô hình đã được huấn luyện sẵn (pre-trained), thay vì huấn luyện từ đầu.

**Lý do:**
- Huấn luyện LLM từ đầu đòi hỏi hàng nghìn GPU và petabyte dữ liệu — vượt xa phạm vi đề tài
- Llama 3.3 70B đã được Meta AI huấn luyện trên hơn **15 nghìn tỷ token** văn bản đa ngôn ngữ
- Cách tiếp cận hiện đại nhất trong NLP là tận dụng LLM pre-trained và điều chỉnh qua prompt

**Kỹ thuật áp dụng: Prompt Engineering (Zero-shot)**

Zero-shot Prompting là kỹ thuật hướng dẫn LLM thực hiện nhiệm vụ chỉ qua mô tả trong prompt, không cần ví dụ hay dữ liệu training:

```
System Prompt:
"Bạn là trợ lý thông minh hỗ trợ truy vấn thông tin.
Khi người dùng hỏi về thông tin cần tìm từ internet hoặc API:
- Dùng tool fetch_webpage để đọc nội dung từ một URL cụ thể
- Dùng tool call_api để lấy dữ liệu từ các REST API công khai
Hướng dẫn:
- Trả lời bằng ngôn ngữ người dùng sử dụng (Việt/Anh)
- Luôn trích dẫn nguồn (URL) khi đã fetch thông tin từ web
- Tổng hợp thông tin rõ ràng, ngắn gọn và chính xác"
```

**Không cần tập train/validation/test** — đây là ưu điểm của Zero-shot với LLM.

**Thiết kế Agentic Loop:**

```
Bước 1: Gửi {messages, tools} → Groq API
Bước 2: Nếu finish_reason = "tool_calls"
         → Thực thi tool → Thêm kết quả → Quay lại Bước 1
         Nếu finish_reason = "stop"
         → Stream câu trả lời → Kết thúc
```

**Tham số sử dụng:**

| Tham số | Giá trị | Ý nghĩa |
|---------|---------|---------|
| model | llama-3.3-70b-versatile | Model Llama 3.3 70B qua Groq |
| max_tokens | 4096 | Độ dài tối đa câu trả lời |
| tool_choice | "auto" | LLM tự quyết định dùng tool |
| stream | True | Bật streaming response |
| temperature | mặc định | Độ ngẫu nhiên (Groq default) |

### 3.2. Công cụ và thư viện sử dụng

**Ngôn ngữ lập trình:** Python 3.13

**Thư viện backend:**

| Thư viện | Phiên bản | Mục đích |
|----------|-----------|----------|
| `groq` | >=0.11.0 | Gọi Groq API, sử dụng Llama 3.3 70B |
| `fastapi` | >=0.115.0 | Web framework, xây dựng REST API |
| `uvicorn` | >=0.30.0 | ASGI server chạy FastAPI |
| `pydantic` | >=2.0 | Validation và serialization dữ liệu |
| `pydantic-settings` | >=2.0 | Quản lý cấu hình từ `.env` |
| `requests` | >=2.31.0 | HTTP client — fetch webpage, gọi API |
| `beautifulsoup4` | >=4.12.0 | Parse HTML, trích xuất văn bản |
| `python-dotenv` | >=1.0.0 | Load biến môi trường từ file `.env` |
| `httpx` | >=0.27.0 | HTTP client async (dự phòng) |

**Thư viện frontend:** HTML5, CSS3, JavaScript (Vanilla — không dùng framework)

**Cấu trúc dự án:**
```
chatbot/
├── app/
│   ├── main.py              # FastAPI entry point, mount static files
│   ├── config.py            # Cấu hình: API key, model, max_tokens
│   ├── models.py            # Pydantic schemas: ChatRequest, Message
│   ├── routes/
│   │   └── chat.py          # Endpoints: POST /api/chat, GET/DELETE /api/history
│   ├── tools/
│   │   ├── definitions.py   # Tool schemas cung cấp cho LLM
│   │   ├── web_tools.py     # Wrapper gọi scraper
│   │   └── api_tools.py     # Gọi HTTP API, xử lý JSON
│   └── services/
│       ├── claude_client.py # Agentic loop + streaming với Groq
│       └── scraper.py       # Thu thập web: requests + BeautifulSoup
├── static/
│   ├── index.html           # Giao diện chat
│   ├── style.css            # Styling
│   └── app.js               # Logic frontend, streaming render
├── .env                     # GROQ_API_KEY
└── requirements.txt
```

### 3.3. Demo sản phẩm

**Giao diện người dùng:**

Hệ thống cung cấp giao diện web chat với bố cục 2 cột:
- **Cột trái (Sidebar):** Danh sách các cuộc hội thoại, nút tạo mới
- **Cột phải (Chat area):** Vùng hiển thị tin nhắn và ô nhập liệu

**Các tính năng NLP thể hiện:**

1. **Hiểu ngôn ngữ tự nhiên:** Người dùng gõ câu hỏi bình thường, không cần cú pháp đặc biệt

2. **Tự động quyết định dùng tool:** Khi cần tra cứu thực, hệ thống hiển thị:
   ```
   [Đang dùng tool: fetch_webpage...]
   ```

3. **Trả lời streaming:** Chữ xuất hiện từng từ như đang được gõ — cải thiện trải nghiệm người dùng

4. **Đa ngôn ngữ:** Hỏi tiếng Việt → trả lời tiếng Việt; hỏi tiếng Anh → trả lời tiếng Anh

5. **Quản lý hội thoại:** Lưu lịch sử, tạo nhiều cuộc trò chuyện riêng biệt

**Cách tương tác:**
1. Mở trình duyệt tại `http://localhost:8000`
2. Gõ câu hỏi vào ô nhập, nhấn Enter hoặc nút "Gửi"
3. Xem chatbot tự động tìm kiếm và trả lời

**Ví dụ câu hỏi demo:**
- *"Tỷ giá USD/VND hiện tại là bao nhiêu?"*
- *"Tin tức về AI mới nhất hôm nay?"*
- *"Lấy dữ liệu từ https://api.exchangerate-api.com/v4/latest/USD"*

*(Thêm ảnh chụp màn hình giao diện vào đây)*

### 3.4. Đánh giá kết quả

**Kết quả đạt được:**

| Tiêu chí | Kết quả |
|----------|---------|
| Hiểu câu hỏi tiếng Việt | Đạt — mô hình hiểu tốt |
| Tự động fetch web | Đạt — lấy được nội dung từ hầu hết trang tĩnh |
| Gọi REST API | Đạt — xử lý JSON chính xác |
| Trích dẫn nguồn URL | Đạt — luôn kèm URL nguồn |
| Streaming phản hồi | Đạt — trải nghiệm tốt |
| Tốc độ phản hồi | Tốt — Groq inference nhanh (khoảng 2-5 giây) |

**Hạn chế:**
- Groq free tier giới hạn **30 requests/phút** và 6.000 token/phút
- Không fetch được trang web yêu cầu đăng nhập hoặc có captcha
- Thông tin phụ thuộc chất lượng trang web được fetch
- Session lưu in-memory — mất khi restart server
- Chưa xác thực người dùng (authentication)

---

## KẾT LUẬN

### Kết quả đạt được

Đề tài đã xây dựng thành công hệ thống chatbot ứng dụng NLP có khả năng:
- Hiểu và xử lý câu hỏi ngôn ngữ tự nhiên tiếng Việt và tiếng Anh
- Tự động truy vấn thông tin từ web/API thời gian thực thông qua cơ chế Tool Use
- Tổng hợp thông tin và trả lời có trích dẫn nguồn
- Cung cấp giao diện web với trải nghiệm streaming mượt mà

### Ý nghĩa của sản phẩm

Đề tài minh họa hướng tiếp cận NLP hiện đại nhất: thay vì xây dựng và huấn luyện mô hình từ đầu, tận dụng LLM đã có thông qua Transfer Learning và Prompt Engineering. Đây là phương pháp được áp dụng rộng rãi trong các sản phẩm thực tế, từ trợ lý ảo doanh nghiệp đến hệ thống hỏi đáp tự động.

Cơ chế **Tool Use / Agentic Loop** được áp dụng là nền tảng của các hệ thống AI agent hiện đại — cho phép LLM không chỉ sinh văn bản mà còn tương tác với thế giới bên ngoài (web, API, database...).

### Hạn chế của hệ thống NLP

1. **Không học tập liên tục:** Mô hình không tự cập nhật từ các câu hỏi mới
2. **Phụ thuộc API:** Phụ thuộc Groq API — nếu API ngừng hoạt động, hệ thống dừng
3. **Hạn chế ngữ cảnh:** Context window 128K token — hội thoại rất dài có thể vượt giới hạn
4. **Ảo giác (Hallucination):** LLM đôi khi sinh thông tin không chính xác khi thiếu dữ liệu

### Hướng cải tiến trong tương lai

1. **Thêm RAG (Retrieval-Augmented Generation):** Xây dựng vector database lưu tài liệu nội bộ (PDF, Word...) — chatbot tìm kiếm trong cơ sở tri thức riêng trước khi fetch web

2. **Fine-tuning với dữ liệu tiếng Việt:** Huấn luyện thêm mô hình trên corpus tiếng Việt chuyên ngành để cải thiện độ chính xác

3. **Mở rộng tools:** Thêm tìm kiếm Google Search API, đọc file PDF, truy vấn cơ sở dữ liệu SQL

4. **Lưu trữ bền vững:** Thay in-memory sessions bằng database (PostgreSQL/SQLite)

5. **Đánh giá định lượng:** Xây dựng bộ test câu hỏi + đáp án chuẩn để đo BLEU score, accuracy

---

## TÀI LIỆU THAM KHẢO

[1] Vaswani, A., Shazeer, N., Parmar, N., et al. (2017). *Attention Is All You Need*. Advances in Neural Information Processing Systems (NeurIPS), 30.

[2] Mikolov, T., Chen, K., Corrado, G., & Dean, J. (2013). *Efficient Estimation of Word Representations in Vector Space*. arXiv preprint arXiv:1301.3781.

[3] Dubey, A., Jauhri, A., Pandey, A., et al. (2024). *The Llama 3 Herd of Models*. arXiv preprint arXiv:2407.21783.

[4] Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). *BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding*. NAACL-HLT 2019.

[5] Brown, T., Mann, B., Ryder, N., et al. (2020). *Language Models are Few-Shot Learners (GPT-3)*. NeurIPS 2020.

[6] Groq. (2024). *Groq API Documentation*. https://console.groq.com/docs

[7] FastAPI. (2024). *FastAPI Documentation*. https://fastapi.tiangolo.com

[8] Richardson, L. (2007). *Beautiful Soup Documentation*. https://www.crummy.com/software/BeautifulSoup/bs4/doc/
