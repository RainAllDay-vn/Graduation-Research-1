# Database Schema: AI Cache

The `ai_cache.db` database stores historical AI model responses, structured around specific system instructions, user templates, and datasets. This cache facilitates efficient retrieval of generated responses to minimize redundant API calls and processing time.

## Overview of Tables

### 1. `system_prompts`
Definitions of instructions provided to the AI models to guide their behavior and output format.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | INTEGER | PRIMARY KEY | Unique identifier for the system prompt. |
| `content` | TEXT | | The actual instruction text (e.g., role definition, formatting constraints). |

**Sample Record:**
> **ID 1**: "You are an expert in Cypher query language. Translate the natural language question into a Cypher qu..."

---

### 2. `user_prompt_templates`
Templates used to structure the user input before it is sent to the model.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `id` | INTEGER | PRIMARY KEY | Unique identifier for the template. |
| `content` | TEXT | | Template string containing placeholders (e.g., `{{question}}`). |

**Sample Record:**
> **ID 1**: "Translate this question into a Cypher query:\nQuestion: {{question}}"

---

### 3. `ai_cache`
The central repository for cached AI responses. It uses a composite primary key to ensure unique entries for every unique combination of model, dataset, instruction set, and input question.

| Column | Type | Constraints | Description |
| :--- | :--- | :--- | :--- |
| `model_name` | TEXT | PK (1) | The identifier of the AI model. |
| `dataset_name` | TEXT | PK (2) | The name of the dataset the question belongs to. |
| `system_prompt_id` | INTEGER | PK (3) | Reference to `system_prompts.id`. |
| `template_id` | INTEGER | PK (4) | Reference to `user_prompt_templates.id`. |
| `question` | TEXT | PK (5) | The specific input question. |
| `include_reasoning` | INTEGER | PK (6) | Boolean flag (0/1) indicating if reasoning was requested. |
| `response` | TEXT | | The model's output (often in JSON format). |
| `logprobs` | TEXT | | Log probability information, if available. |

**Database Stats:**
- **Record Count:** ~32,986 entries.
- **Composite Primary Key:** Ensures uniqueness across all variables (Model, Dataset, Prompts, Question, Reasoning toggle).

---

### 4. `sqlite_sequence`
Internal SQLite system table used to manage autoincrementing primary keys for `system_prompts` and other related tables.
