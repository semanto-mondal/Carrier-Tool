# 🚀 Carrier-Tool: Multi-Agent CV Analysis Assistant

**Carrier-Tool** is an intelligent CV analysis assistant built using **CrewAI** and **RAG (Retrieval-Augmented Generation)** techniques. Designed with a multi-agent architecture, it automates your resume review, skill extraction, improvement suggestions, and job recommendations — all in one seamless flow.

This is not the best model out there, but this can be used as a starting point or idea which can be improvised in a better way.

---

## 🧠 How It Works

The system runs a sequence of specialised agents, each responsible for a unique task:

1. **CV Analysis Agent** – Reviews and interprets your uploaded CV.
2. **Skill Extraction Agent** – Identifies key and missing skills relevant to your profile.
3. **Improvement Agent** – Offers targeted suggestions to enhance your CV.
4. **Job Search Agent** – Recommends the latest jobs based on your skills and background.

---

## 🔧 Tech Stack Highlights

Carrier-Tool comes with two flexible setup options:

- ⚙️ **Cloud Mode**: Uses **CrewAI** with the blazing-fast **Groq API** for powerful LLM inference.
- 💻 **Local Mode**: Powered by **Ollama** and CrewAI for a fully offline experience — perfect for privacy-conscious users.

---

## 🧰 Tools & Integrations

- 🗂 **PDF Reader Tool** – Parses your uploaded CV in PDF format.
- 🌐 **Web Search Tool** – Gathers real-time context when needed.
- 💼 **Apify Actors Tool** – Leverages Apify API to fetch the latest job listings tailored to your profile.

---

## 📸 Few of the outputs 
<img width="624" alt="screen 3" src="https://github.com/user-attachments/assets/615b735c-24bc-49ae-af1f-471c6444249a" />
<img width="650" alt="S1" src="https://github.com/user-attachments/assets/3bc406f0-c2be-47f3-80ab-4f63da2d535b" />
<img width="661" alt="Screen 2" src="https://github.com/user-attachments/assets/320fcf99-2ae1-4b88-9f91-3112c3ea9511" />

These images shows a few output based on the CV provided to it.

---

## Limitations 

1. The first limitation is using an API that has limited token access
2. Another one is that while using locally installed ollama, the inference is very slow because multiple agents work together.

---


