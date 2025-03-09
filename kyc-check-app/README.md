# 🏗️ kyc-app-bos Frontend

This folder contains the **frontend** of the KYC application built using **Next.js**. It serves as the user interface for onboarding clients, uploading documents, document verification, and AI-powered identity verification.

---

## 🚀 Getting Started

### 1️⃣ Install Dependencies

Make sure you have **Node.js** installed, then run:

```sh
npm install
```

### 2️⃣ Run the Development Server

Start the Next.js frontend locally:

```sh
npm run dev
```

By default, the app will be available at `http://localhost:3000`.

---

## 📌 Features

✅ **User-friendly KYC form** – Step-by-step document submission  
✅ **AI-powered verification** – Connects with Flask backend for fraud detection  
✅ **Real-time feedback** – Shows progress and validation results

---

## 🔗 Backend Integration

This frontend communicates with the **Flask backend**, which handles:

- Document processing & fraud detection
- AI-powered risk assessment
- Identity verification & approval

Ensure the backend is running before testing full functionality (running on port 8000).
