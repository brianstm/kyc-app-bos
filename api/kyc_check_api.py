import os
import base64
import json
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from PIL import Image
import cv2
import pytesseract
from pdf2image import convert_from_path
import fitz
import re
from openai import OpenAI
from dotenv import load_dotenv
from flask import Flask, request, jsonify, Response
from flask_cors import CORS
import tempfile
import threading
import uuid
from threading import Thread

load_dotenv()

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "http://localhost:3000"}})

verification_jobs = {}

class DocumentVerifier:
    def __init__(self, api_key):
        self.client = OpenAI(api_key=api_key)
        self.verification_results = {}
        self.document_texts = {}
        self.document_images = {}
        self.overall_assessment = {}
        self.status_updates = []
        
    def add_status_update(self, status, details=None):
        """Add a status update with timestamp"""
        update = {
            'timestamp': datetime.now().isoformat(),
            'status': status,
            'details': details
        }
        self.status_updates.append(update)
        return update
        
    def load_document(self, file_path):
        """Load document (PDF or image) and extract text and images"""
        self.add_status_update("Loading document", {"file_path": os.path.basename(file_path)})
        
        file_name = os.path.basename(file_path)
        file_ext = os.path.splitext(file_path)[1].lower()
        
        if file_ext in ['.pdf']:
            return self._process_pdf(file_path, file_name)
        elif file_ext in ['.jpg', '.jpeg', '.png', '.tiff', '.bmp']:
            return self._process_image(file_path, file_name)
        else:
            self.add_status_update("Error", {"message": f"Unsupported file format: {file_ext}"})
            raise ValueError(f"Unsupported file format: {file_ext}")
    
    def _process_pdf(self, file_path, file_name):
        """Extract text and images from PDF"""
        self.add_status_update("Processing PDF", {"file_name": file_name})
        document_id = f"doc_{len(self.document_texts) + 1}"
        
        doc = fitz.open(file_path)
        full_text = ""
        
        images = []
        for page_num, page in enumerate(doc):
            self.add_status_update("Processing PDF page", {"page": page_num + 1, "total_pages": len(doc)})
            full_text += page.get_text()
            
            image_list = page.get_images(full=True)
            for img_index, img in enumerate(image_list):
                xref = img[0]
                base_image = doc.extract_image(xref)
                image_bytes = base_image["image"]
                
                nparr = np.frombuffer(image_bytes, np.uint8)
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                images.append(image)
                
        pages = convert_from_path(file_path)
        for i, page in enumerate(pages):
            page_image = np.array(page)
            page_image = cv2.cvtColor(page_image, cv2.COLOR_RGB2BGR)
            images.append(page_image)
        
        self.document_texts[document_id] = {
            'file_name': file_name,
            'text': full_text,
            'file_type': 'pdf'
        }
        
        self.document_images[document_id] = images
        
        self.add_status_update("PDF processed", {"document_id": document_id, "file_name": file_name})
        return document_id
    
    def _process_image(self, file_path, file_name):
        """Process image document"""
        self.add_status_update("Processing image", {"file_name": file_name})
        document_id = f"doc_{len(self.document_texts) + 1}"
        
        image = cv2.imread(file_path)
        if image is None:
            self.add_status_update("Error", {"message": f"Failed to load image: {file_path}"})
            raise ValueError(f"Failed to load image: {file_path}")
        
        text = pytesseract.image_to_string(image)
        
        self.document_texts[document_id] = {
            'file_name': file_name,
            'text': text,
            'file_type': 'image'
        }
        
        self.document_images[document_id] = [image]
        
        self.add_status_update("Image processed", {"document_id": document_id, "file_name": file_name})
        return document_id
    
    def analyze_document_type(self, document_id):
        """Determine document type using GPT"""
        self.add_status_update("Analyzing document type", {"document_id": document_id})
        doc_info = self.document_texts.get(document_id)
        if not doc_info:
            self.add_status_update("Error", {"message": f"Document ID not found: {document_id}"})
            raise ValueError(f"Document ID not found: {document_id}")
        
        text = doc_info['text']
        
        text_sample = text[:2000]
        
        prompt = f"""
        Analyze the following document text and identify what type of document it is. 
        Consider options like:
        - Passport
        - ID Card
        - Driver's License
        - Bank Statement
        - Tax Return
        - Business Registration
        - Utility Bill
        - Financial Statement
        - Certificate of Incorporation
        - Proof of Address
        - Income Verification
        - Source of Funds Declaration
        
        If you cannot determine the document type, indicate "Unknown".
        Provide your assessment and explain your reasoning.
        
        Document text sample:
        {text_sample}
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a document analysis specialist with expertise in identifying document types from text content."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        analysis = response.choices[0].message.content
        
        doc_type_match = re.search(r"document type(?:.*?):(?:.*?)(\w[\w\s-]+(?:card|passport|license|statement|return|registration|bill|certificate|declaration|verification|form))", analysis, re.IGNORECASE)
        doc_type = doc_type_match.group(1).strip() if doc_type_match else "Unknown"
        
        result = {
            'document_type': doc_type,
            'analysis': analysis
        }
        
        if 'document_analysis' not in self.verification_results:
            self.verification_results['document_analysis'] = {}
        
        self.verification_results['document_analysis'][document_id] = result
        self.add_status_update("Document type analyzed", {"document_id": document_id, "document_type": doc_type})
        return result
    
    def check_document_consistency(self, document_id):
        """Check internal consistency of the document"""
        self.add_status_update("Checking document consistency", {"document_id": document_id})
        doc_info = self.document_texts.get(document_id)
        if not doc_info:
            self.add_status_update("Error", {"message": f"Document ID not found: {document_id}"})
            raise ValueError(f"Document ID not found: {document_id}")
        
        text = doc_info['text']
        doc_type = self.verification_results.get('document_analysis', {}).get(document_id, {}).get('document_type', 'Unknown')
        
        prompt = f"""
        Analyze the following document text for internal consistency. 
        This appears to be a {doc_type}.
        
        Check for:
        1. Dates - Are all dates consistent? Are there any future dates or impossible dates?
        2. Names - Are names consistent throughout the document?
        3. Formatting - Does the document formatting match what would be expected for this type?
        4. Content - Is the content logically consistent?
        5. Inconsistencies or red flags - Are there any contradictions or suspicious elements?
        
        Document text:
        {text[:4000]}  # Using first 4000 chars to stay within token limits
        
        Provide a detailed assessment of consistency issues, give a confidence score (0-100%) for document consistency, and explain your reasoning.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a forensic document examiner with expertise in detecting inconsistencies and authenticity issues in documents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        analysis = response.choices[0].message.content
        
        confidence_match = re.search(r"confidence score:?\s*(\d+)%", analysis, re.IGNORECASE)
        confidence = int(confidence_match.group(1)) if confidence_match else 50
        
        result = {
            'consistency_score': confidence,
            'analysis': analysis
        }
        
        if 'consistency_check' not in self.verification_results:
            self.verification_results['consistency_check'] = {}
        
        self.verification_results['consistency_check'][document_id] = result
        self.add_status_update("Document consistency checked", {"document_id": document_id, "consistency_score": confidence})
        return result
    
    def extract_personal_information(self, document_id):
        """Extract key personal information from document"""
        self.add_status_update("Extracting personal information", {"document_id": document_id})
        doc_info = self.document_texts.get(document_id)
        if not doc_info:
            self.add_status_update("Error", {"message": f"Document ID not found: {document_id}"})
            raise ValueError(f"Document ID not found: {document_id}")
        
        text = doc_info['text']
        doc_type = self.verification_results.get('document_analysis', {}).get(document_id, {}).get('document_type', 'Unknown')
        
        prompt = f"""
        Extract key personal information from this {doc_type}.
        
        Depending on document type, look for:
        - Full name
        - Date of birth
        - Document number (passport number, ID number, etc.)
        - Issue date / Expiry date
        - Address
        - Financial figures (account balances, income amounts)
        - Business names
        - Transaction details
        - Tax identification numbers
        
        Present the information in a structured format with clear labels.
        If you cannot find certain information, indicate "Not found" for that field.
        
        Document text:
        {text[:4000]}
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a data extraction specialist focusing on accurately extracting personal and financial information from documents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        analysis = response.choices[0].message.content
        
        result = {
            'extracted_info': analysis
        }
        
        if 'personal_information' not in self.verification_results:
            self.verification_results['personal_information'] = {}
        
        self.verification_results['personal_information'][document_id] = result
        self.add_status_update("Personal information extracted", {"document_id": document_id})
        return result
    
    def check_identity_consistency(self, document_ids):
        """Check consistency of identity information across multiple documents"""
        self.add_status_update("Checking identity consistency across documents", {"document_ids": document_ids})
        if len(document_ids) < 2:
            self.add_status_update("Error", {"message": "Need at least 2 documents to compare identity consistency"})
            return {"error": "Need at least 2 documents to compare identity consistency"}
        
        personal_info_by_doc = {}
        for doc_id in document_ids:
            if doc_id in self.verification_results.get('personal_information', {}):
                personal_info_by_doc[doc_id] = self.verification_results['personal_information'][doc_id]['extracted_info']
            else:
                info = self.extract_personal_information(doc_id)
                personal_info_by_doc[doc_id] = info['extracted_info']
        
        docs_info = []
        for doc_id, info in personal_info_by_doc.items():
            doc_type = self.verification_results.get('document_analysis', {}).get(doc_id, {}).get('document_type', 'Unknown')
            file_name = self.document_texts[doc_id]['file_name']
            docs_info.append(f"Document {doc_id} ({file_name} - {doc_type}):\n{info}")
        
        comparison_text = "\n\n".join(docs_info)
        
        prompt = f"""
        Compare the personal information extracted from the following documents for consistency:
        
        {comparison_text}
        
        Analyze:
        1. Name consistency across documents
        2. Date of birth consistency
        3. Address consistency
        4. ID numbers consistency (where applicable)
        5. Any other relevant personal details
        
        Identify any inconsistencies or discrepancies between documents.
        Calculate a confidence score (0-100%) for identity consistency across documents.
        Provide detailed reasoning for your assessment.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a forensic identity verification specialist with expertise in detecting identity fraud and inconsistencies across multiple documents."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        analysis = response.choices[0].message.content
        
        confidence_match = re.search(r"confidence score:?\s*(\d+)%", analysis, re.IGNORECASE)
        confidence = int(confidence_match.group(1)) if confidence_match else 50
        
        result = {
            'identity_consistency_score': confidence,
            'analysis': analysis,
            'documents_compared': document_ids
        }
        
        self.verification_results['identity_consistency'] = result
        self.add_status_update("Identity consistency checked", {"confidence_score": confidence})
        return result
    
    def analyze_financial_legitimacy(self, document_ids):
        """Analyze financial documents for source of wealth and source of funds legitimacy"""
        self.add_status_update("Analyzing financial legitimacy", {"document_ids": document_ids})
        financial_texts = []
        
        for doc_id in document_ids:
            if doc_id not in self.document_texts:
                continue
                
            doc_info = self.document_texts[doc_id]
            doc_type = self.verification_results.get('document_analysis', {}).get(doc_id, {}).get('document_type', 'Unknown')
            
            if any(term in doc_type.lower() for term in ['bank', 'statement', 'tax', 'income', 'financial', 'funds', 'wealth']):
                financial_texts.append(f"Document {doc_id} ({doc_info['file_name']} - {doc_type}):\n{doc_info['text'][:2000]}")
        
        if not financial_texts:
            self.add_status_update("Warning", {"message": "No financial documents found for analysis"})
            return {"error": "No financial documents found for analysis"}
        
        financial_content = "\n\n".join(financial_texts)
        
        prompt = f"""
        Analyze the following financial documents for Source of Wealth (SoW) and Source of Funds (SoF) legitimacy.
        
        {financial_content}
        
        Evaluate:
        1. Income sources identified - Are they clear and legitimate?
        2. Transaction patterns - Are there any suspicious patterns?
        3. Asset holdings - Are they consistent with declared income?
        4. Unexplained wealth indicators - Are there unexplained large sums?
        5. Red flags for money laundering or financial fraud
        
        Calculate a legitimacy confidence score (0-100%) for the financial information.
        Provide detailed reasoning for your assessment.
        Highlight any specific areas of concern.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a financial crime specialist with expertise in anti-money laundering, source of wealth verification, and financial fraud detection."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        analysis = response.choices[0].message.content
        
        confidence_match = re.search(r"confidence score:?\s*(\d+)%", analysis, re.IGNORECASE)
        confidence = int(confidence_match.group(1)) if confidence_match else 50
        
        result = {
            'financial_legitimacy_score': confidence,
            'analysis': analysis,
            'documents_analyzed': document_ids
        }
        
        self.verification_results['financial_legitimacy'] = result
        self.add_status_update("Financial legitimacy analyzed", {"confidence_score": confidence})
        return result
    
    def detect_visual_manipulation(self, document_id):
        """Detect signs of image manipulation or forgery"""
        self.add_status_update("Detecting visual manipulation", {"document_id": document_id})
        images = self.document_images.get(document_id, [])
        if not images:
            self.add_status_update("Error", {"message": "No images found for this document"})
            return {"error": "No images found for this document"}
        
        doc_info = self.document_texts[document_id]
        doc_type = self.verification_results.get('document_analysis', {}).get(document_id, {}).get('document_type', 'Unknown')
        
        image_descriptions = []
        for i, img in enumerate(images):
            height, width, channels = img.shape
            
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            edges = cv2.Canny(gray, 100, 200)
            edge_count = np.count_nonzero(edges)
            edge_density = edge_count / (height * width)
            
            noise_std = np.std(gray)
            
            description = f"Image {i+1}: {width}x{height}, Edge density: {edge_density:.4f}, Noise std: {noise_std:.2f}"
            image_descriptions.append(description)
        
        visual_info = "\n".join(image_descriptions)
        
        prompt = f"""
        Analyze the potential for visual manipulation in this {doc_type} document.
        
        Document details:
        - Filename: {doc_info['file_name']}
        - Document type: {doc_type}
        
        Visual characteristics:
        {visual_info}
        
        Based on these characteristics and your knowledge of document forgery techniques:
        1. Assess the likelihood of visual manipulation in this document
        2. Identify potential indicators of manipulation (high edge density in unusual areas, inconsistent noise patterns)
        3. Evaluate typical manipulation techniques for this document type
        
        Calculate a visual integrity confidence score (0-100%) and explain your reasoning.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a forensic document examiner specializing in detecting visual manipulation, forgery, and document tampering."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        analysis = response.choices[0].message.content
        
        confidence_match = re.search(r"confidence score:?\s*(\d+)%", analysis, re.IGNORECASE)
        confidence = int(confidence_match.group(1)) if confidence_match else 50
        
        result = {
            'visual_integrity_score': confidence,
            'analysis': analysis
        }
        
        if 'visual_integrity' not in self.verification_results:
            self.verification_results['visual_integrity'] = {}
        
        self.verification_results['visual_integrity'][document_id] = result
        self.add_status_update("Visual manipulation detection completed", {"document_id": document_id, "confidence_score": confidence})
        return result
    
    def verify_document_authenticity(self, document_id):
        """Perform comprehensive authenticity check on a single document"""
        self.add_status_update("Verifying document authenticity", {"document_id": document_id})
        if document_id not in self.document_texts:
            self.add_status_update("Error", {"message": f"Document ID not found: {document_id}"})
            raise ValueError(f"Document ID not found: {document_id}")
            
        if 'document_analysis' not in self.verification_results or document_id not in self.verification_results['document_analysis']:
            self.analyze_document_type(document_id)
            
        if 'consistency_check' not in self.verification_results or document_id not in self.verification_results['consistency_check']:
            self.check_document_consistency(document_id)
            
        if 'personal_information' not in self.verification_results or document_id not in self.verification_results['personal_information']:
            self.extract_personal_information(document_id)
            
        if 'visual_integrity' not in self.verification_results or document_id not in self.verification_results['visual_integrity']:
            self.detect_visual_manipulation(document_id)
        
        doc_type = self.verification_results['document_analysis'][document_id]['document_type']
        consistency = self.verification_results['consistency_check'][document_id]
        personal_info = self.verification_results['personal_information'][document_id]
        visual = self.verification_results['visual_integrity'][document_id]
        
        consistency_score = consistency['consistency_score']
        visual_score = visual['visual_integrity_score']
        
        overall_score = int(0.5 * consistency_score + 0.5 * visual_score)
        
        risk_level = "Low" if overall_score >= 80 else "Medium" if overall_score >= 60 else "High"
        
        result = {
            'document_id': document_id,
            'file_name': self.document_texts[document_id]['file_name'],
            'document_type': doc_type,
            'authenticity_score': overall_score,
            'risk_level': risk_level,
            'consistency_score': consistency_score,
            'visual_integrity_score': visual_score,
            'consistency_analysis': consistency['analysis'],
            'visual_analysis': visual['analysis'],
            'extracted_personal_info': personal_info['extracted_info']
        }
        
        if 'document_authenticity' not in self.verification_results:
            self.verification_results['document_authenticity'] = {}
            
        self.verification_results['document_authenticity'][document_id] = result
        self.add_status_update("Document authenticity verified", {"document_id": document_id, "authenticity_score": overall_score, "risk_level": risk_level})
        return result
    
    def generate_overall_assessment(self, document_ids):
        """Generate comprehensive assessment across all documents"""
        self.add_status_update("Generating overall assessment", {"document_ids": document_ids})
        for doc_id in document_ids:
            if 'document_authenticity' not in self.verification_results or doc_id not in self.verification_results['document_authenticity']:
                self.verify_document_authenticity(doc_id)
        
        if len(document_ids) >= 2:
            if 'identity_consistency' not in self.verification_results:
                self.check_identity_consistency(document_ids)
                
            if 'financial_legitimacy' not in self.verification_results:
                self.analyze_financial_legitimacy(document_ids)
        
        doc_scores = [self.verification_results['document_authenticity'][doc_id]['authenticity_score'] for doc_id in document_ids]
        avg_doc_score = sum(doc_scores) / len(doc_scores)
        
        identity_score = self.verification_results.get('identity_consistency', {}).get('identity_consistency_score', 0)
        financial_score = self.verification_results.get('financial_legitimacy', {}).get('financial_legitimacy_score', 0)
        
        if len(document_ids) >= 2:
            weights = [0.4, 0.35, 0.25]  
            overall_score = int(weights[0] * avg_doc_score + 
                               weights[1] * identity_score +
                               weights[2] * financial_score)
        else:
            overall_score = int(avg_doc_score)
        
        risk_level = "Low" if overall_score >= 80 else "Medium" if overall_score >= 60 else "High"
        verification_status = "Verified" if overall_score >= 75 else "Requires Further Verification"
        
        documents_summary = []
        for doc_id in document_ids:
            doc_result = self.verification_results['document_authenticity'][doc_id]
            doc_summary = (f"Document: {doc_result['file_name']} ({doc_result['document_type']})\n"
                          f"Authenticity Score: {doc_result['authenticity_score']}%\n"
                          f"Risk Level: {doc_result['risk_level']}")
            documents_summary.append(doc_summary)
        
        docs_text = "\n\n".join(documents_summary)
        
        findings = []
        findings.append(f"Document Authenticity: Average score of {avg_doc_score:.1f}%")
        
        if 'identity_consistency' in self.verification_results:
            findings.append(f"Identity Consistency: {identity_score}%")
            
        if 'financial_legitimacy' in self.verification_results:
            findings.append(f"Financial Legitimacy: {financial_score}%")
            
        findings_text = "\n".join(findings)
        
        context = f"""
        DOCUMENT VERIFICATION SUMMARY:
        
        {docs_text}
        
        KEY FINDINGS:
        {findings_text}
        
        OVERALL ASSESSMENT:
        Verification Score: {overall_score}%
        Risk Level: {risk_level}
        Status: {verification_status}
        """
        
        prompt = f"""
        Based on the comprehensive document verification results below, provide a detailed assessment of whether the person appears to be genuine or potentially fraudulent.
        
        {context}
        
        Provide:
        1. An executive summary of verification findings (2-3 sentences)
        2. Key risk factors and suspicious indicators (if any)
        3. Key positive verification elements
        4. Detailed reasoning for the final verification decision
        5. Recommendations for further verification steps (if needed)
        
        Use a formal, objective tone suitable for compliance documentation.
        """
        
        response = self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a senior compliance officer specializing in KYC (Know Your Customer) verification and financial fraud prevention. Provide detailed, evidence-based assessments."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        final_assessment = response.choices[0].message.content
        
        result = {
            'verification_score': overall_score,
            'risk_level': risk_level,
            'verification_status': verification_status,
            'document_count': len(document_ids),
            'documents_analyzed': document_ids,
            'average_document_score': avg_doc_score,
            'identity_consistency_score': identity_score if len(document_ids) >= 2 else None,
            'financial_legitimacy_score': financial_score if 'financial_legitimacy' in self.verification_results else None,
            'detailed_assessment': final_assessment
        }
        
        self.overall_assessment = result
        self.add_status_update("Overall assessment complete", {"verification_score": overall_score, "risk_level": risk_level, "status": verification_status})
        return result
    
    def generate_verification_report(self, document_ids):
        """Generate a comprehensive verification report"""
        self.add_status_update("Generating final verification report", {"document_ids": document_ids})
        if not self.overall_assessment:
            self.generate_overall_assessment(document_ids)
            
        document_details = []
        for doc_id in document_ids:
            auth_result = self.verification_results['document_authenticity'][doc_id]
            doc_detail = {
                'document_id': doc_id,
                'file_name': auth_result['file_name'],
                'document_type': auth_result['document_type'],
                'authenticity_score': auth_result['authenticity_score'],
                'risk_level': auth_result['risk_level'],
                'extracted_info': auth_result['extracted_personal_info']
            }
            document_details.append(doc_detail)
            
        report = {
            'report_date': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'overall_assessment': self.overall_assessment,
            'document_details': document_details,
            'cross_document_checks': {
                'identity_consistency': self.verification_results.get('identity_consistency'),
                'financial_legitimacy': self.verification_results.get('financial_legitimacy')
            },
            'status_updates': self.status_updates,
            'raw_verification_results': self.verification_results
        }
        
        self.add_status_update("Verification report complete")
        return report

def run_verification_job(job_id, api_key, temp_files):
    """Run the verification process as a background job"""
    try:
        verifier = DocumentVerifier(api_key)
        document_ids = []
        
        for temp_file in temp_files:
            doc_id = verifier.load_document(temp_file)
            document_ids.append(doc_id)
            
            # Update job status with each document loaded
            verification_jobs[job_id]['status_updates'] = verifier.status_updates
            verification_jobs[job_id]['progress'] = {
                'current_step': 'Document Loading',
                'documents_loaded': len(document_ids),
                'total_documents': len(temp_files)
            }
        
        verifier.generate_overall_assessment(document_ids)
        report = verifier.generate_verification_report(document_ids)
        
        verification_jobs[job_id]['status'] = 'complete'
        verification_jobs[job_id]['report'] = report
        verification_jobs[job_id]['status_updates'] = verifier.status_updates
        
    except Exception as e:
        verification_jobs[job_id]['status'] = 'failed'
        verification_jobs[job_id]['error'] = str(e)
        verification_jobs[job_id]['status_updates'].append({
            'timestamp': datetime.now().isoformat(),
            'status': 'Error',
            'details': {'message': str(e)}
        })

@app.route('/api/verify', methods=['POST'])
def verify_documents():
    """Endpoint to initiate document verification"""
    try:
        if 'files' not in request.files:
            return jsonify({'error': 'No files provided'}), 400
        
        files = request.files.getlist('files')
        if not files:
            return jsonify({'error': 'No files provided'}), 400
        
        job_id = str(uuid.uuid4())
        
        temp_files = []
        for file in files:
            if file.filename == '':
                continue
            temp_path = os.path.join('/tmp', file.filename)
            file.save(temp_path)
            temp_files.append(temp_path)
        
        verification_jobs[job_id] = {
            'status': 'pending',
            'start_time': datetime.now().isoformat(),
            'status_updates': [],
            'progress': {},
            'temp_files': temp_files
        }
        
        api_key = request.headers.get('X-API-Key') or os.getenv('OPENAI_API_KEY')
        if not api_key:
            return jsonify({'error': 'API key required'}), 401
        
        Thread(target=run_verification_job, args=(job_id, api_key, temp_files)).start()
        
        return jsonify({
            'job_id': job_id,
            'status': 'pending',
            'message': 'Verification job started'
        }), 202
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/status/<job_id>', methods=['GET'])
def check_status(job_id):
    """Endpoint to check verification job status"""
    if job_id not in verification_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = verification_jobs[job_id]
    response = {
        'job_id': job_id,
        'status': job['status'],
        'start_time': job['start_time'],
        'status_updates': job['status_updates'],
        'progress': job.get('progress', {})
    }
    
    if job['status'] == 'complete':
        response['report'] = job['report']
    elif job['status'] == 'failed':
        response['error'] = job.get('error')
    
    return jsonify(response), 200

@app.route('/api/report/<job_id>', methods=['GET'])
def get_report(job_id):
    """Endpoint to retrieve verification report"""
    if job_id not in verification_jobs:
        return jsonify({'error': 'Job not found'}), 404
    
    job = verification_jobs[job_id]
    if job['status'] != 'complete':
        return jsonify({'error': 'Report not ready yet'}), 202
    
    return jsonify(job['report']), 200

@app.route('/api/cleanup', methods=['POST'])
def cleanup_jobs():
    """Endpoint to clean up completed jobs and temporary files"""
    try:
        cutoff_time = datetime.now() - timedelta(hours=24)
        jobs_to_delete = []
        
        for job_id, job in verification_jobs.items():
            if job['status'] in ['complete', 'failed'] and datetime.fromisoformat(job['start_time']) < cutoff_time:
                for temp_file in job.get('temp_files', []):
                    try:
                        os.remove(temp_file)
                    except:
                        pass
                jobs_to_delete.append(job_id)
        
        for job_id in jobs_to_delete:
            del verification_jobs[job_id]
        
        return jsonify({
            'message': f'Cleaned up {len(jobs_to_delete)} jobs',
            'deleted_jobs': jobs_to_delete
        }), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)  