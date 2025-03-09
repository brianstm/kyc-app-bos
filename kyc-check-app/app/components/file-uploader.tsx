"use client";

import { useState, useCallback } from "react";
import { useDropzone } from "react-dropzone";
import axios from "axios";
import { Button } from "@/components/ui/button";
import { Upload } from "lucide-react";

interface FileUploaderProps {
  onUploadStart: (jobId: string) => void;
}

export function FileUploader({ onUploadStart }: FileUploaderProps) {
  const [files, setFiles] = useState<File[]>([]);

  const onDrop = useCallback((acceptedFiles: File[]) => {
    setFiles(acceptedFiles);
  }, []);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({ onDrop });

  const uploadFiles = async () => {
    const formData = new FormData();
    files.forEach((file) => formData.append("files", file));

    try {
      const response = await axios.post(
        "http://localhost:8000/api/verify",
        formData,
        {
          headers: { "Content-Type": "multipart/form-data" },
        }
      );
      onUploadStart(response.data.job_id);
    } catch (error) {
      console.error("Upload failed:", error);
    }
  };

  return (
    <div className="space-y-4">
      <div
        {...getRootProps()}
        className={`border-2 border-dashed rounded-lg p-8 text-center cursor-pointer ${
          isDragActive ? "border-primary" : "border-gray-300"
        }`}
      >
        <input {...getInputProps()} />
        <Upload className="mx-auto h-12 w-12 text-gray-400" />
        <p className="mt-2">Drag & drop files here, or click to select files</p>
      </div>
      {files.length > 0 && (
        <div>
          <h3 className="font-semibold mb-2">Selected Files:</h3>
          <ul className="list-disc pl-5">
            {files.map((file) => (
              <li key={file.name}>{file.name}</li>
            ))}
          </ul>
          <Button onClick={uploadFiles} className="mt-4">
            Start Analysis
          </Button>
        </div>
      )}
    </div>
  );
}
