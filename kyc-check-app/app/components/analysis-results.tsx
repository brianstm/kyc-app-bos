"use client";

import { useState } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "@/components/ui/card";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
} from "recharts";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { JSONTree } from "./json-tree";

interface AnalysisResultsProps {
  results: any;
}

export function AnalysisResults({ results }: AnalysisResultsProps) {
  const [activeTab, setActiveTab] = useState("overview");

  const overviewData = [
    {
      name: "Verification Score",
      value: results?.overall_assessment?.verification_score,
    },
    {
      name: "Document Count",
      value: results?.overall_assessment?.document_count,
    },
    {
      name: "Avg Document Score",
      value: results?.overall_assessment?.average_document_score,
    },
  ];

  return (
    <Card className="w-full">
      <CardHeader>
        <CardTitle>Analysis Results</CardTitle>
        <CardDescription>
          Detailed breakdown of the document analysis
        </CardDescription>
      </CardHeader>
      <CardContent>
        <Tabs value={activeTab} onValueChange={setActiveTab}>
          <TabsList className="grid w-full grid-cols-5">
            <TabsTrigger value="overview">Overview</TabsTrigger>
            <TabsTrigger value="details">Document Details</TabsTrigger>
            <TabsTrigger value="consistency">Consistency</TabsTrigger>
            <TabsTrigger value="visual">Visual Analysis</TabsTrigger>
            <TabsTrigger value="raw">Raw Data</TabsTrigger>
          </TabsList>
          <TabsContent value="overview">
            <div className="space-y-4">
              <div className="grid grid-cols-2 gap-4">
                <Card>
                  <CardHeader>
                    <CardTitle>Risk Level</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-2xl font-bold">
                      {results?.overall_assessment?.risk_level}
                    </p>
                  </CardContent>
                </Card>
                <Card>
                  <CardHeader>
                    <CardTitle>Verification Status</CardTitle>
                  </CardHeader>
                  <CardContent>
                    <p className="text-2xl font-bold">
                      {results?.overall_assessment?.verification_status}
                    </p>
                  </CardContent>
                </Card>
              </div>
              <Card>
                <CardHeader>
                  <CardTitle>Assessment Details</CardTitle>
                </CardHeader>
                <CardContent className="prose dark:prose-invert max-w-none">
                  <ReactMarkdown remarkPlugins={[remarkGfm]}>
                    {results?.overall_assessment?.detailed_assessment}
                  </ReactMarkdown>
                </CardContent>
              </Card>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart data={overviewData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Bar dataKey="value" fill="#8884d8" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </TabsContent>
          <TabsContent value="details">
            {results.document_details.map((doc: any, index: number) => (
              <Card key={index} className="mb-4">
                <CardHeader>
                  <CardTitle>{doc.file_name}</CardTitle>
                  <CardDescription>
                    Document ID: {doc.document_id}
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="grid grid-cols-3 gap-4 mb-4">
                    <div>
                      <p className="font-medium">Authenticity Score</p>
                      <p className="text-2xl">{doc.authenticity_score}</p>
                    </div>
                    <div>
                      <p className="font-medium">Risk Level</p>
                      <p className="text-2xl">{doc.risk_level}</p>
                    </div>
                    <div>
                      <p className="font-medium">Document Type</p>
                      <p className="text-2xl">{doc.document_type}</p>
                    </div>
                  </div>
                  <div className="prose dark:prose-invert max-w-none">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {doc.extracted_info}
                    </ReactMarkdown>
                  </div>
                </CardContent>
              </Card>
            ))}
          </TabsContent>
          <TabsContent value="consistency">
            {Object.entries(
              results.raw_verification_results.consistency_check
            ).map(([docId, data]: [string, any]) => (
              <Card key={docId} className="mb-4">
                <CardHeader>
                  <CardTitle>Consistency Check - {docId}</CardTitle>
                  <CardDescription>
                    Consistency Score: {data.consistency_score}%
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="prose dark:prose-invert max-w-none">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {data.analysis}
                    </ReactMarkdown>
                  </div>
                </CardContent>
              </Card>
            ))}
          </TabsContent>
          <TabsContent value="visual">
            {Object.entries(
              results.raw_verification_results.visual_integrity
            ).map(([docId, data]: [string, any]) => (
              <Card key={docId} className="mb-4">
                <CardHeader>
                  <CardTitle>Visual Analysis - {docId}</CardTitle>
                  <CardDescription>
                    Visual Integrity Score: {data.visual_integrity_score}%
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  <div className="prose dark:prose-invert max-w-none">
                    <ReactMarkdown remarkPlugins={[remarkGfm]}>
                      {data.analysis}
                    </ReactMarkdown>
                  </div>
                </CardContent>
              </Card>
            ))}
          </TabsContent>
          <TabsContent value="raw">
            <Card>
              <CardHeader>
                <CardTitle>Raw Verification Results</CardTitle>
              </CardHeader>
              <CardContent>
                <JSONTree data={results} />
              </CardContent>
            </Card>
          </TabsContent>
        </Tabs>
      </CardContent>
    </Card>
  );
}
