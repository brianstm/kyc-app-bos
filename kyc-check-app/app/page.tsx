"use client";

import { useState, useEffect } from "react";
import axios from "axios";
import { FileUploader } from "./components/file-uploader";
import { AnalysisResults } from "./components/analysis-results";
import { Progress } from "@/components/ui/progress";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { useRouter, useSearchParams } from "next/navigation";
import ReactMarkdown from "react-markdown";
import { AnimatedList } from "@/components/magicui/animated-list";
import { cn } from "@/lib/utils";
import { BicepsFlexed } from "lucide-react";
import { Button } from "@/components/ui/button";

export default function Home() {
  const [jobId, setJobId] = useState<string | null>(null);
  const [status, setStatus] = useState("idle");
  const [progress, setProgress] = useState(0);
  const [statusUpdates, setStatusUpdates] = useState<any[]>([]);
  const [results, setResults] = useState(null);
  const router = useRouter();
  const searchParams = useSearchParams();

  useEffect(() => {
    const jobIdFromUrl = searchParams.get("jobId");
    if (jobIdFromUrl) {
      setJobId(jobIdFromUrl);
      setStatus("pending");
    }
  }, [searchParams]);

  useEffect(() => {
    let intervalId: NodeJS.Timeout;

    const pollStatus = async () => {
      if (!jobId) return;

      try {
        const response = await axios.get(
          `http://localhost:8000/api/status/${jobId}`
        );
        setStatus(response.data.status);
        setProgress(
          (response.data.progress.documents_loaded /
            response.data.progress.total_documents) *
            100
        );
        setStatusUpdates(response.data.status_updates);

        if (response.data.status === "complete") {
          clearInterval(intervalId);
          const reportResponse = await axios.get(
            `http://localhost:8000/api/report/${jobId}`
          );
          setResults(reportResponse.data);
        }
      } catch (error) {
        console.error("Error polling status:", error);
      }
    };

    if (jobId && status !== "complete") {
      intervalId = setInterval(pollStatus, 1000);
    }

    return () => clearInterval(intervalId);
  }, [jobId, status]);

  const handleUploadStart = (id: string) => {
    setJobId(id);
    setStatus("pending");
    setStatusUpdates([]);
    setProgress(0);
    setResults(null);
    router.push(`/?jobId=${id}`);
  };

  const handleReset = () => {
    setJobId(null);
    setStatus("idle");
    setStatusUpdates([]);
    setProgress(0);
    setResults(null);
    router.push("/");
  };

  const StatusUpdate = ({ status, timestamp, details }: any) => {
    return (
      <figure
        className={cn(
          "relative mx-auto min-h-fit w-full max-w-[800px] cursor-pointer overflow-hidden rounded-2xl p-4",
          "transition-all duration-200 ease-in-out hover:scale-[101%]",
          "bg-white [box-shadow:0_0_0_1px_rgba(0,0,0,.03),0_2px_4px_rgba(0,0,0,.05),0_12px_24px_rgba(0,0,0,.05)]",
          "transform-gpu dark:bg-transparent dark:backdrop-blur-md dark:[border:1px_solid_rgba(255,255,255,.1)] dark:[box-shadow:0_-20px_80px_-20px_#ffffff1f_inset]"
        )}
      >
        <div className="flex flex-row items-center gap-3">
          <div
            className="flex size-10 items-center justify-center rounded-2xl"
            style={{
              backgroundColor: "#1E86FF",
            }}
          >
            <span className="text-lg">🗞️</span>
          </div>
          <div className="flex flex-col overflow-hidden">
            <figcaption className="flex flex-row items-center whitespace-pre text-lg font-medium dark:text-white ">
              <span className="text-sm sm:text-lg">{status}</span>
              <span className="mx-1">·</span>
              <span className="text-xs text-gray-500">
                {new Date(timestamp).toLocaleString()}
              </span>
            </figcaption>
            {details && (
              <div className="text-sm font-normal dark:text-white/60">
                <ReactMarkdown>{`\`\`\`json\n${JSON.stringify(
                  details,
                  null,
                  2
                )}\n\`\`\``}</ReactMarkdown>
              </div>
            )}
          </div>
        </div>
      </figure>
    );
  };

  return (
    <main className="container mx-auto py-6">
      <div className="flex items-center justify-between mb-4">
        <h1 className="text-3xl font-bold flex items-center gap-2">
          <BicepsFlexed />
          KYC Analysis
        </h1>
        {status === "complete" && results && (
          <>
            <Button onClick={handleReset} variant="secondary">
              Start New Analysis
            </Button>
          </>
        )}
      </div>
      {!jobId && <FileUploader onUploadStart={handleUploadStart} />}
      {jobId && status !== "complete" && (
        <Card>
          <CardHeader>
            <CardTitle>Analysis in Progress</CardTitle>
          </CardHeader>
          <CardContent>
            <Progress value={progress} className="w-full mb-4" />
            <p className="mb-2">Job ID: {jobId}</p>
            <h3 className="font-semibold mb-2">Status Updates:</h3>
            <div className="relative h-[500px] w-full overflow-hidden p-2">
              <AnimatedList>
                {statusUpdates.reverse().map((update, idx) => (
                  <StatusUpdate key={idx} {...update} />
                ))}
              </AnimatedList>
              <div className="pointer-events-none absolute inset-x-0 bottom-0 h-1/4 bg-gradient-to-t from-background"></div>
            </div>
          </CardContent>
        </Card>
      )}
      {status === "complete" && results && (
        <>
          <AnalysisResults results={results} />
          <Button onClick={handleReset} variant="secondary" className="mt-8">
            Start New Analysis
          </Button>
        </>
      )}
    </main>
  );
}
