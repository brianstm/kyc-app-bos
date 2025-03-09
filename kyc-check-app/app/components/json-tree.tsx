"use client";

import type React from "react";
import { useState } from "react";
import { ChevronRight, ChevronDown } from "lucide-react";

interface JSONTreeProps {
  data: any;
  level?: number;
}

export const JSONTree: React.FC<JSONTreeProps> = ({ data, level = 0 }) => {
  const [isExpanded, setIsExpanded] = useState(level < 2);

  const toggleExpand = () => setIsExpanded(!isExpanded);

  if (typeof data !== "object" || data === null) {
    return <span className="text-green-600">{JSON.stringify(data)}</span>;
  }

  const isArray = Array.isArray(data);

  return (
    <div style={{ marginLeft: level > 0 ? "1.5rem" : "0" }}>
      <span onClick={toggleExpand} className="cursor-pointer select-none">
        {isExpanded ? (
          <ChevronDown className="inline w-4 h-4" />
        ) : (
          <ChevronRight className="inline w-4 h-4" />
        )}
        {isArray ? "Array" : "Object"}
      </span>
      {isExpanded && (
        <div>
          {Object.entries(data).map(([key, value]) => (
            <div key={key}>
              <span className="text-blue-600">{key}: </span>
              <JSONTree data={value} level={level + 1} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
};
