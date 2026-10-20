"use client";

import React, { useState, useEffect, useRef } from "react";
import {
  Sparkles,
  UploadCloud,
  Send,
  Image as ImageIcon,
  Paperclip,
  Activity,
  Layers,
  FileText,
  Clock,
  Cpu,
  CheckCircle2,
  AlertCircle,
  RefreshCw,
  Search,
  Zap,
  Sliders,
  Download,
} from "lucide-react";

interface SearchSource {
  doc_id: string;
  score: number;
  content: string;
  image_path?: string | null;
  metadata?: Record<string, any>;
}

interface PerformanceStats {
  ttft_ms: number;
  total_latency_ms: number;
  tokens_per_sec: number;
  retrieved_chunks_count: number;
}

interface Message {
  id: string;
  role: "user" | "assistant";
  content: string;
  imagePreview?: string | null;
  sources?: SearchSource[];
  stats?: PerformanceStats;
  isStreaming?: boolean;
}

export default function VisionRAGDashboard() {
  // Backend & Connection State
  const [backendConnected, setBackendConnected] = useState<boolean>(false);
  const [backendHealth, setBackendHealth] = useState<any>(null);

  // Upload State
  const [isDragging, setIsDragging] = useState<boolean>(false);
  const [isUploading, setIsUploading] = useState<boolean>(false);
  const [uploadMessage, setUploadMessage] = useState<{ type: "success" | "error"; text: string } | null>(null);

  // Hybrid Search Tuning State (Alpha Weight: 0.0 Sparse BM25 -> 1.0 Dense CLIP)
  const [alphaWeight, setAlphaWeight] = useState<number>(0.5);

  // Chat & WebSocket State
  const [messages, setMessages] = useState<Message[]>([
    {
      id: "welcome-1",
      role: "assistant",
      content: "Welcome to VisionRAG Engine! 👁️⚡ Upload documents/images or ask any multimodal question to perform hybrid CLIP + BM25 search with real-time streaming.",
    }
  ]);
  const [inputQuery, setInputQuery] = useState<string>("");
  const [selectedImage, setSelectedImage] = useState<string | null>(null);
  const [activeSources, setActiveSources] = useState<SearchSource[]>([]);
  const [isGenerating, setIsGenerating] = useState<boolean>(false);

  const chatContainerRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const imageAttachRef = useRef<HTMLInputElement>(null);

  // Check Backend Health
  const checkHealth = async () => {
    try {
      const res = await fetch("http://localhost:8000/health");
      if (res.ok) {
        const data = await res.json();
        setBackendConnected(true);
        setBackendHealth(data);
      } else {
        setBackendConnected(false);
      }
    } catch {
      setBackendConnected(false);
    }
  };

  useEffect(() => {
    checkHealth();
    const interval = setInterval(checkHealth, 10000);
    return () => clearInterval(interval);
  }, []);

  // Auto-scroll chat
  useEffect(() => {
    if (chatContainerRef.current) {
      chatContainerRef.current.scrollTop = chatContainerRef.current.scrollHeight;
    }
  }, [messages]);

  // Export Chat & RAG Citations Report as Markdown
  const handleExportReport = () => {
    let report = `# Vision RAG Engine — Chat & Citation Report\n`;
    report += `Generated On: ${new Date().toLocaleString()}\n`;
    report += `Hybrid Search Alpha (α): ${alphaWeight.toFixed(2)}\n\n`;

    report += `---\n\n## 💬 Conversation Log & Context Citations\n\n`;

    messages.forEach((msg, i) => {
      if (msg.role === "user") {
        report += `### 👤 User Query [${i + 1}]\n`;
        report += `> ${msg.content}\n`;
        if (msg.imagePreview) {
          report += `*Image Query Attached*\n`;
        }
        report += `\n`;
      } else {
        report += `### 🤖 Vision RAG Answer\n`;
        report += `${msg.content}\n\n`;

        if (msg.stats) {
          report += `**📊 Performance Metrics Telemetry:**\n`;
          report += `- **Time-To-First-Token (TTFT):** ${msg.stats.ttft_ms.toFixed(1)} ms\n`;
          report += `- **Total Latency:** ${msg.stats.total_latency_ms.toFixed(1)} ms\n`;
          report += `- **Generation Throughput:** ${msg.stats.tokens_per_sec.toFixed(1)} tok/s\n\n`;
        }

        if (msg.sources && msg.sources.length > 0) {
          report += `**📚 Cited Document Context Chunks (RRF Score):**\n`;
          msg.sources.forEach((src, idx) => {
            report += `${idx + 1}. **${src.metadata?.filename || src.doc_id}** *(RRF Score: ${src.score.toFixed(4)})*\n`;
            report += `   > "${src.content.replace(/\n/g, " ")}"\n\n`;
          });
        }
      }
    });

    const blob = new Blob([report], { type: "text/markdown;charset=utf-8" });
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `vision_rag_report_${Date.now()}.md`;
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    URL.revokeObjectURL(url);
  };

  // Handle Drag & Drop Upload
  const handleUploadFiles = async (files: FileList) => {
    if (!files || files.length === 0) return;
    setIsUploading(true);
    setUploadMessage(null);

    const formData = new FormData();
    Array.from(files).forEach((file) => formData.append("files", file));

    try {
      const res = await fetch("http://localhost:8000/api/v1/upload", {
        method: "POST",
        body: formData,
      });

      if (res.ok) {
        const data = await res.json();
        setUploadMessage({ type: "success", text: data.message || "Files successfully indexed!" });
        checkHealth();
      } else {
        const err = await res.json();
        setUploadMessage({ type: "error", text: err.detail || "Failed to upload files." });
      }
    } catch {
      setUploadMessage({ type: "error", text: "Network error while uploading." });
    } finally {
      setIsUploading(false);
    }
  };

  // Image attachment for Query
  const handleImageAttach = (file: File) => {
    const reader = new FileReader();
    reader.onloadend = () => {
      setSelectedImage(reader.result as string);
    };
    reader.readAsDataURL(file);
  };

  // Send Message via WebSocket
  const handleSendMessage = async (overrideQuery?: string) => {
    const queryToSend = overrideQuery || inputQuery;
    if ((!queryToSend.trim() && !selectedImage) || isGenerating) return;

    const userMessageId = `user-${Date.now()}`;
    const assistantMessageId = `assistant-${Date.now()}`;

    const userMsg: Message = {
      id: userMessageId,
      role: "user",
      content: queryToSend,
      imagePreview: selectedImage,
    };

    const assistantMsg: Message = {
      id: assistantMessageId,
      role: "assistant",
      content: "",
      isStreaming: true,
    };

    setMessages((prev) => [...prev, userMsg, assistantMsg]);
    setInputQuery("");
    const attachedImg = selectedImage;
    setSelectedImage(null);
    setIsGenerating(true);

    try {
      const ws = new WebSocket("ws://localhost:8000/api/v1/ws/chat");

      ws.onopen = () => {
        ws.send(
          JSON.stringify({
            query: queryToSend,
            image_base64: attachedImg,
            top_k: 3,
            alpha: alphaWeight,
          })
        );
      };

      ws.onmessage = (event) => {
        const data = JSON.parse(event.data);

        if (data.type === "sources") {
          setActiveSources(data.sources || []);
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMessageId ? { ...msg, sources: data.sources } : msg
            )
          );
        } else if (data.type === "chunk") {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMessageId
                ? { ...msg, content: msg.content + data.token }
                : msg
            )
          );
        } else if (data.type === "end") {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMessageId
                ? { ...msg, isStreaming: false, stats: data.stats }
                : msg
            )
          );
          setIsGenerating(false);
          ws.close();
        } else if (data.type === "error") {
          setMessages((prev) =>
            prev.map((msg) =>
              msg.id === assistantMessageId
                ? { ...msg, isStreaming: false, content: `Error: ${data.error}` }
                : msg
            )
          );
          setIsGenerating(false);
          ws.close();
        }
      };

      ws.onerror = () => {
        setMessages((prev) =>
          prev.map((msg) =>
            msg.id === assistantMessageId
              ? { ...msg, isStreaming: false, content: "WebSocket error connecting to server." }
              : msg
          )
        );
        setIsGenerating(false);
      };
    } catch {
      setIsGenerating(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-slate-950 text-slate-100 font-sans antialiased overflow-hidden">
      {/* HEADER */}
      <header className="h-16 border-b border-slate-800/80 bg-slate-900/60 backdrop-blur-md flex items-center justify-between px-6 z-10">
        <div className="flex items-center space-x-3">
          <div className="w-10 h-10 rounded-xl bg-gradient-to-tr from-cyan-500 via-indigo-500 to-purple-600 p-0.5 shadow-lg shadow-cyan-500/20">
            <div className="w-full h-full bg-slate-950 rounded-[10px] flex items-center justify-center">
              <Sparkles className="w-5 h-5 text-cyan-400 animate-pulse" />
            </div>
          </div>
          <div>
            <h1 className="text-base font-bold tracking-tight bg-gradient-to-r from-slate-100 via-cyan-200 to-indigo-200 bg-clip-text text-transparent">
              VisionRAG Engine
            </h1>
            <p className="text-[11px] text-slate-400 flex items-center space-x-2">
              <span>Multimodal Hybrid CLIP + BM25 Retrieval</span>
              <span className="text-slate-600">•</span>
              <span className="text-cyan-400 font-mono">v0.1.0</span>
            </p>
          </div>
        </div>

        {/* Header Actions & Health Indicator */}
        <div className="flex items-center space-x-3">
          <button
            onClick={handleExportReport}
            className="flex items-center space-x-1.5 px-3 py-1.5 rounded-lg bg-cyan-950/60 hover:bg-cyan-900/80 border border-cyan-800/80 text-cyan-300 text-xs font-medium transition-colors shadow-sm"
            title="Export Conversation & Citations Report as Markdown"
          >
            <Download className="w-3.5 h-3.5 text-cyan-400" />
            <span>Export Report</span>
          </button>

          <div className="flex items-center space-x-2 px-3 py-1.5 rounded-full bg-slate-950/80 border border-slate-800 text-xs">
            <span className={`w-2.5 h-2.5 rounded-full ${backendConnected ? "bg-emerald-500 shadow-lg shadow-emerald-500/50" : "bg-rose-500"}`} />
            <span className="font-medium text-slate-300">
              {backendConnected ? "Backend Connected" : "Backend Offline"}
            </span>
            {backendConnected && (
              <span className="text-[10px] text-slate-500 pl-1 border-l border-slate-800">
                localhost:8000
              </span>
            )}
          </div>
          <button
            onClick={checkHealth}
            className="p-1.5 hover:bg-slate-800 rounded-lg text-slate-400 hover:text-slate-200 transition-colors"
            title="Refresh Connection"
          >
            <RefreshCw className="w-4 h-4" />
          </button>
        </div>
      </header>

      {/* MAIN SPLIT-SCREEN LAYOUT */}
      <div className="flex flex-1 overflow-hidden">
        {/* LEFT SIDEBAR: INGESTION & CONTEXT DRAWER */}
        <aside className="w-80 md:w-96 flex flex-col bg-slate-900/50 border-r border-slate-800/80 overflow-y-auto">
          {/* SECTION 1: DRAG & DROP FILE UPLOADER */}
          <div className="p-4 border-b border-slate-800/80">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center space-x-1.5">
                <UploadCloud className="w-4 h-4 text-cyan-400" />
                <span>Ingest Documents & Images</span>
              </h2>
            </div>

            <div
              onDragOver={(e) => {
                e.preventDefault();
                setIsDragging(true);
              }}
              onDragLeave={() => setIsDragging(false)}
              onDrop={(e) => {
                e.preventDefault();
                setIsDragging(false);
                if (e.dataTransfer.files) {
                  handleUploadFiles(e.dataTransfer.files);
                }
              }}
              onClick={() => fileInputRef.current?.click()}
              className={`flex flex-col items-center justify-center p-5 border-2 border-dashed rounded-xl cursor-pointer transition-all ${
                isDragging
                  ? "border-cyan-400 bg-cyan-950/20"
                  : "border-slate-700/80 hover:border-slate-600 bg-slate-950/40 hover:bg-slate-950/60"
              }`}
            >
              <input
                ref={fileInputRef}
                type="file"
                multiple
                className="hidden"
                accept=".txt,.md,.pdf,.png,.jpg,.jpeg"
                onChange={(e) => e.target.files && handleUploadFiles(e.target.files)}
              />
              <UploadCloud className="w-8 h-8 text-cyan-400/80 mb-2" />
              <p className="text-xs font-medium text-slate-200 text-center">
                Drag files here or <span className="text-cyan-400 underline">browse</span>
              </p>
              <p className="text-[10px] text-slate-500 mt-1">Supports PDF, TXT, MD, PNG, JPG</p>
            </div>

            {isUploading && (
              <div className="mt-3 flex items-center space-x-2 text-xs text-cyan-400">
                <RefreshCw className="w-3.5 h-3.5 animate-spin" />
                <span>Extracting features & indexing vectors...</span>
              </div>
            )}

            {uploadMessage && (
              <div
                className={`mt-3 p-2.5 rounded-lg text-xs flex items-center space-x-2 ${
                  uploadMessage.type === "success"
                    ? "bg-emerald-950/50 border border-emerald-800/80 text-emerald-300"
                    : "bg-rose-950/50 border border-rose-800/80 text-rose-300"
                }`}
              >
                {uploadMessage.type === "success" ? (
                  <CheckCircle2 className="w-4 h-4 shrink-0 text-emerald-400" />
                ) : (
                  <AlertCircle className="w-4 h-4 shrink-0 text-rose-400" />
                )}
                <span>{uploadMessage.text}</span>
              </div>
            )}
          </div>

          {/* SECTION 2: HYBRID RETRIEVAL ALPHA SLIDER */}
          <div className="p-4 border-b border-slate-800/80 bg-slate-950/20">
            <div className="flex items-center justify-between mb-2">
              <label className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center space-x-1.5">
                <Sliders className="w-4 h-4 text-purple-400" />
                <span>Hybrid Weight Balancing (α)</span>
              </label>
              <span className="text-xs font-mono font-bold text-cyan-400 bg-slate-900 px-2 py-0.5 rounded border border-slate-800">
                {alphaWeight.toFixed(2)}
              </span>
            </div>
            <input
              type="range"
              min="0.0"
              max="1.0"
              step="0.05"
              value={alphaWeight}
              onChange={(e) => setAlphaWeight(parseFloat(e.target.value))}
              className="w-full h-1.5 bg-slate-800 rounded-lg appearance-none cursor-pointer accent-cyan-400"
            />
            <div className="flex justify-between text-[10px] text-slate-500 mt-1 font-mono">
              <span>0.0 (BM25 Sparse)</span>
              <span>0.5 (Balanced RRF)</span>
              <span>1.0 (CLIP Dense)</span>
            </div>
          </div>

          {/* SECTION 3: INDEXED CONTEXT & RRF SCORE BREAKDOWNS */}
          <div className="flex-1 p-4 overflow-y-auto">
            <div className="flex items-center justify-between mb-3">
              <h2 className="text-xs font-bold uppercase tracking-wider text-slate-400 flex items-center space-x-1.5">
                <Layers className="w-4 h-4 text-indigo-400" />
                <span>Retrieved Context (RRF)</span>
              </h2>
              <span className="text-[10px] font-mono text-slate-500 bg-slate-800 px-1.5 py-0.5 rounded">
                k = 60
              </span>
            </div>

            {activeSources.length === 0 ? (
              <div className="text-center py-8 px-4 border border-dashed border-slate-800 rounded-xl bg-slate-950/20">
                <Search className="w-6 h-6 text-slate-600 mx-auto mb-2" />
                <p className="text-xs text-slate-500">No active context sources.</p>
                <p className="text-[10px] text-slate-600 mt-1">
                  Ask a question to see real-time Reciprocal Rank Fusion scores.
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {activeSources.map((source, idx) => (
                  <div
                    key={source.doc_id || idx}
                    className="p-3 bg-slate-950/60 border border-slate-800/80 rounded-xl hover:border-slate-700 transition-all space-y-2"
                  >
                    <div className="flex items-center justify-between text-xs">
                      <span className="font-mono text-cyan-400 font-medium truncate max-w-[180px]">
                        [{idx + 1}] {source.metadata?.filename || source.doc_id}
                      </span>
                      <span className="px-2 py-0.5 rounded bg-indigo-950/80 border border-indigo-800/50 text-indigo-300 font-mono text-[10px] font-bold">
                        RRF: {source.score.toFixed(4)}
                      </span>
                    </div>

                    <p className="text-xs text-slate-300 line-clamp-3 leading-relaxed font-sans">
                      {source.content}
                    </p>

                    {source.image_path && (
                      <div className="mt-2 text-[10px] text-slate-400 flex items-center space-x-1">
                        <ImageIcon className="w-3 h-3 text-cyan-400" />
                        <span className="truncate">{source.image_path}</span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            )}
          </div>
        </aside>

        {/* MAIN CHAT & STREAMING SECTION */}
        <main className="flex-1 flex flex-col bg-slate-950">
          {/* MESSAGES HISTORY */}
          <div ref={chatContainerRef} className="flex-1 overflow-y-auto p-6 space-y-6">
            {messages.map((msg) => (
              <div
                key={msg.id}
                className={`flex space-x-4 max-w-4xl ${
                  msg.role === "user" ? "ml-auto flex-row-reverse space-x-reverse" : ""
                }`}
              >
                <div
                  className={`w-8 h-8 rounded-xl flex items-center justify-center shrink-0 text-xs font-bold ${
                    msg.role === "user"
                      ? "bg-cyan-500 text-slate-950 shadow-md shadow-cyan-500/20"
                      : "bg-indigo-600 text-white shadow-md shadow-indigo-600/20"
                  }`}
                >
                  {msg.role === "user" ? "U" : "AI"}
                </div>

                <div className="space-y-3 flex-1">
                  <div
                    className={`p-4 rounded-2xl border text-sm leading-relaxed ${
                      msg.role === "user"
                        ? "bg-cyan-950/30 border-cyan-800/50 text-slate-100 rounded-tr-none"
                        : "bg-slate-900/80 border-slate-800/80 text-slate-200 rounded-tl-none"
                    }`}
                  >
                    {msg.imagePreview && (
                      <div className="mb-3 max-w-xs overflow-hidden rounded-lg border border-slate-700">
                        <img src={msg.imagePreview} alt="Attached Preview" className="w-full object-cover" />
                      </div>
                    )}

                    <div className="whitespace-pre-wrap font-sans">{msg.content}</div>

                    {msg.isStreaming && (
                      <span className="inline-block w-2 h-4 ml-1 bg-cyan-400 animate-pulse align-middle" />
                    )}
                  </div>

                  {/* Performance Metrics Telemetry Bar */}
                  {msg.stats && (
                    <div className="flex flex-wrap gap-2 text-[11px] font-mono">
                      <div className="px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-cyan-400 flex items-center space-x-1.5">
                        <Clock className="w-3 h-3 text-cyan-400" />
                        <span>TTFT: {msg.stats.ttft_ms.toFixed(1)} ms</span>
                      </div>
                      <div className="px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-indigo-400 flex items-center space-x-1.5">
                        <Zap className="w-3 h-3 text-indigo-400" />
                        <span>Latency: {msg.stats.total_latency_ms.toFixed(1)} ms</span>
                      </div>
                      <div className="px-2.5 py-1 rounded-md bg-slate-900 border border-slate-800 text-emerald-400 flex items-center space-x-1.5">
                        <Cpu className="w-3 h-3 text-emerald-400" />
                        <span>Throughput: {msg.stats.tokens_per_sec.toFixed(1)} tok/s</span>
                      </div>
                    </div>
                  )}
                </div>
              </div>
            ))}
          </div>

          {/* INPUT BAR */}
          <div className="p-4 border-t border-slate-800/80 bg-slate-900/40 backdrop-blur-md">
            {selectedImage && (
              <div className="mb-3 flex items-center space-x-2 bg-slate-900 p-2 rounded-lg border border-slate-800 max-w-xs">
                <img src={selectedImage} alt="Attached query" className="w-10 h-10 object-cover rounded" />
                <span className="text-xs text-slate-300 truncate flex-1">Attached Image Query</span>
                <button
                  onClick={() => setSelectedImage(null)}
                  className="text-slate-500 hover:text-slate-300 text-xs px-2"
                >
                  ✕
                </button>
              </div>
            )}

            <form
              onSubmit={(e) => {
                e.preventDefault();
                handleSendMessage();
              }}
              className="flex items-center space-x-3 bg-slate-950 border border-slate-800 rounded-2xl p-2 focus-within:border-cyan-500/50 focus-within:ring-1 focus-within:ring-cyan-500/50 transition-all"
            >
              <input
                ref={imageAttachRef}
                type="file"
                accept="image/*"
                className="hidden"
                onChange={(e) => e.target.files?.[0] && handleImageAttach(e.target.files[0])}
              />
              <button
                type="button"
                onClick={() => imageAttachRef.current?.click()}
                className="p-2 hover:bg-slate-800 rounded-xl text-slate-400 hover:text-cyan-400 transition-colors"
                title="Attach Image Query"
              >
                <ImageIcon className="w-5 h-5" />
              </button>

              <input
                type="text"
                value={inputQuery}
                onChange={(e) => setInputQuery(e.target.value)}
                placeholder="Ask a question or search documents with vision & text..."
                className="flex-1 bg-transparent text-sm text-slate-100 placeholder-slate-500 focus:outline-none px-2"
              />

              <button
                type="submit"
                disabled={isGenerating || (!inputQuery.trim() && !selectedImage)}
                className="p-2.5 bg-gradient-to-r from-cyan-500 to-indigo-600 text-slate-950 rounded-xl font-bold hover:opacity-90 disabled:opacity-40 transition-opacity shadow-lg shadow-cyan-500/20"
              >
                <Send className="w-4 h-4 text-slate-950" />
              </button>
            </form>
          </div>
        </main>
      </div>
    </div>
  );
}
