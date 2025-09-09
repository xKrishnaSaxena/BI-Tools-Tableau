import { useState, useRef, useEffect } from "react";

function DataTable({ rows, columns }) {
  if (!rows || rows.length === 0) return null;
  const cols = columns && columns.length ? columns : Object.keys(rows[0]);

  return (
    <div className="overflow-x-auto rounded-xl border border-gray-700">
      <table className="min-w-full text-sm">
        <thead className="bg-gray-800">
          <tr>
            {cols.map((c) => (
              <th key={c} className="px-3 py-2 text-left font-semibold">
                {c}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((r, i) => (
            <tr key={i} className={i % 2 === 0 ? "bg-gray-900" : "bg-gray-800"}>
              {cols.map((c) => (
                <td key={c} className="px-3 py-2 whitespace-nowrap">
                  {String(r[c] ?? "")}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
const rowsToCsv = (rows, columns) => {
  if (!rows?.length) return "";
  const cols = columns && columns.length ? columns : Object.keys(rows[0]);
  const header = cols.join(",");
  const body = rows
    .map((r) =>
      cols
        .map((c) => {
          const v = r[c] ?? "";
          // CSV-escape
          const s = String(v).replace(/"/g, '""');
          return /[",\n]/.test(s) ? `"${s}"` : s;
        })
        .join(",")
    )
    .join("\n");
  return `${header}\n${body}`;
};

const downloadCsv = (rows, columns, filename = "tableau_view.csv") => {
  const csv = rowsToCsv(rows, columns);
  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = filename;
  a.click();
  URL.revokeObjectURL(url);
};

export default function App() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      text: "Hey! I’m Alex, your Tableau assistant.",
      sender: "bot",
      attachments: [],
    },
  ]);
  const [input, setInput] = useState("");
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () =>
    messagesEndRef.current?.scrollIntoView({ behavior: "smooth" });
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const handleSend = async () => {
    if (input.trim() === "" || isLoading) return;

    const userMessage = {
      id: Date.now(),
      text: input,
      sender: "user",
      attachments: [],
    };
    setMessages((prev) => [...prev, userMessage]);
    setInput("");
    setIsLoading(true);

    try {
      const response = await fetch(
        "https://agent-tableau-backend.onrender.com/chat",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: userMessage.text }),
        }
      );

      const data = await response.json();
      const botMessage = {
        id: Date.now() + 1,
        text: data.response || "Sorry, I encountered an issue.",
        sender: "bot",
        attachments: data.attachments || [],
      };
      setMessages((prev) => [...prev, botMessage]);
    } catch (error) {
      console.error("Failed to get response from backend:", error);
      const errorMessage = {
        id: Date.now() + 1,
        text: "Sorry, I couldn't connect to the backend. Is it running on :8000?",
        sender: "bot",
        attachments: [],
      };
      setMessages((prev) => [...prev, errorMessage]);
    } finally {
      setIsLoading(false);
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") handleSend();
  };

  return (
    <div className="font-sans bg-gray-900 text-white flex flex-col h-screen">
      {/* Header */}
      <header className="bg-gray-800 shadow-md p-4 flex items-center">
        <svg
          xmlns="http://www.w3.org/2000/svg"
          width="24"
          height="24"
          viewBox="0 0 24 24"
          fill="none"
          stroke="currentColor"
          strokeWidth="2"
          strokeLinecap="round"
          strokeLinejoin="round"
          className="text-blue-400 mr-3"
        >
          <path d="M3 3h18v4H3z" />
          <path d="M3 11h18v10H3z" />
        </svg>
        <h1 className="text-xl font-bold">Tableau AI Agent</h1>
      </header>

      {/* Messages */}
      <main className="flex-1 overflow-y-auto p-4 md:p-6 space-y-6">
        {messages.map((message) => (
          <div
            key={message.id}
            className={`flex items-start gap-3 ${
              message.sender === "user" ? "justify-end" : "justify-start"
            }`}
          >
            {message.sender === "bot" && (
              <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white font-bold text-sm flex-shrink-0">
                B
              </div>
            )}
            <div
              className={`max-w-xs md:max-w-2xl rounded-2xl px-4 py-3 shadow ${
                message.sender === "user"
                  ? "bg-blue-600 rounded-br-none"
                  : "bg-gray-700 rounded-bl-none"
              }`}
            >
              {message.text &&
                !message.text.trim().startsWith("data:image/") && (
                  <p className="whitespace-pre-wrap text-sm">{message.text}</p>
                )}

              {/* Attachments */}
              {message.attachments?.map((att, idx) => (
                <div key={idx} className="mt-3">
                  {att.type === "image" && (
                    <>
                      {att.caption ? (
                        <div className="text-xs text-gray-300 mb-1">
                          {att.caption}
                        </div>
                      ) : null}
                      <img
                        src={att.dataUrl}
                        alt={att.caption || "view"}
                        className="rounded-lg max-h-[480px]"
                      />
                    </>
                  )}
                  {att.type === "table" && (
                    <>
                      {att.caption ? (
                        <div className="text-xs text-gray-300 mb-2">
                          {att.caption}
                        </div>
                      ) : null}

                      <div className="flex justify-end mb-2">
                        <button
                          onClick={() => downloadCsv(att.rows, att.columns)}
                          className="text-xs px-2 py-1 rounded bg-gray-600 hover:bg-gray-500"
                        >
                          Download CSV
                        </button>
                      </div>

                      <DataTable rows={att.rows} columns={att.columns} />
                    </>
                  )}
                </div>
              ))}
            </div>
            {message.sender === "user" && (
              <div className="w-8 h-8 rounded-full bg-gray-600 flex items-center justify-center text-white font-bold text-sm flex-shrink-0">
                U
              </div>
            )}
          </div>
        ))}

        {isLoading && (
          <div className="flex items-end gap-3 justify-start">
            <div className="w-8 h-8 rounded-full bg-blue-500 flex items-center justify-center text-white font-bold text-sm flex-shrink-0">
              B
            </div>
            <div className="max-w-xs md:max-w-md rounded-2xl px-4 py-3 shadow bg-gray-700 rounded-bl-none">
              <div className="flex items-center justify-center space-x-1">
                <div className="w-2 h-2 bg-gray-300 rounded-full animate-pulse [animation-delay:-0.3s]"></div>
                <div className="w-2 h-2 bg-gray-300 rounded-full animate-pulse [animation-delay:-0.15s]"></div>
                <div className="w-2 h-2 bg-gray-300 rounded-full animate-pulse"></div>
              </div>
            </div>
          </div>
        )}
        <div ref={messagesEndRef} />
      </main>

      {/* Input */}
      <footer className="bg-gray-800 p-4">
        <div className="flex items-center bg-gray-700 rounded-full p-2">
          <input
            type="text"
            className="flex-1 bg-transparent text-white placeholder-gray-400 focus:outline-none px-4"
            placeholder="Ask me about your Tableau views…"
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyPress={handleKeyPress}
            disabled={isLoading}
          />
          <button
            onClick={handleSend}
            disabled={isLoading || input.trim() === ""}
            className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-500 disabled:cursor-not-allowed text-white rounded-full p-2 focus:outline-none transition-colors"
          >
            <svg
              xmlns="http://www.w3.org/2000/svg"
              width="24"
              height="24"
              viewBox="0 0 24 24"
              fill="none"
              stroke="currentColor"
              strokeWidth="2"
              strokeLinecap="round"
              strokeLinejoin="round"
            >
              <path d="m22 2-7 20-4-9-9-4Z" />
              <path d="M22 2 11 13" />
            </svg>
          </button>
        </div>
      </footer>
    </div>
  );
}
