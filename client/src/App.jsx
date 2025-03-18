import { Fragment, useState, useRef } from "react";

const App = () => {
  const [text, setText] = useState("");
  const [summary, setSummary] = useState("");
  const [loading, setLoading] = useState(false);
  const [summaryLength, setSummaryLength] = useState(3); 
  const [fileName, setFileName] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleSummarize = async () => {
    if (!text.trim()) return;

    setLoading(true);
    setSummary("");

    try {
      const response = await fetch("http://127.0.0.1:5000/summarize", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          text,
          sentences: summaryLength 
        }),
      });

      const data = await response.json();
      if (data.summary) setSummary(data.summary);
    } catch (error) {
      console.error("Error summarizing text:", error);
    }
    setLoading(false);
  };

  const processFile = async (file) => {
    if (!file) return;

    setFileName(file.name);
    setLoading(true);
    setText("");
    setSummary("");

    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://127.0.0.1:5000/extract", {
        method: "POST",
        body: formData,
      });

      const data = await response.json();
      if (data.text) {
        setText(data.text);
      }
    } catch (error) {
      console.error("Error extracting text from file:", error);
    }

    setLoading(false);
  };

  const handleFileUpload = (e) => {
    const file = e.target.files[0];
    processFile(file);
  };

  const removeFile = () => {
    setFileName("");
    setText("");
    setSummary("");
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
    }
  };

  const handleDragEnter = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (!isDragging) {
      setIsDragging(true);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setIsDragging(false);
    
    const files = e.dataTransfer.files;
    if (files && files.length > 0) {
      processFile(files[0]);
    }
  };

  return (
    <Fragment>
      <div className="bg-gray-100 min-h-screen pt-20">
        <h1 className="md:text-5xl text-4xl text-center font-medium text-bold mb-10">Монгол хураангуй</h1>
        <div className="flex items-center justify-center">
          <div className="p-6">
            <div className="bg-white rounded-lg shadow-lg lg:min-w-4xl w-full max-w-6xl">
              <h1 className="md:text-2xl text-xlq font-medium p-6">Текст хураангуйлах</h1>

              <div className="flex flex-col lg:flex-row gap-6 p-6">
                <div className="flex-1">
                  {/* File Upload Section with Drag & Drop */}

                  {!fileName && (
                    <div 
                      className={`mb-4 border-2 border-dashed rounded-lg p-6 flex flex-col items-center justify-center
                        ${isDragging ? 'border-green-500 bg-green-50' : 'border-gray-300 hover:border-green-400'}
                        transition-all duration-200`}
                      onDragEnter={handleDragEnter}
                      onDragOver={handleDragOver}
                      onDragLeave={handleDragLeave}
                      onDrop={handleDrop}
                      onClick={() => fileInputRef.current.click()}
                    >
                      <svg 
                        xmlns="http://www.w3.org/2000/svg" 
                        className={`h-12 w-12 mb-2 ${isDragging ? 'text-green-500' : 'text-gray-400'}`}
                        fill="none" 
                        viewBox="0 0 24 24" 
                        stroke="currentColor"
                      >
                        <path 
                          strokeLinecap="round" 
                          strokeLinejoin="round" 
                          strokeWidth={2} 
                          d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" 
                        />
                      </svg>
                      <p className="mb-2 text-sm text-gray-700">
                        <span className="font-semibold">Файл чирэх</span> эсвэл дарж сонгох
                      </p>
                      <p className="text-xs text-gray-500">
                        PDF, DOC, DOCX, TXT (Хамгийн ихдээ 10MB)
                      </p>
                      <input
                        ref={fileInputRef}
                        type="file"
                        className="hidden"
                        accept=".txt,.doc,.docx,.pdf"
                        onChange={handleFileUpload}
                      />
                    </div>
                  )}
                  
                  {fileName && (
                    <div className="mb-4 p-3 bg-green-50 border border-green-200 rounded-lg flex items-center justify-between">
                      <div className="flex items-center">
                        <svg 
                          xmlns="http://www.w3.org/2000/svg" 
                          className="h-5 w-5 text-green-500 mr-2" 
                          fill="none" 
                          viewBox="0 0 24 24" 
                          stroke="currentColor"
                        >
                          <path 
                            strokeLinecap="round" 
                            strokeLinejoin="round" 
                            strokeWidth={2} 
                            d="M9 12l2 2 4-4m6 2a9 9 0 11-18 0 9 9 0 0118 0z" 
                          />
                        </svg>
                        <span className="text-sm text-gray-700">
                          {fileName}
                        </span>
                      </div>
                      <button 
                        onClick={removeFile}
                        className="text-red-500 hover:text-red-700 p-1 rounded-full hover:bg-red-50 transition-colors"
                        title="Файл устгах"
                      >
                        <svg 
                          xmlns="http://www.w3.org/2000/svg" 
                          className="h-5 w-5" 
                          fill="none" 
                          viewBox="0 0 24 24" 
                          stroke="currentColor"
                        >
                          <path 
                            strokeLinecap="round" 
                            strokeLinejoin="round" 
                            strokeWidth={2} 
                            d="M6 18L18 6M6 6l12 12" 
                          />
                        </svg>
                      </button>
                    </div>
                  )}

                  <textarea
                    className="w-full h-64 p-4 border rounded-lg focus:outline-none focus:ring-2 focus:ring-green-500 resize-none"
                    placeholder="Текст оруулна уу... эсвэл файл чирэх"
                    value={text}
                    onChange={(e) => setText(e.target.value)}
                  />

                  {/* Summary Length Control */}
                  <div className="mt-4 mb-4">
                    <div className="flex justify-between mb-2">
                      <span className="text-gray-700">Хураангуйн урт:</span>
                      <span className="text-gray-700 font-medium">{summaryLength*10} %</span>
                    </div>
                    <div className="flex items-center w-full">
                      <span className="mr-2 text-sm text-gray-500">Богино</span>
                      <input
                        type="range"
                        min="1"
                        max="10"
                        value={summaryLength}
                        onChange={(e) => setSummaryLength(parseInt(e.target.value))}
                        className="w-full h-2 bg-gray-200 rounded-lg appearance-none cursor-pointer"
                      />
                      <span className="ml-2 text-sm text-gray-500">Урт</span>
                    </div>
                  </div>

                  <button
                    className="w-full bg-blue-800 text-white py-3 rounded-lg hover:bg-blue-600 disabled:opacity-50 transition-colors"
                    onClick={handleSummarize}
                    disabled={loading || !text.trim()}
                  >
                    {loading ? "Хураангуйлж байна..." : "Хураангуйлах"}
                  </button>
                </div>

                {/* Summary Section */}
                <div className="flex-1 bg-gray-50 rounded-lg p-6">
                  <h2 className="text-lg font-semibold mb-4">Хураангуй:</h2>
                  
                  {summary ? (
                    <p className="text-gray-700">{summary}</p>
                  ) : (
                    <p className="text-gray-500 italic">
                      Хураангуйлагдсан текст
                    </p>
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Fragment>
  );
};

export default App;