import React, { useState } from 'react';
import './index.css'; // Ensure Tailwind CSS is imported
import { Upload, Lightbulb, CheckCircle, XCircle, Loader2, Image as ImageIcon, MousePointerClick } from 'lucide-react';
import ReactMarkdown from 'react-markdown'; // Import ReactMarkdown
import remarkGfm from 'remark-gfm'; // Import remarkGfm for GitHub Flavored Markdown

// IMPORTANT: Replace with your Render backend URL when deployed
// For local development, keep it as localhost
const BASE_API_URL = 'https://ai-problem-solver.onrender.com/';

function App() {
  const [selectedImage, setSelectedImage] = useState(null);
  const [imagePreview, setImagePreview] = useState(null);
  const [userContext, setUserContext] = useState('');
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);

  // Helper function to resize the image before upload
  const resizeImage = (file) => {
    return new Promise((resolve, reject) => {
      const reader = new FileReader();
      reader.onload = (event) => {
        const img = new Image();
        img.onload = () => {
          const canvas = document.createElement('canvas');
          const MAX_DIMENSION = 256; // Target max dimension (e.g., 256x256 for ResNet input)
          let width = img.width;
          let height = img.height;

          // Calculate new dimensions while maintaining aspect ratio
          if (width > height) {
            if (width > MAX_DIMENSION) {
              height *= MAX_DIMENSION / width;
              width = MAX_DIMENSION;
            }
          } else {
            if (height > MAX_DIMENSION) {
              width *= MAX_DIMENSION / height;
              height = MAX_DIMENSION;
            }
          }

          canvas.width = width;
          canvas.height = height;
          const ctx = canvas.getContext('2d');
          ctx.drawImage(img, 0, 0, width, height);

          // Convert canvas content to a Blob (File-like object)
          // Adjust quality (0.7 is a good balance for web)
          canvas.toBlob((blob) => {
            if (blob) {
              // Create a new File object with the original name but resized content
              const resizedFile = new File([blob], file.name, {
                type: 'image/jpeg', // Force JPEG for smaller size, or use original file.type
                lastModified: Date.now(),
              });
              resolve(resizedFile);
            } else {
              reject(new Error("Canvas to Blob conversion failed."));
            }
          }, 'image/jpeg', 0.7); // Output as JPEG with 70% quality
        };
        img.onerror = (err) => reject(new Error("Image loading failed: " + err.message));
        img.src = event.target.result;
      };
      reader.onerror = (err) => reject(new Error("File reading failed: " + err.message));
      reader.readAsDataURL(file);
    });
  };

  const handleFileChange = (file) => {
    if (file && file.type.startsWith('image/')) {
      setSelectedImage(file);
      setImagePreview(URL.createObjectURL(file));
      setResult(null);
      setError(null);
    } else if (file) {
      setError("Please upload a valid image file (e.g., JPG, PNG).");
      setSelectedImage(null);
      setImagePreview(null);
    }
  };

  const handleFileInputChange = (event) => {
    handleFileChange(event.target.files[0]);
  };

  // Drag and Drop Handlers
  const handleDragOver = (event) => {
    event.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  const handleDrop = (event) => {
    event.preventDefault();
    setIsDragging(false);
    if (event.dataTransfer.files && event.dataTransfer.files.length > 0) {
      handleFileChange(event.dataTransfer.files[0]);
      event.dataTransfer.clearData();
    }
  };

  const handleSubmit = async () => {
    if (!selectedImage) {
      setError("Please select or drag & drop an image first.");
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);

    try {
      // Resize the image before sending
      const resizedImage = await resizeImage(selectedImage);

      const formData = new FormData();
      formData.append('image', resizedImage); // Append the resized image
      formData.append('context', userContext);

      const response = await fetch(`${BASE_API_URL}/identify_issue`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        // Attempt to parse error data if available
        let errorData = {};
        try {
          errorData = await response.json();
        } catch (jsonError) {
          // If JSON parsing fails, use a generic message
          throw new Error(`HTTP error! status: ${response.status}. Could not parse error response.`);
        }
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();
      setResult(data);

    } catch (err) {
      console.error("Error identifying issue:", err);
      setError(`Failed to identify issue: ${err.message}. Please try again.`);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 to-gray-800 flex flex-col items-center justify-center p-4 font-inter text-gray-100">
      <div className="bg-gray-800 rounded-2xl shadow-2xl shadow-indigo-900/50 p-8 w-full max-w-2xl text-center border border-gray-700 backdrop-blur-sm">
        <h1 className="text-4xl md:text-5xl font-extrabold text-transparent bg-clip-text bg-gradient-to-r from-teal-400 to-emerald-500 mb-4">
          AI Problem Solver
        </h1>
        <p className="text-xl text-gray-300 mb-8">
          Point. Detect. Get Solutions.
        </p>

        <div
          className={`mb-6 p-4 rounded-xl border-2 border-dashed transition-all duration-300 ease-in-out
            ${isDragging ? 'border-teal-400 bg-gray-700/50' : 'border-gray-600 bg-gray-700'}
            ${imagePreview ? 'h-auto' : 'h-48 flex flex-col justify-center items-center'}`
          }
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          {!imagePreview ? (
            <div className="text-gray-400">
              <ImageIcon size={48} className="mx-auto mb-2" />
              <p className="text-lg font-semibold mb-2">Drag & Drop your image here</p>
              <p className="text-sm mb-4">or</p>
              <label htmlFor="image-upload" className="inline-flex items-center justify-center space-x-2
                bg-teal-500 text-white py-2 px-4 rounded-full font-semibold text-sm
                hover:bg-teal-600 cursor-pointer transition duration-200 ease-in-out">
                <Upload size={20} />
                <span>Upload Image</span>
                <input
                  id="image-upload"
                  type="file"
                  accept="image/*"
                  onChange={handleFileInputChange}
                  className="hidden"
                />
              </label>
            </div>
          ) : (
            <>
              <div className="border border-gray-600 rounded-lg overflow-hidden flex justify-center items-center p-2 bg-gray-900">
                <img src={imagePreview} alt="Preview" className="max-w-full h-auto max-h-64 rounded-md object-contain" />
              </div>
              <label htmlFor="image-upload" className="mt-4 inline-flex items-center justify-center space-x-2
                bg-teal-500 text-white py-2 px-4 rounded-full font-semibold text-sm
                hover:bg-teal-600 cursor-pointer transition duration-200 ease-in-out">
                <MousePointerClick size={20} />
                <span>Change Image</span>
                <input
                  id="image-upload"
                  type="file"
                  accept="image/*"
                  onChange={handleFileInputChange}
                  className="hidden"
                />
              </label>
            </>
          )}
        </div>

        <div className="mb-6">
          <textarea
            className="w-full p-3 border border-gray-600 rounded-lg bg-gray-700 text-gray-100 placeholder-gray-400
              focus:outline-none focus:ring-2 focus:ring-emerald-500 resize-y min-h-[80px]"
            placeholder="Describe what you're seeing or concerned about (e.g., 'yellow spots on leaves', 'small crack on the side of the tool', 'mold in the corner')."
            value={userContext}
            onChange={(e) => setUserContext(e.target.value)}
          ></textarea>
        </div>

        <button
          onClick={handleSubmit}
          className="w-full bg-gradient-to-r from-teal-600 to-emerald-600 text-white py-3 px-6 rounded-lg font-semibold text-lg
            hover:from-teal-700 hover:to-emerald-700 transition duration-300 ease-in-out shadow-lg shadow-teal-500/30
            hover:shadow-xl hover:shadow-teal-500/40 focus:outline-none focus:ring-2 focus:ring-emerald-500 focus:ring-opacity-75
            flex items-center justify-center space-x-2"
          disabled={loading || !selectedImage}
        >
          {loading ? (
            <>
              <Loader2 className="animate-spin" size={20} />
              <span>Analyzing Problem...</span>
            </>
          ) : (
            <>
              <Lightbulb size={20} />
              <span>Identify & Get Solution</span>
            </>
          )}
        </button>

        {error && (
          <div className="mt-4 p-4 bg-red-900/40 rounded-lg border border-red-700 text-red-300 flex items-center space-x-2">
            <XCircle size={20} />
            <p className="font-medium">{error}</p>
          </div>
        )}

        {result && (
          <div className="mt-6 p-5 bg-green-900/40 rounded-lg border border-green-700 text-green-300 text-left">
            <h3 className="text-xl font-semibold mb-2 flex items-center space-x-2 text-green-400">
              <CheckCircle size={20} />
              <span>Identified Problem:</span>
            </h3>
            <p className="text-2xl font-bold text-white mb-3">{result.identified_problem}</p>

            <h3 className="text-xl font-semibold mb-2 flex items-center space-x-2 text-emerald-400">
              <Lightbulb size={20} />
              <span>AI Solution:</span>
            </h3>
            {/* Use ReactMarkdown to render the advice */}
            <div className="prose prose-invert max-w-none text-gray-200 leading-relaxed">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>
                {result.advice}
              </ReactMarkdown>
            </div>
          </div>
        )}
      </div>

      <footer className="mt-8 text-gray-400 text-sm">
        &copy; {new Date().getFullYear()} AI Problem Solver. Solve Issues, Instantly.
      </footer>
    </div>
  );
}

export default App;
