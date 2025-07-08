import React, { useState } from "react";
import axios from "axios";
import "./App.css"; 

function App() {
  const [video, setVideo] = useState(null);
  const [result, setResult] = useState("");
  const [loading, setLoading] = useState(false);

  const handleVideoChange = (event) => {
    const file = event.target.files[0];
    if (file) {
      setVideo(file);
      setResult(""); // Reset previous result
    }
  };

  const handleUpload = async () => {
    if (!video) {
      alert("Please upload a video first.");
      return;
    }

    setLoading(true);
    setResult("");

    const formData = new FormData();
    formData.append("video", video);

    try {
      const response = await axios.post("http://127.0.0.1:5000/predict", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      setResult(response.data.Prediction);
    } catch (error) {
      console.error("Error:", error);
      setResult("Error analyzing the video. Please try again.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="container">
      <div className="content">
        <h1>Deepfake Video Detector</h1>
        <div className="input-container">
          <input type="file" accept="video/*" onChange={handleVideoChange} />
        </div>
        <div className="button-container">
          <button onClick={handleUpload} disabled={loading}>
            {loading ? "Analyzing..." : "Upload & Detect"}
          </button>
        </div>
        {result && <p className="result">Prediction: {result}</p>}
      </div>
    </div>
  );
}

export default App;