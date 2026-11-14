const form = document.getElementById("predictForm");
const result = document.getElementById("result");

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  result.className = "result";
  result.innerHTML = "<h2>Analyzing...</h2><p>Please wait.</p>";

  try {
    const response = await fetch("/predict", {
      method: "POST",
      body: new FormData(form)
    });

    const data = await response.json();

    if (!response.ok) {
      throw new Error(data.error || "Prediction failed.");
    }

    const isFraud = data.class === 1;
    const probability = data.probability === null
      ? "N/A"
      : (data.probability * 100).toFixed(2) + "%";

    result.className = "result " + (isFraud ? "fraud" : "safe");
    result.innerHTML = `
      <div class="result-icon">${isFraud ? "⚠️" : "✓"}</div>
      <div>
        <p class="eyebrow">PREDICTION RESULT</p>
        <h2>${data.prediction.toUpperCase()} TRANSACTION</h2>
        <p><b>Risk:</b> ${data.risk}</p>
        <p><b>Fraud probability:</b> ${probability}</p>
        <p><b>Model:</b> ${data.model}</p>
      </div>
    `;
  } catch (error) {
    result.className = "result error";
    result.innerHTML = `<h2>Something went wrong</h2><p>${error.message}</p>`;
  }
});
