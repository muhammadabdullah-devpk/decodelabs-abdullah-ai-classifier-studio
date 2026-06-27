/**
 * Abdullah AI Classification Studio Controller (app.js)
 * Coordinates data parsing, training cycles, graphics, and local storage state.
 */

// Application Global State
const state = {
  rawDataset: [], // Parsed rows of [x1, x2, label]
  X: [],          // Features matrix
  y: [],          // Target vector
  features: ['Feature 1', 'Feature 2'],
  X_train: [],
  X_test: [],
  y_train: [],
  y_test: [],
  scaleParams: null, // { means, stds }
  trainedModel: null,
  activeTab: 'tab-studio'
};

/* =========================================================================
   1. Built-in Dataset Generators
   ========================================================================= */

// Sample subset of Iris flower dataset (Petal Length, Petal Width)
// Setosa (Label 0), Versicolor (Label 1)
const IRIS_DATA = [
  [1.4, 0.2, 0], [1.4, 0.2, 0], [1.3, 0.2, 0], [1.5, 0.2, 0], [1.4, 0.2, 0],
  [1.7, 0.4, 0], [1.4, 0.3, 0], [1.5, 0.2, 0], [1.4, 0.2, 0], [1.5, 0.1, 0],
  [1.5, 0.2, 0], [1.6, 0.2, 0], [1.4, 0.1, 0], [1.1, 0.1, 0], [1.2, 0.2, 0],
  [1.5, 0.4, 0], [1.3, 0.4, 0], [1.4, 0.3, 0], [1.7, 0.3, 0], [1.5, 0.3, 0],
  [4.7, 1.4, 1], [4.5, 1.5, 1], [4.9, 1.5, 1], [4.0, 1.3, 1], [4.6, 1.5, 1],
  [4.5, 1.3, 1], [4.7, 1.6, 1], [3.3, 1.0, 1], [4.6, 1.3, 1], [3.9, 1.3, 1],
  [3.5, 1.0, 1], [4.2, 1.5, 1], [4.0, 1.3, 1], [4.7, 1.4, 1], [3.6, 1.3, 1],
  [4.4, 1.4, 1], [4.5, 1.5, 1], [4.1, 1.0, 1], [4.5, 1.5, 1], [3.9, 1.1, 1]
];

function generateSyntheticLinear(samples = 100, noise = 0.15) {
  const data = [];
  for (let i = 0; i < samples; i++) {
    const label = Math.random() > 0.5 ? 1 : 0;
    const offset = label === 1 ? 1.0 : -1.0;
    const x = offset + (Math.random() - 0.5) * 1.5 + (Math.random() - 0.5) * noise;
    const y = offset + (Math.random() - 0.5) * 1.5 + (Math.random() - 0.5) * noise;
    data.push([x, y, label]);
  }
  return data;
}

function generateSyntheticMoons(samples = 120, noise = 0.1) {
  const data = [];
  const halfSamples = Math.floor(samples / 2);

  // First Moon (Class 0)
  for (let i = 0; i < halfSamples; i++) {
    const theta = (i / halfSamples) * Math.PI;
    const x = Math.cos(theta) + (Math.random() - 0.5) * noise;
    const y = Math.sin(theta) + (Math.random() - 0.5) * noise;
    data.push([x, y, 0]);
  }

  // Second Moon (Class 1)
  for (let i = 0; i < halfSamples; i++) {
    const theta = (i / halfSamples) * Math.PI;
    const x = 1 - Math.cos(theta) + (Math.random() - 0.5) * noise;
    const y = 0.5 - Math.sin(theta) + (Math.random() - 0.5) * noise;
    data.push([x, y, 1]);
  }

  return data;
}

function generateSyntheticCircles(samples = 120, noise = 0.08) {
  const data = [];
  const halfSamples = Math.floor(samples / 2);

  // Outer Ring (Class 0)
  for (let i = 0; i < halfSamples; i++) {
    const theta = (i / halfSamples) * 2 * Math.PI;
    const r = 1.0;
    const x = r * Math.cos(theta) + (Math.random() - 0.5) * noise;
    const y = r * Math.sin(theta) + (Math.random() - 0.5) * noise;
    data.push([x, y, 0]);
  }

  // Inner Ring (Class 1)
  for (let i = 0; i < halfSamples; i++) {
    const theta = (i / halfSamples) * 2 * Math.PI;
    const r = 0.4;
    const x = r * Math.cos(theta) + (Math.random() - 0.5) * noise;
    const y = r * Math.sin(theta) + (Math.random() - 0.5) * noise;
    data.push([x, y, 1]);
  }

  return data;
}

/* =========================================================================
   2. DOM Helper & System Logging Functions
   ========================================================================= */
const $ = (id) => document.getElementById(id);

function logToConsole(message, type = 'info') {
  const consoleEl = $('consoleLogs');
  if (!consoleEl) return;

  const timestamp = new Date().toLocaleTimeString();
  const line = document.createElement('span');
  line.className = `terminal-line ${type}`;
  line.innerText = `[${timestamp}] ${message}`;
  
  consoleEl.appendChild(line);
  consoleEl.scrollTop = consoleEl.scrollHeight;
}

function showToast(message, type = 'success') {
  const container = $('toastContainer');
  const toast = document.createElement('div');
  toast.className = `toast ${type}`;
  toast.innerHTML = `
    <i class="${type === 'success' ? 'ri-checkbox-circle-line' : 'ri-error-warning-line'}"></i>
    <span>${message}</span>
  `;
  
  container.appendChild(toast);
  
  setTimeout(() => toast.classList.add('show'), 10);

  setTimeout(() => {
    toast.classList.remove('show');
    setTimeout(() => toast.remove(), 300);
  }, 4000);
}

/* =========================================================================
   3. File Uploading & Parsing
   ========================================================================= */
function parseCSV(text) {
  const lines = text.split(/\r?\n/).map(line => line.trim()).filter(Boolean);
  if (lines.length < 2) throw new Error('CSV has insufficient rows');

  // Check headers: support comma or semicolon
  let delimiter = ',';
  let firstLineCols = lines[0].split(',');
  if (firstLineCols.length < 3) {
    const semiCols = lines[0].split(';');
    if (semiCols.length >= 3) {
      delimiter = ';';
      firstLineCols = semiCols;
    }
  }

  const headers = firstLineCols.map(h => h.trim().replace(/^["']|["']$/g, ''));
  if (headers.length < 3) {
    throw new Error('CSV must contain at least 2 features and 1 label column. Header found: ' + lines[0]);
  }

  const rawRows = [];
  for (let i = 1; i < lines.length; i++) {
    const cols = lines[i].split(delimiter).map(c => c.trim().replace(/^["']|["']$/g, ''));
    if (cols.length < headers.length) continue; // Skip incomplete lines

    const parsedCols = cols.map(c => parseFloat(c));
    if (parsedCols.every(val => !isNaN(val))) {
      rawRows.push(parsedCols);
    }
  }

  if (rawRows.length === 0) {
    throw new Error('No valid numeric rows parsed from CSV. Delimiter: "' + delimiter + '". Total lines checked: ' + (lines.length - 1));
  }

  // Verify labels are binary (0 and 1)
  const labels = rawRows.map(row => row[row.length - 1]);
  const uniqueLabels = [...new Set(labels)];
  if (uniqueLabels.some(l => l !== 0 && l !== 1)) {
    throw new Error('Label column must contain strictly binary classes (0 and 1 only). Classes found: ' + uniqueLabels.join(', '));
  }

  return {
    headers: headers.slice(0, 2),
    data: rawRows.map(row => [row[0], row[1], row[row.length - 1]])
  };
}

/* =========================================================================
   4. UI State Synchronization & Data Loading
   ========================================================================= */
function loadDatasetIntoState(dataRows, headers = ['Feature X', 'Feature Y']) {
  state.rawDataset = dataRows;
  state.features = headers;
  state.X = dataRows.map(row => [row[0], row[1]]);
  state.y = dataRows.map(row => row[2]);

  // Update Data Studio summary views
  $('statsFeatures').innerText = '2 (2D Visualization Mode)';
  $('datasetRowStats').innerText = `${state.X.length} Rows Loaded`;

  const class0Count = state.y.filter(val => val === 0).length;
  const class1Count = state.y.filter(val => val === 1).length;
  $('statsRatio').innerText = `${class0Count} / ${class1Count}`;

  // Update tabular summary
  const tbody = $('datasetPreviewTable').querySelector('tbody');
  tbody.innerHTML = '';
  
  // Show first 8 rows for summary
  const previewRows = state.rawDataset.slice(0, 8);
  previewRows.forEach((row, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="font-family: var(--font-mono); color: var(--text-muted);">#${i + 1}</td>
      <td>${row[0].toFixed(4)}</td>
      <td>${row[1].toFixed(4)}</td>
      <td>
        <span class="legend-color" style="display:inline-block; vertical-align:middle; background: ${row[2] === 1 ? '#00f0ff' : '#ff007f'}; margin-right:6px;"></span>
        ${row[2]}
      </td>
    `;
    tbody.appendChild(tr);
  });

  if (state.rawDataset.length > 8) {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td colspan="4" style="text-align: center; color: var(--text-dim); font-size: 0.8rem;">
        ... Showing 8 of ${state.rawDataset.length} rows ...
      </td>
    `;
    tbody.appendChild(tr);
  }

  // Set predictor label details
  $('lblPredictX').innerText = `${headers[0]} Value`;
  $('lblPredictY').innerText = `${headers[1]} Value`;

  logToConsole(`Loaded dataset: ${state.X.length} samples. Columns: [${headers.join(', ')}]`, 'success');

  // Trigger empty viz charts
  VisualizationEngine.drawScatterAndBoundary($('scatterCanvas'), state.X, state.y, null, {
    xLabel: state.features[0],
    yLabel: state.features[1]
  });
}

function handleDatasetChange() {
  const source = $('datasetSource').value;
  
  if (source === 'custom') {
    $('customUploadContainer').style.display = 'block';
    return;
  } else {
    $('customUploadContainer').style.display = 'none';
  }

  let data = [];
  let headers = ['Feature X', 'Feature Y'];

  switch(source) {
    case 'iris':
      data = IRIS_DATA;
      headers = ['Petal Length (cm)', 'Petal Width (cm)'];
      break;
    case 'synthetic_linear':
      data = generateSyntheticLinear();
      headers = ['X Coordinate', 'Y Coordinate'];
      break;
    case 'synthetic_moons':
      data = generateSyntheticMoons();
      headers = ['Dimension A', 'Dimension B'];
      break;
    case 'synthetic_circular':
      data = generateSyntheticCircles();
      headers = ['Radius Axis', 'Angle Axis'];
      break;
  }

  loadDatasetIntoState(data, headers);
}

/* =========================================================================
   5. Model Training & Testing Workflow
   ========================================================================= */
function executeModelPipeline() {
  if (state.X.length === 0) {
    showToast('Please load a dataset first!', 'error');
    return;
  }

  const testRatio = parseInt($('splitRatio').value) / 100;
  const isScaleChecked = $('scaleFeatures').checked;
  const algo = $('modelAlgorithm').value;

  logToConsole(`Initiating classification training pipeline (${algo.toUpperCase()})...`);

  // Step 1: Split train and test
  const { X_train, X_test, y_train, y_test } = MLUtils.trainTestSplit(state.X, state.y, testRatio);

  state.X_train = X_train;
  state.X_test = X_test;
  state.y_train = y_train;
  state.y_test = y_test;

  logToConsole(`Data split completed: Training samples = ${X_train.length}, Testing samples = ${X_test.length}`);

  // Step 2: Feature Standardization Scaling
  let fitXTrain = X_train;
  let fitXTest = X_test;
  
  if (isScaleChecked) {
    const { X_scaled, means, stds } = MLUtils.standardize(X_train);
    fitXTrain = X_scaled;
    fitXTest = MLUtils.applyStandardize(X_test, means, stds);
    state.scaleParams = { means, stds };
    logToConsole('Feature scaling applied (Z-Score Standardisation).');
  } else {
    state.scaleParams = null;
    logToConsole('Skipped feature scaling.', 'warn');
  }

  // Step 3: Instantiate Model
  let modelInstance;
  const startTime = performance.now();

  switch(algo) {
    case 'knn':
      const k = parseInt($('param-knn-k').value);
      const metric = $('param-knn-metric').value;
      modelInstance = new KNNClassifier(k, metric);
      modelInstance.fit(fitXTrain, y_train);
      logToConsole(`Fit K-NN model parameters: neighbors (k) = ${k}, metric = ${metric}`);
      break;

    case 'dt':
      const depth = parseInt($('param-dt-depth').value);
      const minSplit = parseInt($('param-dt-split').value);
      modelInstance = new DecisionTreeClassifier(depth, minSplit);
      modelInstance.fit(fitXTrain, y_train);
      logToConsole(`Fit Decision Tree model: max_depth = ${depth}, min_split = ${minSplit}`);
      break;

    case 'lr':
      const lrRate = parseFloat($('param-lr-rate').value);
      const epochs = parseInt($('param-lr-epochs').value);
      modelInstance = new LogisticRegressionClassifier(lrRate, epochs);
      modelInstance.fit(fitXTrain, y_train, (epoch, loss) => {
        if (epoch % 50 === 0) {
          logToConsole(`Epoch ${epoch} -> Cross Entropy Loss: ${loss.toFixed(4)}`);
        }
      });
      logToConsole(`Fit Logistic Regression: lr = ${lrRate}, epochs = ${epochs}`);
      break;

    case 'nb':
      modelInstance = new NaiveBayesClassifier();
      modelInstance.fit(fitXTrain, y_train);
      logToConsole('Fit Gaussian Naive Bayes parameters calculated.');
      break;
  }

  const duration = performance.now() - startTime;
  state.trainedModel = modelInstance;

  logToConsole(`Model training completed in ${duration.toFixed(2)}ms.`, 'success');

  // Step 4: Evaluate predictions
  const testPredictions = modelInstance.predict(fitXTest);
  const metrics = MLUtils.calculateMetrics(y_test, testPredictions);

  // Update Metrics display
  $('val-accuracy').innerText = `${(metrics.accuracy * 100).toFixed(2)}%`;
  $('val-f1').innerText = metrics.f1.toFixed(3);
  $('val-precision').innerText = metrics.precision.toFixed(3);
  $('val-recall').innerText = metrics.recall.toFixed(3);

  // Update Confusion matrix grid values
  $('cm-tp').innerText = metrics.confusion.tp;
  $('cm-tn').innerText = metrics.confusion.tn;
  $('cm-fp').innerText = metrics.confusion.fp;
  $('cm-fn').innerText = metrics.confusion.fn;

  // Step 5: Calculate ROC / AUC
  const probabilities = modelInstance.predictProbs(fitXTest);
  const rocResults = MLUtils.calculateRocCurve(y_test, probabilities);

  // Render Visual charts
  const boundaryVisualWrapper = {
    predict: (X_input) => {
      let scaledInput = X_input;
      if (state.scaleParams) {
        scaledInput = MLUtils.applyStandardize(X_input, state.scaleParams.means, state.scaleParams.stds);
      }
      return state.trainedModel.predict(scaledInput);
    }
  };

  // Find index of test rows in original matrix X (for highlight circles)
  const testIndicesInX = [];
  X_test.forEach(testRow => {
    const matchIdx = state.X.findIndex(origRow => origRow[0] === testRow[0] && origRow[1] === testRow[1]);
    if (matchIdx !== -1) testIndicesInX.push(matchIdx);
  });

  VisualizationEngine.drawScatterAndBoundary(
    $('scatterCanvas'), 
    state.X, 
    state.y, 
    boundaryVisualWrapper, 
    {
      xLabel: state.features[0],
      yLabel: state.features[1],
      testIndices: testIndicesInX
    }
  );

  VisualizationEngine.drawRocCurve($('rocCanvas'), rocResults.points, rocResults.auc);

  // Active Predictor input elements
  $('predictorEmptyState').style.display = 'none';
  $('predictorForm').style.display = 'block';

  // Always show local save model config card
  $('saveModelCard').style.display = 'block';

  // Navigation shift to Visual tab
  switchTab('tab-trainer');
  showToast(`Model evaluated successfully! Accuracy: ${(metrics.accuracy * 100).toFixed(1)}%`);
  logToConsole(`Validation Metrics: Accuracy = ${(metrics.accuracy * 100).toFixed(2)}%, F1 = ${metrics.f1.toFixed(3)}, AUC = ${rocResults.auc.toFixed(3)}`, 'success');
}

/* =========================================================================
   6. Inference Sandbox Execution
   ========================================================================= */
function handleSandboxPrediction(event) {
  event.preventDefault();
  if (!state.trainedModel) {
    showToast('Train a classifier first before predicting!', 'error');
    return;
  }

  const px = parseFloat($('predictX').value);
  const py = parseFloat($('predictY').value);

  if (isNaN(px) || isNaN(py)) {
    showToast('Please insert valid numeric coordinates!', 'error');
    return;
  }

  logToConsole(`Running sandbox inference on coordinate values: (${px}, ${py})`);

  let inputFeature = [[px, py]];
  if (state.scaleParams) {
    inputFeature = MLUtils.applyStandardize(inputFeature, state.scaleParams.means, state.scaleParams.stds);
  }

  // Inference prediction and confidence
  const result = state.trainedModel.predictSingle(inputFeature[0]);
  const displayResult = $('predictionResult');
  
  if (result.label === 1) {
    displayResult.innerText = 'Class 1';
    displayResult.className = 'accent-text-cyan';
  } else {
    displayResult.innerText = 'Class 0';
    displayResult.className = 'accent-text-pink';
  }

  const confidence = result.probability * 100;
  $('predictionConfidence').innerText = `${confidence.toFixed(1)}%`;

  logToConsole(`Predicted Outcome: Class ${result.label} (Confidence: ${confidence.toFixed(1)}%)`, 'success');
}

/* =========================================================================
   7. Local Database Persistence Methods (localStorage)
   ========================================================================= */
function saveTrainedModelToServer() {
  const name = $('modelSaveName').value.trim();
  if (!name) {
    showToast('Please insert a name for your model configuration!', 'error');
    return;
  }

  if (!state.trainedModel) {
    showToast('No trained model found to save!', 'error');
    return;
  }

  const algo = $('modelAlgorithm').value;
  let configuration = { scale: $('scaleFeatures').checked };
  
  if (algo === 'knn') {
    configuration.k = parseInt($('param-knn-k').value);
    configuration.metric = $('param-knn-metric').value;
  } else if (algo === 'dt') {
    configuration.depth = parseInt($('param-dt-depth').value);
    configuration.split = parseInt($('param-dt-split').value);
  } else if (algo === 'lr') {
    configuration.lr = parseFloat($('param-lr-rate').value);
    configuration.epochs = parseInt($('param-lr-epochs').value);
  }

  const metrics = {
    accuracy: $('val-accuracy').innerText,
    f1: $('val-f1').innerText
  };

  const localModelData = {
    id: 'local-' + Date.now(),
    name,
    type: algo,
    configuration,
    metrics,
    created_at: new Date().toISOString()
  };

  const localModels = JSON.parse(localStorage.getItem('local_saved_models') || '[]');
  localModels.push(localModelData);
  localStorage.setItem('local_saved_models', JSON.stringify(localModels));

  showToast('Model saved to history catalog!');
  logToConsole(`Saved model config locally: ${name}`, 'success');
  $('modelSaveName').value = '';
  fetchModelHistory();
}

function fetchModelHistory() {
  const data = JSON.parse(localStorage.getItem('local_saved_models') || '[]');
  renderHistoryTable(data);
}

function renderHistoryTable(data) {
  const tbody = $('historyTable').querySelector('tbody');
  tbody.innerHTML = '';

  if (data.length === 0) {
    tbody.innerHTML = `
      <tr>
        <td colspan="6" class="empty-state">
          <i class="ri-inbox-line"></i>
          No saved models found yet. Train a classifier and click Save.
        </td>
      </tr>
    `;
    return;
  }

  data.forEach(model => {
    const tr = document.createElement('tr');
    const date = new Date(model.created_at).toLocaleDateString();
    tr.innerHTML = `
      <td style="font-weight: 600;">${model.name}</td>
      <td style="text-transform: uppercase; font-family: var(--font-mono); font-size: 0.8rem;">${model.type}</td>
      <td style="color: var(--accent-success); font-weight: 700;">${model.metrics.accuracy}</td>
      <td>${model.metrics.f1}</td>
      <td>${date}</td>
      <td>
        <button class="btn btn-danger btn-sm" onclick="deleteModelRecord('${model.id}')">
          <i class="ri-delete-bin-line"></i>
        </button>
      </td>
    `;
    tbody.appendChild(tr);
  });
}

window.deleteModelRecord = function(modelId) {
  if (!confirm('Are you sure you want to delete this saved model config?')) return;

  let localModels = JSON.parse(localStorage.getItem('local_saved_models') || '[]');
  localModels = localModels.filter(m => m.id !== modelId);
  localStorage.setItem('local_saved_models', JSON.stringify(localModels));
  
  showToast('Saved model configuration removed.');
  logToConsole(`Deleted model config ID: ${modelId}`);
  fetchModelHistory();
};

/* =========================================================================
   8. Navigation and Tab Bindings
   ========================================================================= */
function switchTab(tabId) {
  document.querySelectorAll('.tab-btn').forEach(btn => btn.classList.remove('active'));
  document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

  const activeBtn = document.querySelector(`.tab-btn[data-tab="${tabId}"]`);
  const activeContent = $(tabId);

  if (activeBtn && activeContent) {
    activeBtn.classList.add('active');
    activeContent.classList.add('active');
    state.activeTab = tabId;
  }
}

/* =========================================================================
   9. Initialization & Event Bindings Setup
   ========================================================================= */
document.addEventListener('DOMContentLoaded', () => {
  // Initial Dataset Load
  handleDatasetChange();

  // Load Saved History catalog immediately
  fetchModelHistory();

  // Tab bindings
  document.querySelectorAll('.tab-btn').forEach(btn => {
    btn.addEventListener('click', () => {
      switchTab(btn.getAttribute('data-tab'));
    });
  });

  // Parameter slider values sync
  const bindSlider = (sliderId, labelId, suffix = '') => {
    const slider = $(sliderId);
    const label = $(labelId);
    if (!slider || !label) return;
    slider.addEventListener('input', () => {
      label.innerText = `${slider.value}${suffix}`;
    });
  };

  bindSlider('param-knn-k', 'val-knn-k');
  bindSlider('param-dt-depth', 'val-dt-depth');
  bindSlider('param-dt-split', 'val-dt-split');
  bindSlider('param-lr-rate', 'val-lr-rate');
  bindSlider('param-lr-epochs', 'val-lr-epochs');
  bindSlider('splitRatio', 'val-split-ratio', '%');

  // Trigger dataset source shifts
  $('datasetSource').addEventListener('change', handleDatasetChange);

  // Trigger algorithm parameters display layout toggle
  $('modelAlgorithm').addEventListener('change', (e) => {
    document.querySelectorAll('.algo-params').forEach(el => el.style.display = 'none');
    $(`params-${e.target.value}`).style.display = 'block';
    logToConsole(`Algorithm toggled: ${e.target.value.toUpperCase()}`);
  });

  // Train button trigger
  $('btnTrainModel').addEventListener('click', executeModelPipeline);

  // Sandbox inference submit
  $('inferenceForm').addEventListener('submit', handleSandboxPrediction);

  // Save model click
  $('btnSaveModel').addEventListener('click', saveTrainedModelToServer);

  // Refresh history catalog button
  $('btnRefreshHistory').addEventListener('click', fetchModelHistory);
  $('tabHistoryBtn').addEventListener('click', fetchModelHistory);

  // Clear Console logs
  $('btnClearLogs').addEventListener('click', () => {
    $('consoleLogs').innerHTML = '<span class="terminal-line info">[INFO] Console cleared. System running.</span>';
  });

  // Custom File Uploader Drag and Drop interactions
  const dropZone = $('dropZone');
  const fileInput = $('fileInput');

  dropZone.addEventListener('click', () => fileInput.click());

  dropZone.addEventListener('dragover', (e) => {
    e.preventDefault();
    dropZone.classList.add('dragover');
  });

  dropZone.addEventListener('dragleave', () => {
    dropZone.classList.remove('dragover');
  });

  dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.classList.remove('dragover');
    const files = e.dataTransfer.files;
    if (files.length > 0) handleFile(files[0]);
  });

  fileInput.addEventListener('change', (e) => {
    if (e.target.files.length > 0) handleFile(e.target.files[0]);
  });

  function handleFile(file) {
    if (file.type !== 'text/csv' && !file.name.endsWith('.csv')) {
      showToast('Invalid file format. Please upload a CSV file.', 'error');
      return;
    }

    const reader = new FileReader();
    reader.onload = (event) => {
      try {
        const parsed = parseCSV(event.target.result);
        loadDatasetIntoState(parsed.data, parsed.headers);
        showToast('CSV dataset parsed successfully!');
      } catch (err) {
        showToast(err.message, 'error');
        logToConsole(`CSV parse error: ${err.message}`, 'error');
      }
    };
    reader.onerror = () => {
      showToast('Error reading file!', 'error');
    };
    reader.readAsText(file);
  }
});
