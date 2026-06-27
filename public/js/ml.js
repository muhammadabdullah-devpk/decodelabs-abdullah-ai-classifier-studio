/**
 * DecodeLabs Client-Side Supervised Machine Learning Library (ml.js)
 * Implements custom classification algorithms and metric calculations.
 */

class MLUtils {
  /**
   * Splits X and y into training and testing sets.
   * @param {Array<Array<number>>} X Features matrix
   * @param {Array<number>} y Labels array
   * @param {number} testRatio Value between 0.0 and 1.0
   * @returns {Object} { X_train, X_test, y_train, y_test }
   */
  static trainTestSplit(X, y, testRatio = 0.2) {
    const n = X.length;
    const indices = Array.from({ length: n }, (_, i) => i);
    
    // Fisher-Yates Shuffle
    for (let i = n - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    const testCount = Math.floor(n * testRatio);
    const testIndices = indices.slice(0, testCount);
    const trainIndices = indices.slice(testCount);

    return {
      X_train: trainIndices.map(i => X[i]),
      X_test: testIndices.map(i => X[i]),
      y_train: trainIndices.map(i => y[i]),
      y_test: testIndices.map(i => y[i])
    };
  }

  /**
   * Standardizes features by subtracting mean and dividing by standard deviation.
   * Returns scaled features along with scaling parameters for inference.
   * @param {Array<Array<number>>} X
   * @returns {Object} { X_scaled, means, stds }
   */
  static standardize(X) {
    const numSamples = X.length;
    const numFeatures = X[0].length;
    const means = Array(numFeatures).fill(0);
    const stds = Array(numFeatures).fill(0);

    // Calculate means
    for (let j = 0; j < numFeatures; j++) {
      let sum = 0;
      for (let i = 0; i < numSamples; i++) {
        sum += X[i][j];
      }
      means[j] = sum / numSamples;
    }

    // Calculate standard deviations
    for (let j = 0; j < numFeatures; j++) {
      let varianceSum = 0;
      for (let i = 0; i < numSamples; i++) {
        varianceSum += Math.pow(X[i][j] - means[j], 2);
      }
      stds[j] = Math.sqrt(varianceSum / numSamples) || 1e-8; // Avoid division by zero
    }

    // Scale features
    const X_scaled = X.map(row => 
      row.map((val, j) => (val - means[j]) / stds[j])
    );

    return { X_scaled, means, stds };
  }

  /**
   * Standardize custom test samples using pre-calculated training scale parameters.
   * @param {Array<Array<number>>} X 
   * @param {Array<number>} means 
   * @param {Array<number>} stds 
   */
  static applyStandardize(X, means, stds) {
    return X.map(row => 
      row.map((val, j) => (val - means[j]) / stds[j])
    );
  }

  /**
   * Calculates performance metrics for binary/multi-class predictions.
   * @param {Array<number>} y_true 
   * @param {Array<number>} y_pred 
   */
  static calculateMetrics(y_true, y_pred) {
    let correct = 0;
    y_true.forEach((val, idx) => {
      if (val === y_pred[idx]) correct++;
    });
    const accuracy = correct / y_true.length;

    // Build unique class set
    const classes = [...new Set(y_true)].sort((a,b) => a - b);
    
    // Standard Confusion Matrix for Binary (Classes 0 and 1)
    let tp = 0, tn = 0, fp = 0, fn = 0;
    if (classes.length <= 2) {
      y_true.forEach((val, idx) => {
        const predVal = y_pred[idx];
        if (val === 1 && predVal === 1) tp++;
        else if (val === 0 && predVal === 0) tn++;
        else if (val === 0 && predVal === 1) fp++;
        else if (val === 1 && predVal === 0) fn++;
      });
    }

    const precision = (tp + fp) > 0 ? tp / (tp + fp) : 0;
    const recall = (tp + fn) > 0 ? tp / (tp + fn) : 0;
    const f1 = (precision + recall) > 0 ? (2 * precision * recall) / (precision + recall) : 0;

    return {
      accuracy,
      confusion: { tp, tn, fp, fn },
      precision,
      recall,
      f1
    };
  }

  /**
   * Computes points for a Receiver Operating Characteristic (ROC) curve.
   * @param {Array<number>} y_true 
   * @param {Array<number>} probabilities Class 1 confidence probabilities
   */
  static calculateRocCurve(y_true, probabilities) {
    // Pair each truth label with its probability
    const samples = y_true.map((val, i) => ({ val, prob: probabilities[i] }));
    // Sort descending by probability
    samples.sort((a, b) => b.prob - a.prob);

    const totalPositives = y_true.filter(v => v === 1).length;
    const totalNegatives = y_true.filter(v => v === 0).length;

    const rocPoints = [{ fpr: 0, tpr: 0 }];

    if (totalPositives === 0 || totalNegatives === 0) {
      return [{ fpr: 0, tpr: 0 }, { fpr: 1, tpr: 1 }];
    }

    let tp = 0;
    let fp = 0;

    for (let i = 0; i < samples.length; i++) {
      if (samples[i].val === 1) {
        tp++;
      } else {
        fp++;
      }
      rocPoints.push({
        fpr: fp / totalNegatives,
        tpr: tp / totalPositives
      });
    }
    rocPoints.push({ fpr: 1, tpr: 1 });

    // Calculate Area Under Curve (AUC) using trapezoidal rule
    let auc = 0;
    for (let i = 1; i < rocPoints.length; i++) {
      const prev = rocPoints[i - 1];
      const curr = rocPoints[i];
      auc += ((curr.fpr - prev.fpr) * (curr.tpr + prev.tpr)) / 2;
    }

    return { points: rocPoints, auc };
  }
}

/* =========================================================================
   1. k-Nearest Neighbors (k-NN) Classifier
   ========================================================================= */
class KNNClassifier {
  constructor(k = 3, metric = 'euclidean') {
    this.k = k;
    this.metric = metric;
    this.X_train = [];
    this.y_train = [];
  }

  fit(X, y) {
    this.X_train = X;
    this.y_train = y;
  }

  calculateDistance(x1, x2) {
    let sum = 0;
    const length = Math.min(x1.length, x2.length);
    if (this.metric === 'manhattan') {
      for (let i = 0; i < length; i++) {
        sum += Math.abs(x1[i] - x2[i]);
      }
      return sum;
    } else { // Euclidean by default
      for (let i = 0; i < length; i++) {
        sum += Math.pow(x1[i] - x2[i], 2);
      }
      return Math.sqrt(sum);
    }
  }

  predictSingle(x) {
    // Map index and distance values
    const distances = this.X_train.map((trainRow, index) => ({
      index,
      dist: this.calculateDistance(trainRow, x)
    }));

    // Sort ascending
    distances.sort((a, b) => a.dist - b.dist);

    // Get K nearest neighbors
    const kNeighbors = distances.slice(0, this.k);
    
    // Count label frequencies
    const labelCounts = {};
    kNeighbors.forEach(n => {
      const label = this.y_train[n.index];
      labelCounts[label] = (labelCounts[label] || 0) + 1;
    });

    // Find label with highest vote
    let bestLabel = null;
    let maxCount = -1;
    for (const label in labelCounts) {
      if (labelCounts[label] > maxCount) {
        maxCount = labelCounts[label];
        bestLabel = parseInt(label);
      }
    }
    
    // Estimate pseudo probability based on vote ratio
    const probability = maxCount / this.k;

    return { label: bestLabel, probability };
  }

  predict(X) {
    return X.map(x => this.predictSingle(x).label);
  }

  predictProbs(X) {
    return X.map(x => {
      const res = this.predictSingle(x);
      return res.label === 1 ? res.probability : 1 - res.probability;
    });
  }
}

/* =========================================================================
   2. Decision Tree Classifier
   ========================================================================= */
class DecisionTreeNode {
  constructor(feature = null, threshold = null, left = null, right = null, value = null) {
    this.feature = feature;     // Feature index split on
    this.threshold = threshold; // Value threshold split on
    this.left = left;           // Left child node
    this.right = right;         // Right child node
    this.value = value;         // Prediction value if leaf node
  }

  isLeaf() {
    return this.value !== null;
  }
}

class DecisionTreeClassifier {
  constructor(maxDepth = 5, minSamplesSplit = 2) {
    this.maxDepth = maxDepth;
    this.minSamplesSplit = minSamplesSplit;
    this.root = null;
  }

  fit(X, y) {
    this.root = this.buildTree(X, y, 0);
  }

  entropy(y) {
    const counts = {};
    y.forEach(val => {
      counts[val] = (counts[val] || 0) + 1;
    });
    let ent = 0;
    const len = y.length;
    for (const key in counts) {
      const p = counts[key] / len;
      ent -= p * Math.log2(p);
    }
    return ent;
  }

  splitData(X, y, featureIndex, threshold) {
    const leftX = [], leftY = [], rightX = [], rightY = [];
    for (let i = 0; i < X.length; i++) {
      if (X[i][featureIndex] <= threshold) {
        leftX.push(X[i]);
        leftY.push(y[i]);
      } else {
        rightX.push(X[i]);
        rightY.push(y[i]);
      }
    }
    return { leftX, leftY, rightX, rightY };
  }

  findBestSplit(X, y) {
    let bestGain = -1;
    let bestFeature = null;
    let bestThreshold = null;
    const currentEntropy = this.entropy(y);
    const numFeatures = X[0].length;

    for (let j = 0; j < numFeatures; j++) {
      // Find all unique values in feature column j
      const uniqueValues = [...new Set(X.map(row => row[j]))].sort((a,b) => a - b);
      
      // Calculate splitting points
      for (let i = 0; i < uniqueValues.length - 1; i++) {
        const threshold = (uniqueValues[i] + uniqueValues[i + 1]) / 2;
        const { leftY, rightY } = this.splitData(X, y, j, threshold);

        if (leftY.length === 0 || rightY.length === 0) continue;

        // Calculate Information Gain
        const wLeft = leftY.length / y.length;
        const wRight = rightY.length / y.length;
        const gain = currentEntropy - (wLeft * this.entropy(leftY) + wRight * this.entropy(rightY));

        if (gain > bestGain) {
          bestGain = gain;
          bestFeature = j;
          bestThreshold = threshold;
        }
      }
    }

    return { feature: bestFeature, threshold: bestThreshold, gain: bestGain };
  }

  majorityVote(y) {
    const counts = {};
    let maxCount = -1;
    let bestVal = null;
    y.forEach(val => {
      counts[val] = (counts[val] || 0) + 1;
      if (counts[val] > maxCount) {
        maxCount = counts[val];
        bestVal = val;
      }
    });
    return bestVal;
  }

  buildTree(X, y, depth) {
    const numSamples = X.length;
    const uniqueLabels = [...new Set(y)];

    // Base cases
    if (uniqueLabels.length === 1) {
      return new DecisionTreeNode(null, null, null, null, uniqueLabels[0]);
    }
    if (depth >= this.maxDepth || numSamples < this.minSamplesSplit) {
      return new DecisionTreeNode(null, null, null, null, this.majorityVote(y));
    }

    const { feature, threshold, gain } = this.findBestSplit(X, y);

    // If no gain, return leaf
    if (gain <= 0 || feature === null) {
      return new DecisionTreeNode(null, null, null, null, this.majorityVote(y));
    }

    const { leftX, leftY, rightX, rightY } = this.splitData(X, y, feature, threshold);
    const leftChild = this.buildTree(leftX, leftY, depth + 1);
    const rightChild = this.buildTree(rightX, rightY, depth + 1);

    return new DecisionTreeNode(feature, threshold, leftChild, rightChild);
  }

  predictSingle(x, node = this.root) {
    if (node.isLeaf()) {
      return { label: node.value, probability: 1.0 };
    }
    if (x[node.feature] <= node.threshold) {
      return this.predictSingle(x, node.left);
    } else {
      return this.predictSingle(x, node.right);
    }
  }

  predict(X) {
    return X.map(x => this.predictSingle(x).label);
  }

  predictProbs(X) {
    // Simple decision tree binary output proxy
    return X.map(x => {
      const res = this.predictSingle(x);
      return res.label === 1 ? 0.95 : 0.05;
    });
  }
}

/* =========================================================================
   3. Logistic Regression Classifier (Stochastic Gradient Descent)
   ========================================================================= */
class LogisticRegressionClassifier {
  constructor(learningRate = 0.1, epochs = 100) {
    this.lr = learningRate;
    this.epochs = epochs;
    this.weights = [];
    this.bias = 0;
  }

  sigmoid(z) {
    return 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, z)))); // Clamp to avoid overflow
  }

  fit(X, y, progressCallback = null) {
    const numSamples = X.length;
    const numFeatures = X[0].length;
    
    this.weights = Array(numFeatures).fill(0);
    this.bias = 0;

    for (let epoch = 0; epoch < this.epochs; epoch++) {
      let loss = 0;
      for (let i = 0; i < numSamples; i++) {
        const x = X[i];
        const label = y[i];

        // Linear combinations
        let linearSum = this.bias;
        for (let j = 0; j < numFeatures; j++) {
          linearSum += x[j] * this.weights[j];
        }

        const prediction = this.sigmoid(linearSum);
        const error = prediction - label;
        
        // Calculate Cross Entropy Loss sum
        loss += -label * Math.log(prediction + 1e-15) - (1 - label) * Math.log(1 - prediction + 1e-15);

        // Update bias & weights
        this.bias -= this.lr * error;
        for (let j = 0; j < numFeatures; j++) {
          this.weights[j] -= this.lr * error * x[j];
        }
      }
      
      if (progressCallback && epoch % 10 === 0) {
        progressCallback(epoch, loss / numSamples);
      }
    }
  }

  predictSingle(x) {
    let linearSum = this.bias;
    for (let j = 0; j < this.weights.length; j++) {
      linearSum += x[j] * this.weights[j];
    }
    const prob = this.sigmoid(linearSum);
    return { label: prob >= 0.5 ? 1 : 0, probability: prob };
  }

  predict(X) {
    return X.map(x => this.predictSingle(x).label);
  }

  predictProbs(X) {
    return X.map(x => this.predictSingle(x).probability);
  }
}

/* =========================================================================
   4. Naive Bayes Classifier (Gaussian Naive Bayes)
   ========================================================================= */
class NaiveBayesClassifier {
  constructor() {
    this.classes = [];
    this.priors = {};
    this.means = {};
    this.variances = {};
  }

  fit(X, y) {
    const numSamples = X.length;
    this.classes = [...new Set(y)];
    
    this.classes.forEach(c => {
      // Filter rows belonging to class c
      const classRows = X.filter((_, i) => y[i] === c);
      this.priors[c] = classRows.length / numSamples;

      this.means[c] = [];
      this.variances[c] = [];

      const numFeatures = X[0].length;
      for (let j = 0; j < numFeatures; j++) {
        const featureVals = classRows.map(row => row[j]);
        
        // Compute mean
        const mean = featureVals.reduce((a, b) => a + b, 0) / featureVals.length;
        this.means[c].push(mean);

        // Compute variance
        const varianceSum = featureVals.reduce((a, b) => a + Math.pow(b - mean, 2), 0);
        this.variances[c].push((varianceSum / featureVals.length) || 1e-8); // Add epsilon
      }
    });
  }

  calculateProbabilityDensity(val, mean, variance) {
    const exponent = Math.exp(-Math.pow(val - mean, 2) / (2 * variance));
    return (1 / Math.sqrt(2 * Math.PI * variance)) * exponent;
  }

  predictSingle(x) {
    const scores = {};
    
    this.classes.forEach(c => {
      let score = Math.log(this.priors[c]);
      for (let j = 0; j < x.length; j++) {
        const density = this.calculateProbabilityDensity(x[j], this.means[c][j], this.variances[c][j]);
        score += Math.log(density + 1e-15); // Add log likelihood
      }
      scores[c] = score;
    });

    // Softmax-like probability estimation on scores to return prediction scores
    const scoreVals = Object.values(scores);
    const maxScore = Math.max(...scoreVals);
    const expScores = scoreVals.map(s => Math.exp(s - maxScore));
    const sumExp = expScores.reduce((a, b) => a + b, 0);
    const probs = expScores.map(val => val / sumExp);

    let bestClass = null;
    let bestScore = -Infinity;
    
    this.classes.forEach(c => {
      if (scores[c] > bestScore) {
        bestScore = scores[c];
        bestClass = c;
      }
    });

    // Probability estimation index map
    const classIndex = this.classes.indexOf(bestClass);
    const probability = probs[classIndex] || 1.0;

    return { label: bestClass, probability };
  }

  predict(X) {
    return X.map(x => this.predictSingle(x).label);
  }

  predictProbs(X) {
    return X.map(x => {
      const res = this.predictSingle(x);
      return res.label === 1 ? res.probability : 1 - res.probability;
    });
  }
}
