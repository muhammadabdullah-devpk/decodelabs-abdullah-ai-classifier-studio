/**
 * DecodeLabs Sandbox Unit & Integration Tests (test.js)
 * Verifies machine learning math logic and backend REST API routing.
 */

const test = require('node:test');
const assert = require('node:assert');
const path = require('path');
const fs = require('fs');

// Set test database path before loading database
const testDbPath = path.join(__dirname, 'db/test_database.db');
if (fs.existsSync(testDbPath)) {
  fs.unlinkSync(testDbPath);
}
process.env.DB_PATH = 'db/test_database.db';
process.env.JWT_SECRET = 'test_secret_key_987654321';

const app = require('./index');
const db = require('./db/db');
const authMiddleware = require('./middleware/auth');

// We will also test client-side ML classifiers here in Node.js
// by loading the code from ml.js
const mlJsPath = path.join(__dirname, '../public/js/ml.js');
const mlJsCode = fs.readFileSync(mlJsPath, 'utf8');

// Simple wrapper evaluation context to load class definitions
const evalContext = {};
const fn = new Function('context', mlJsCode + '\n' +
  'context.MLUtils = MLUtils;\n' +
  'context.KNNClassifier = KNNClassifier;\n' +
  'context.DecisionTreeClassifier = DecisionTreeClassifier;\n' +
  'context.LogisticRegressionClassifier = LogisticRegressionClassifier;\n' +
  'context.NaiveBayesClassifier = NaiveBayesClassifier;\n'
);
fn(evalContext);

const { 
  MLUtils, 
  KNNClassifier, 
  DecisionTreeClassifier, 
  LogisticRegressionClassifier, 
  NaiveBayesClassifier 
} = evalContext;

/* =========================================================================
   1. Machine Learning Engine Mathematical Unit Tests
   ========================================================================= */

test('MLUtils.trainTestSplit splits correct ratios', () => {
  const X = Array.from({ length: 100 }, (_, i) => [i, i * 2]);
  const y = Array.from({ length: 100 }, (_, i) => i % 2);

  const { X_train, X_test, y_train, y_test } = MLUtils.trainTestSplit(X, y, 0.2);

  assert.strictEqual(X_train.length, 80);
  assert.strictEqual(X_test.length, 20);
  assert.strictEqual(y_train.length, 80);
  assert.strictEqual(y_test.length, 20);
  assert.strictEqual(X_train[0].length, 2);
});

test('MLUtils.standardize scales data to zero-mean and unit variance', () => {
  const X = [[1, 10], [2, 20], [3, 30], [4, 40], [5, 50]];
  const { X_scaled, means, stds } = MLUtils.standardize(X);

  // Assert means are correct
  assert.ok(Math.abs(means[0] - 3) < 1e-7);
  assert.ok(Math.abs(means[1] - 30) < 1e-7);

  // Assert standard deviations are correct
  // variance = (2^2 + 1^2 + 0 + 1^2 + 2^2)/5 = 10/5 = 2. std = sqrt(2) = 1.414...
  assert.ok(Math.abs(stds[0] - Math.sqrt(2)) < 1e-7);

  // Scaled values check: first element scaled should be (1 - 3)/sqrt(2) = -2/1.414 = -1.414
  assert.ok(Math.abs(X_scaled[0][0] - (-Math.sqrt(2))) < 1e-7);
});

test('KNN Classifier fits and predicts correctly', () => {
  const X = [[1, 1], [1.5, 1.5], [5, 5], [5.5, 5.5]];
  const y = [0, 0, 1, 1];

  const knn = new KNNClassifier(3, 'euclidean');
  knn.fit(X, y);

  const predictions = knn.predict([[1.2, 1.2], [5.2, 5.2]]);
  assert.deepStrictEqual(predictions, [0, 1]);
});

test('Decision Tree fits and splits correctly on simple separation', () => {
  const X = [[1, 2], [1, 3], [5, 2], [5, 3]];
  const y = [0, 0, 1, 1]; // Linearly split on feature index 0 with threshold around 3

  const dt = new DecisionTreeClassifier(3, 2);
  dt.fit(X, y);

  const predictions = dt.predict([[1.5, 2.5], [4.5, 2.5]]);
  assert.deepStrictEqual(predictions, [0, 1]);
});

test('Naive Bayes calculates correct priors and predicts classes', () => {
  const X = [[0.5, 0.5], [0.6, 0.6], [8.0, 8.0], [9.0, 9.0]];
  const y = [0, 0, 1, 1];

  const nb = new NaiveBayesClassifier();
  nb.fit(X, y);

  const predictions = nb.predict([[0.55, 0.55], [8.5, 8.5]]);
  assert.deepStrictEqual(predictions, [0, 1]);
});

/* =========================================================================
   2. API Endpoints Integration Tests
   ========================================================================= */
const testUser = {
  email: `intern_${Date.now()}@decodelabs.tech`,
  password: 'securedPassword123'
};
let authToken = '';

test('User registration returns 201 and token', async () => {
  const response = await fetch('http://localhost:3000/api/auth/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(testUser)
  });

  const body = await response.json();
  assert.strictEqual(response.status, 201);
  assert.ok(body.token);
  assert.strictEqual(body.email, testUser.email);
});

test('User login returns 200 and token', async () => {
  const response = await fetch('http://localhost:3000/api/auth/login', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(testUser)
  });

  const body = await response.json();
  assert.strictEqual(response.status, 200);
  assert.ok(body.token);
  authToken = body.token; // Save for subsequent tests
});

test('Accessing secure models route without token returns 401', async () => {
  const response = await fetch('http://localhost:3000/api/models');
  assert.strictEqual(response.status, 401);
});

test('Save and list trained model returns success', async () => {
  const testModel = {
    name: 'Test KNN Sandbox Model',
    type: 'knn',
    configuration: { k: 3, metric: 'euclidean', scale: true },
    metrics: { accuracy: '95.00%', f1: '0.945' }
  };

  // 1. Post Model
  const saveResponse = await fetch('http://localhost:3000/api/models', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': `Bearer ${authToken}`
    },
    body: JSON.stringify(testModel)
  });
  
  const saveBody = await saveResponse.json();
  assert.strictEqual(saveResponse.status, 201);
  assert.ok(saveBody.id);

  // 2. Fetch Models
  const listResponse = await fetch('http://localhost:3000/api/models', {
    headers: { 'Authorization': `Bearer ${authToken}` }
  });
  
  const listBody = await listResponse.json();
  assert.strictEqual(listResponse.status, 200);
  assert.ok(listBody.length > 0);
  assert.strictEqual(listBody[0].name, testModel.name);
  assert.strictEqual(listBody[0].type, testModel.type);
});

// Teardown DB connection and remove testing file
test.after(() => {
  db.close();
  try {
    if (fs.existsSync(testDbPath)) {
      // Small timeout to release file system lock before deleting file
      setTimeout(() => {
        fs.unlinkSync(testDbPath);
        console.log('Teardown: Removed test database file.');
      }, 500);
    }
  } catch (err) {
    console.error('Error during cleanup:', err.message);
  }
  
  // Close HTTP server connection
  process.exit(0);
});
