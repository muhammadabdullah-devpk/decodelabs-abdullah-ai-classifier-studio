const express = require('express');
const cors = require('cors');
const path = require('path');
require('dotenv').config();

const authRouter = require('./routes/authRouter');
const dataRouter = require('./routes/dataRouter');
const modelRouter = require('./routes/modelRouter');

const app = express();
const PORT = process.env.PORT || 3000;

// Standard Middlewares
app.use(cors());
app.use(express.json({ limit: '10mb' })); // Support larger datasets uploads
app.use(express.urlencoded({ extended: true }));

// Serve Static Frontend Assets
app.use(express.static(path.join(__dirname, '../public')));

// API Routes
app.use('/api/auth', authRouter);
app.use('/api/datasets', dataRouter);
app.use('/api/models', modelRouter);

// Fallback to index.html for Single Page Application
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, '../public/index.html'));
});

// Start Server
app.listen(PORT, () => {
  console.log(`==================================================`);
  console.log(`DecodeLabs AI Classification Sandbox Server is Active`);
  console.log(`Address: http://localhost:${PORT}`);
  console.log(`Time: ${new Date().toISOString()}`);
  console.log(`==================================================`);
});

module.exports = app; // Export for unit tests
