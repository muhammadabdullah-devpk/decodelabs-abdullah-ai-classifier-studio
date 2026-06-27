const express = require('express');
const crypto = require('crypto');
const db = require('../db/db');
const authMiddleware = require('../middleware/auth');
const router = express.Router();

// Apply auth middleware to all dataset routes
router.use(authMiddleware);

// Get all datasets for logged-in user
router.get('/', (req, res) => {
  try {
    const datasets = db.prepare('SELECT id, name, created_at FROM datasets WHERE user_id = ? ORDER BY created_at DESC').all(req.userId);
    return res.json(datasets);
  } catch (error) {
    console.error('Fetch datasets error:', error);
    return res.status(500).json({ error: 'Failed to fetch datasets' });
  }
});

// Get detailed content of a single dataset
router.get('/:id', (req, res) => {
  const { id } = req.params;
  try {
    const dataset = db.prepare('SELECT * FROM datasets WHERE id = ? AND user_id = ?').get(id, req.userId);
    if (!dataset) {
      return res.status(404).json({ error: 'Dataset not found' });
    }
    return res.json({
      id: dataset.id,
      name: dataset.name,
      content: JSON.parse(dataset.content),
      created_at: dataset.created_at
    });
  } catch (error) {
    console.error('Fetch dataset detail error:', error);
    return res.status(500).json({ error: 'Failed to fetch dataset details' });
  }
});

// Save a new dataset
router.post('/', (req, res) => {
  const { name, content } = req.body;

  if (!name || !content) {
    return res.status(400).json({ error: 'Dataset name and content are required' });
  }

  // Basic structure check for content (must be an array of objects)
  if (!Array.isArray(content) || content.length === 0) {
    return res.status(400).json({ error: 'Dataset content must be a non-empty array of objects' });
  }

  // Limit dataset rows size (e.g. max 1000 rows to keep database file manageable)
  if (content.length > 2000) {
    return res.status(400).json({ error: 'Dataset row limit exceeded (max 2000 rows allowed)' });
  }

  try {
    const datasetId = crypto.randomUUID();
    const stringifiedContent = JSON.stringify(content);

    // Limit overall size (max 5MB stringified)
    if (Buffer.byteLength(stringifiedContent, 'utf8') > 5 * 1024 * 1024) {
      return res.status(400).json({ error: 'Dataset content exceeds size limit (5MB)' });
    }

    db.prepare('INSERT INTO datasets (id, user_id, name, content) VALUES (?, ?, ?, ?)').run(
      datasetId,
      req.userId,
      name,
      stringifiedContent
    );

    return res.status(201).json({
      message: 'Dataset saved successfully',
      id: datasetId,
      name
    });
  } catch (error) {
    console.error('Save dataset error:', error);
    return res.status(500).json({ error: 'Failed to save dataset' });
  }
});

// Delete a dataset
router.delete('/:id', (req, res) => {
  const { id } = req.params;
  try {
    const result = db.prepare('DELETE FROM datasets WHERE id = ? AND user_id = ?').run(id, req.userId);
    if (result.changes === 0) {
      return res.status(404).json({ error: 'Dataset not found or unauthorized' });
    }
    return res.json({ message: 'Dataset deleted successfully' });
  } catch (error) {
    console.error('Delete dataset error:', error);
    return res.status(500).json({ error: 'Failed to delete dataset' });
  }
});

module.exports = router;
