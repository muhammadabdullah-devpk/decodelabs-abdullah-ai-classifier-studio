const express = require('express');
const crypto = require('crypto');
const db = require('../db/db');
const authMiddleware = require('../middleware/auth');
const router = express.Router();

// Apply auth middleware to all model routes
router.use(authMiddleware);

// Get all models for the logged-in user
router.get('/', (req, res) => {
  try {
    const models = db.prepare('SELECT id, name, type, metrics, created_at FROM models WHERE user_id = ? ORDER BY created_at DESC').all(req.userId);
    // Parse metrics for display list
    const parsedModels = models.map(m => ({
      ...m,
      metrics: JSON.parse(m.metrics)
    }));
    return res.json(parsedModels);
  } catch (error) {
    console.error('Fetch models error:', error);
    return res.status(500).json({ error: 'Failed to fetch models' });
  }
});

// Get a single saved model configuration in detail
router.get('/:id', (req, res) => {
  const { id } = req.params;
  try {
    const model = db.prepare('SELECT * FROM models WHERE id = ? AND user_id = ?').get(id, req.userId);
    if (!model) {
      return res.status(404).json({ error: 'Model not found' });
    }
    return res.json({
      id: model.id,
      name: model.name,
      type: model.type,
      configuration: JSON.parse(model.configuration),
      metrics: JSON.parse(model.metrics),
      created_at: model.created_at
    });
  } catch (error) {
    console.error('Fetch model detail error:', error);
    return res.status(500).json({ error: 'Failed to fetch model details' });
  }
});

// Save a new trained model configuration
router.post('/', (req, res) => {
  const { name, type, configuration, metrics } = req.body;

  if (!name || !type || !configuration || !metrics) {
    return res.status(400).json({ error: 'Model name, type, configuration and metrics are required' });
  }

  try {
    const modelId = crypto.randomUUID();
    const configStr = JSON.stringify(configuration);
    const metricsStr = JSON.stringify(metrics);

    db.prepare('INSERT INTO models (id, user_id, name, type, configuration, metrics) VALUES (?, ?, ?, ?, ?, ?)').run(
      modelId,
      req.userId,
      name,
      type,
      configStr,
      metricsStr
    );

    return res.status(201).json({
      message: 'Model saved successfully',
      id: modelId,
      name
    });
  } catch (error) {
    console.error('Save model error:', error);
    return res.status(500).json({ error: 'Failed to save model' });
  }
});

// Delete a saved model
router.delete('/:id', (req, res) => {
  const { id } = req.params;
  try {
    const result = db.prepare('DELETE FROM models WHERE id = ? AND user_id = ?').run(id, req.userId);
    if (result.changes === 0) {
      return res.status(404).json({ error: 'Model not found or unauthorized' });
    }
    return res.json({ message: 'Model deleted successfully' });
  } catch (error) {
    console.error('Delete model error:', error);
    return res.status(500).json({ error: 'Failed to delete model' });
  }
});

module.exports = router;
