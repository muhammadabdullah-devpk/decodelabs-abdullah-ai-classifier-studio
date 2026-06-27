const express = require('express');
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const crypto = require('crypto');
const db = require('../db/db');
const router = express.Router();

const JWT_SECRET = process.env.JWT_SECRET || 'decodelabs_ai_sandbox_super_secret_key_12345';

// User Registration
router.post('/register', (req, res) => {
  const { email, password } = req.body;

  if (!email || !password) {
    return res.status(400).json({ error: 'Email and password are required' });
  }

  if (password.length < 6) {
    return res.status(400).json({ error: 'Password must be at least 6 characters' });
  }

  try {
    // Check if user exists
    const checkUser = db.prepare('SELECT id FROM users WHERE email = ?').get(email.toLowerCase());
    if (checkUser) {
      return res.status(400).json({ error: 'User with this email already exists' });
    }

    // Hash password and save
    const passwordHash = bcrypt.hashSync(password, 10);
    const userId = crypto.randomUUID();

    db.prepare('INSERT INTO users (id, email, password_hash) VALUES (?, ?, ?)').run(
      userId,
      email.toLowerCase(),
      passwordHash
    );

    // Generate JWT token
    const token = jwt.sign({ userId, email: email.toLowerCase() }, JWT_SECRET, { expiresIn: '24h' });

    return res.status(201).json({
      message: 'Registration successful',
      token,
      email: email.toLowerCase()
    });
  } catch (error) {
    console.error('Registration error:', error);
    return res.status(500).json({ error: 'Internal server error during registration' });
  }
});

// User Login
router.post('/login', (req, res) => {
  const { email, password } = req.body;

  if (!email || !password) {
    return res.status(400).json({ error: 'Email and password are required' });
  }

  try {
    const user = db.prepare('SELECT * FROM users WHERE email = ?').get(email.toLowerCase());
    if (!user || !bcrypt.compareSync(password, user.password_hash)) {
      return res.status(400).json({ error: 'Invalid email or password' });
    }

    const token = jwt.sign({ userId: user.id, email: user.email }, JWT_SECRET, { expiresIn: '24h' });

    return res.json({
      message: 'Login successful',
      token,
      email: user.email
    });
  } catch (error) {
    console.error('Login error:', error);
    return res.status(500).json({ error: 'Internal server error during login' });
  }
});

module.exports = router;
