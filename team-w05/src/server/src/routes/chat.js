const express = require('express');
const router = express.Router();
const multer = require('multer');
const pdfParse = require('pdf-parse');
const { callClaude, callClaudeWithCitations, summarizePDF } = require('../services/claude');
const { searchAndFetch } = require('../services/pubmed');
const { triage } = require('../services/triage');
const { enforceSafety } = require('../services/safety');

const VALID_LITERACY_LEVELS = ['Child', 'General Adult', 'Medical/Advanced'];

// POST /api/chat — handles chat requests and calls Claude API
router.post('/chat', async (req, res) => {
  try {
    const { messages, literacy_level } = req.body;

    // Validate messages
    if (!Array.isArray(messages) || messages.length === 0) {
      return res.status(400).json({ 
        error: 'messages must be a non-empty array' 
      });
    }

    // Validate literacy_level
    if (!literacy_level || !VALID_LITERACY_LEVELS.includes(literacy_level)) {
      return res.status(400).json({ 
        error: `literacy_level must be one of: ${VALID_LITERACY_LEVELS.join(', ')}` 
      });
    }

    const lastUserMessage = [...messages].reverse().find(m => m.role === 'user');
    const query = lastUserMessage?.content?.trim() || '';

    // Emergency triage: if user describes emergency symptoms, return short directive only
    const triageStatus = triage(query);
    if (triageStatus === 'EMERGENCY') {
      const emergencyMessage = enforceSafety('', triageStatus);
      return res.json({ content: emergencyMessage, citations: [] });
    }

    // Fetch PubMed articles from the last user message (optional; failures are non-fatal)
    let articles = [];
    if (query) {
      try {
        articles = await searchAndFetch(query, 5);
      } catch (err) {
        console.warn('PubMed fetch failed (continuing without citations):', err.message);
      }
    }

    const assistantReply = articles.length > 0
      ? await callClaudeWithCitations(messages, literacy_level, articles)
      : await callClaude(messages, literacy_level);

    const citations = articles.map(a => ({
      title: a.title,
      url: a.url,
      journal: a.journal,
      year: a.year,
    }));

    const safeReply = enforceSafety(assistantReply, triageStatus);
    res.json({ content: safeReply, citations });
  } catch (error) {
    console.error('Error calling Claude API:', error);
    res.status(500).json({
      error: 'Failed to get response from AI service',
      details: error.message,
    });
  }
});

// Configure multer for file uploads
const upload = multer({
  storage: multer.memoryStorage(),
  limits: {
    fileSize: 10 * 1024 * 1024, // 10MB limit
  },
  fileFilter: (req, file, cb) => {
    if (file.mimetype === 'application/pdf') {
      cb(null, true);
    } else {
      cb(new Error('Only PDF files are allowed'), false);
    }
  },
});

// POST /api/summarize — handles PDF upload and summarization
router.post('/summarize', upload.single('pdf'), async (req, res) => {
  try {
    if (!req.file) {
      return res.status(400).json({ error: 'No PDF file uploaded' });
    }

    const { literacy_level } = req.body;

    // Validate literacy_level
    if (!literacy_level || !VALID_LITERACY_LEVELS.includes(literacy_level)) {
      return res.status(400).json({
        error: `literacy_level must be one of: ${VALID_LITERACY_LEVELS.join(', ')}`,
      });
    }

    // Extract text from PDF
    let pdfText;
    try {
      const pdfData = await pdfParse(req.file.buffer);
      pdfText = pdfData.text;
      if (!pdfText || pdfText.trim().length === 0) {
        return res.status(400).json({
          error: 'PDF appears to be empty or contains no extractable text',
        });
      }
    } catch (pdfError) {
      console.error('Error parsing PDF:', pdfError);
      return res.status(400).json({
        error: 'Failed to extract text from PDF. Please ensure it is a valid PDF file.',
      });
    }

    // Summarize using Claude
    const summary = await summarizePDF(pdfText, literacy_level);

    res.json({ content: summary });
  } catch (error) {
    console.error('Error summarizing PDF:', error);
    res.status(500).json({
      error: 'Failed to summarize PDF',
      details: error.message,
    });
  }
});

module.exports = router;
