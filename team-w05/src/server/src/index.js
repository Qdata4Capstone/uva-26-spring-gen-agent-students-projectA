const path = require('path');
const fs = require('fs');

// Try multiple .env locations
const envPaths = [
  path.join(__dirname, '..', '.env'),
  path.join(process.cwd(), '.env'),
];

let loadedPath = null;
const dotenv = require('dotenv');
for (const envPath of envPaths) {
  if (fs.existsSync(envPath)) {
    loadedPath = envPath;
    let parsed = null;
    for (const enc of ['utf8', 'utf16le']) {
      const result = dotenv.config({ path: envPath, encoding: enc });
      if (result.parsed && Object.keys(result.parsed).length > 0) {
        parsed = result.parsed;
        if (enc !== 'utf8') console.log('Loaded .env using', enc);
        break;
      }
    }
    if (!parsed || !parsed.ANTHROPIC_API_KEY) {
      // Manual fallback: read file, strip BOM, parse KEY=value lines
      for (const enc of ['utf8', 'utf16le']) {
        try {
          let content = fs.readFileSync(envPath, enc).replace(/^\uFEFF/, '');
          const manual = dotenv.parse(content);
          if (Object.keys(manual).length > 0) {
            Object.assign(process.env, manual);
            console.log('Loaded .env via manual parse');
            break;
          }
        } catch (_) {}
      }
    }
    break;
  }
}

if (!process.env.ANTHROPIC_API_KEY) {
  console.error('ERROR: ANTHROPIC_API_KEY is not set.');
  if (loadedPath) {
    console.error('Check server/.env has exactly: ANTHROPIC_API_KEY=sk-ant-...');
    console.error('No spaces around =, no quotes. Variable name must match exactly.');
  }
} else {
  console.log('ANTHROPIC_API_KEY loaded');
}
const express = require('express');
const cors = require('cors');

const app = express();
const PORT = process.env.PORT || 3001;

// Enable CORS for Vite dev server (default port 5173)
app.use(cors({
  origin: 'http://localhost:5173',
  credentials: true
}));
app.use(express.json());

const chatRoutes = require('./routes/chat');
app.use('/api', chatRoutes);

app.listen(PORT, () => {
  console.log(`Server running on http://localhost:${PORT}`);
});
