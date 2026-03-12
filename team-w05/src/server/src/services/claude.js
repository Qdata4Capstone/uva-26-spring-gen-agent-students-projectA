const Anthropic = require('@anthropic-ai/sdk');

/**
 * Builds the system prompt from literacy_level.
 * @param {string} literacy_level - "Child" | "General Adult" | "Medical/Advanced"
 * @returns {string} System prompt
 */
function buildSystemPrompt(literacy_level) {
  return `You are a Patient Education Agent. Your role is to help patients understand medical terminology, conditions, procedures, and medications in simple, clear language. You adapt your explanations to the user's selected literacy level: ${literacy_level}. 

At Child level: use very simple words, short sentences, and relatable analogies a child would understand.
At General Adult level: use plain English with no medical jargon.
At Medical/Advanced level: use more technical language but still explain terms clearly.

You ALWAYS:
- Explain medical concepts in plain language appropriate to the literacy level
- Encourage the user to consult a qualified healthcare professional
- End every single response with: 'This information is for educational purposes only. Always consult your doctor or a qualified healthcare provider.'

You NEVER:
- Diagnose any condition
- Recommend a specific treatment or dosage for the user
- Interpret the user's personal lab results
- Give false reassurance such as 'you are probably fine' or 'you don't need to worry'
- Replace the advice of a clinician

If the user asks you to diagnose them, recommend their personal treatment, interpret their labs, or asks if they are okay or fine, you must politely refuse and redirect them to a healthcare professional.`;
}

/**
 * Calls Claude API with messages and literacy level.
 * @param {Array<{role: string, content: string}>} messages - Conversation messages
 * @param {string} literacy_level - "Child" | "General Adult" | "Medical/Advanced"
 * @returns {Promise<string>} Assistant's reply text
 */
async function callClaude(messages, literacy_level) {
  const apiKey = process.env.ANTHROPIC_API_KEY?.trim();
  if (!apiKey) {
    throw new Error('ANTHROPIC_API_KEY is not set. Add it to server/.env');
  }

  const anthropic = new Anthropic({ apiKey });

  const systemPrompt = buildSystemPrompt(literacy_level);

  const formattedMessages = messages.map((m) => ({
    role: m.role,
    content: typeof m.content === 'string'
      ? [{ type: 'text', text: m.content }]
      : m.content,
  }));

  const response = await anthropic.messages.create({
    model: 'claude-sonnet-4-6',
    max_tokens: 1024,
    system: systemPrompt,
    messages: formattedMessages,
  });

  // Extract the text content from the response
  const textContent = response.content.find(block => block.type === 'text');
  return textContent ? textContent.text : '';
}

/**
 * Summarizes PDF text using Claude API with PDF-specific instructions.
 * @param {string} pdfText - Extracted text from PDF
 * @param {string} literacy_level - "Child" | "General Adult" | "Medical/Advanced"
 * @returns {Promise<string>} Summary text
 */
async function summarizePDF(pdfText, literacy_level) {
  const apiKey = process.env.ANTHROPIC_API_KEY?.trim();
  if (!apiKey) {
    throw new Error('ANTHROPIC_API_KEY is not set. Add it to server/.env');
  }

  const anthropic = new Anthropic({ apiKey });

  const basePrompt = buildSystemPrompt(literacy_level);
  const pdfPrompt = `The user has uploaded a medical PDF document. Please summarize it in plain language appropriate for the selected literacy level. Focus on making complex medical terms easy to understand. Keep the summary clear and structured.

${basePrompt}`;

  const response = await anthropic.messages.create({
    model: 'claude-sonnet-4-6',
    max_tokens: 2048,
    system: pdfPrompt,
    messages: [
      {
        role: 'user',
        content: `Please summarize this medical document:\n\n${pdfText.substring(0, 100000)}`, // Limit to 100k chars
      },
    ],
  });

  const textContent = response.content.find(block => block.type === 'text');
  return textContent ? textContent.text : '';
}

/**
 * Calls Claude with optional PubMed articles for grounded, citable responses.
 * @param {Array<{role: string, content: string}>} messages
 * @param {string} literacy_level
 * @param {Array<{ title: string, url: string, journal: string, year: string, abstract?: string }>} articles - from PubMed
 * @returns {Promise<string>}
 */
async function callClaudeWithCitations(messages, literacy_level, articles = []) {
  const apiKey = process.env.ANTHROPIC_API_KEY?.trim();
  if (!apiKey) {
    throw new Error('ANTHROPIC_API_KEY is not set. Add it to server/.env');
  }

  const anthropic = new Anthropic({ apiKey });
  let systemPrompt = buildSystemPrompt(literacy_level);

  if (articles.length > 0) {
    const sourcesBlock = [
      '',
      'The following relevant medical literature from PubMed is provided for context. Use it to inform your answer when helpful. You may refer to "research" or "studies" without listing them in the body; we will show sources separately.',
      '',
      ...articles.slice(0, 5).map((a, i) => `[${i + 1}] ${a.title} (${a.journal}${a.year ? ', ' + a.year : ''}). ${a.abstract ? a.abstract.substring(0, 400) + '...' : ''}`),
    ].join('\n');
    systemPrompt = systemPrompt + '\n\n' + sourcesBlock;
  }

  const formattedMessages = messages.map((m) => ({
    role: m.role,
    content: typeof m.content === 'string'
      ? [{ type: 'text', text: m.content }]
      : m.content,
  }));

  const response = await anthropic.messages.create({
    model: 'claude-sonnet-4-6',
    max_tokens: 1024,
    system: systemPrompt,
    messages: formattedMessages,
  });

  const textContent = response.content.find(block => block.type === 'text');
  return textContent ? textContent.text : '';
}

module.exports = { buildSystemPrompt, callClaude, callClaudeWithCitations, summarizePDF };
