const { XMLParser } = require("fast-xml-parser");

const BASE = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils";

function getEmail() {
  return process.env.NCBI_EMAIL?.trim();
}
function getApiKey() {
  return process.env.NCBI_API_KEY?.trim();
}

const parser = new XMLParser({
  ignoreAttributes: false,
  attributeNamePrefix: "",
});

function simplifyQuery(q) {
  return q
    .toLowerCase()
    .trim()
    .replace(/\?+$/, "")
    .replace(/^what is\s+/i, "")
    .replace(/^what are\s+/i, "")
    .replace(/^define\s+/i, "")
    .replace(/^explain\s+/i, "")
    .replace(/^tell me about\s+/i, "")
    .replace(/^what are the symptoms of\s+/i, "")
    .replace(/^what is the treatment for\s+/i, "")
    .replace(/^how do you treat\s+/i, "")
    .replace(/^what causes\s+/i, "")
    .trim();
}

function buildQueryVariants(userQuery) {
  const simplified = simplifyQuery(userQuery);
  const variants = [];

  if (simplified) {
    variants.push(`"${simplified}"[MeSH Terms] OR "${simplified}"[Title/Abstract]`);
    variants.push(`"${simplified}"[Title/Abstract]`);
  }

  variants.push(`"${userQuery.trim()}"[Title/Abstract]`);

  const keywords = simplified
    .split(/\s+/)
    .filter(Boolean)
    .filter(
      (w) =>
        ![
          "the","a","an","of","for","and","or","to","is","are",
          "do","you","what","how","about","tell","me"
        ].includes(w)
    );

  if (keywords.length) {
    variants.push(keywords.map(k => `"${k}"[Title/Abstract]`).join(" AND "));
    variants.push(keywords.join(" "));
  }

  return [...new Set(variants)];
}

async function runESearch(term, maxResults = 5) {
  const EMAIL = getEmail();
  const API_KEY = getApiKey();
  if (!EMAIL) return [];
  const url =
    `${BASE}/esearch.fcgi?db=pubmed&retmode=json` +
    `&term=${encodeURIComponent(term)}` +
    `&retmax=${maxResults}` +
    `&sort=relevance` +
    `&email=${encodeURIComponent(EMAIL)}` +
    (API_KEY ? `&api_key=${encodeURIComponent(API_KEY)}` : "");

  console.log("PubMed query:", term);

  const res = await fetch(url);

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`PubMed esearch failed (${res.status}): ${text}`);
  }

  const data = await res.json();
  return data?.esearchresult?.idlist || [];
}

async function searchPubMed(userQuery, maxResults = 5) {
  const variants = buildQueryVariants(userQuery);

  for (const term of variants) {
    const ids = await runESearch(term, maxResults);
    if (ids.length) return ids;
  }

  return [];
}

async function fetchSummary(pmids) {
  if (!pmids.length) return [];
  const EMAIL = getEmail();
  const API_KEY = getApiKey();
  if (!EMAIL) return [];
  const url =
    `${BASE}/esummary.fcgi?db=pubmed&retmode=json` +
    `&id=${pmids.join(",")}` +
    `&email=${encodeURIComponent(EMAIL)}` +
    (API_KEY ? `&api_key=${encodeURIComponent(API_KEY)}` : "");

  const res = await fetch(url);

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`PubMed esummary failed (${res.status}): ${text}`);
  }

  const data = await res.json();
  const result = data?.result || {};

  return pmids
    .map(id => {
      const a = result[id];
      if (!a) return null;

      const year = (a.pubdate || "").split(" ")[0] || "";

      return {
        pmid: id,
        title: a.title || "Untitled",
        journal: a.fulljournalname || a.source || "Unknown journal",
        year,
        url: `https://pubmed.ncbi.nlm.nih.gov/${id}/`
      };
    })
    .filter(Boolean);
}

function extractAbstracts(xmlText) {
  try {
    const json = parser.parse(xmlText);
    const raw = json?.PubmedArticleSet?.PubmedArticle;
    const articles = raw ? (Array.isArray(raw) ? raw : [raw]) : [];

    const map = {};

    for (const a of articles) {
      const pmid =
        a?.MedlineCitation?.PMID?.["#text"] ||
        a?.MedlineCitation?.PMID ||
        "";

      const abs = a?.MedlineCitation?.Article?.Abstract?.AbstractText;

      let abstract = "";

      if (typeof abs === "string") {
        abstract = abs;
      } else if (Array.isArray(abs)) {
        abstract = abs
          .map(x => (typeof x === "string" ? x : x?.["#text"] || ""))
          .join(" ");
      } else if (abs?.["#text"]) {
        abstract = abs["#text"];
      }

      if (pmid) map[String(pmid)] = abstract || "";
    }

    return map;
  } catch (err) {
    console.error("Failed parsing PubMed abstracts:", err.message);
    return {};
  }
}

async function fetchAbstracts(pmids) {
  if (!pmids.length) return {};
  const EMAIL = getEmail();
  const API_KEY = getApiKey();
  if (!EMAIL) return {};
  const url =
    `${BASE}/efetch.fcgi?db=pubmed&rettype=abstract&retmode=xml` +
    `&id=${pmids.join(",")}` +
    `&email=${encodeURIComponent(EMAIL)}` +
    (API_KEY ? `&api_key=${encodeURIComponent(API_KEY)}` : "");

  const res = await fetch(url);

  if (!res.ok) {
    const text = await res.text();
    throw new Error(`PubMed efetch failed (${res.status}): ${text}`);
  }

  const xml = await res.text();
  return extractAbstracts(xml);
}


async function searchAndFetch(userQuery, maxResults = 5) {
  const pmids = await searchPubMed(userQuery, maxResults);

  if (!pmids.length) return [];

  const summaries = await fetchSummary(pmids);
  const abstracts = await fetchAbstracts(pmids);

  return summaries.map(s => ({
    ...s,
    abstract: abstracts[s.pmid] || ""
  }));
}

module.exports = { searchAndFetch };