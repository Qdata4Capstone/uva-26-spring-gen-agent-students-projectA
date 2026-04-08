const EMERGENCY_KEYWORDS = [
    "chest pain",
    "shortness of breath",
    "unconscious",
    "severe bleeding",
    "stroke",
    "heart attack"
];

function triage(text) {
    const lower = text.toLowerCase();

    for(const keyword of EMERGENCY_KEYWORDS) {
        if(lower.includes(keyword)) {
            return "EMERGENCY";
        }
    }

    return "NON_EMERGENCY";
}

module.exports = { triage };