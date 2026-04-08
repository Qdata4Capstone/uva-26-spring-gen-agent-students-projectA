const DISCLAIMER = "\n\n⚠️ This information is for educational purposes only and not medical advice. Consult a licensed healthcare provider.";

function enforceSafety(response, triageStatus) {
    if (triageStatus === "EMERGENCY") {
        return "🚨 Your symptoms may indicate a medical emergency. Call 911 or go to the nearest emergency room immediately.";
    }

    const bannedPhrases = ["you have", "this means you have"];

    bannedPhrases.forEach(phrase => {
        response = response.replaceAll(phrase, "it may be associated with");
    });

    return response + DISCLAIMER;
}

module.exports = { enforceSafety };