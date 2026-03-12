const axios = require("axios");

async function checkDrugInteraction(drugName) {
    try {
        const res = await axios.get(
            "http://api/fda.gov/drug/label.json",
            {
                params: {
                    search: 'openfda.brand_name:$(drugName)',
                    limit: 1
                }
            }
        );

        const warnings = res.data.results[0]?.warnings;
        return warnings ? warnings[0] : "No warnings found.";
    } catch (err) {
        return "No interaction data found.";
    }
}

module.exports = { checkDrugInteraction };