"""clinical_trials module parsing (no HTTP)."""

from app.services.clinical_trials import _study_to_record


def test_study_to_record_maps_fields():
    study = {
        "protocolSection": {
            "identificationModule": {
                "nctId": "NCT00001234",
                "briefTitle": "Example trial",
            },
            "statusModule": {"overallStatus": "RECRUITING"},
            "descriptionModule": {"briefSummary": "A short **summary** for testing."},
        }
    }
    rec = _study_to_record(study)
    assert rec is not None
    assert rec["nct_id"] == "NCT00001234"
    assert rec["title"] == "Example trial"
    assert "summary" in rec["brief_description"].lower()
    assert rec["status"] == "Recruiting"
    assert rec["recruiting"] is True
    assert rec["url"] == "https://clinicaltrials.gov/study/NCT00001234"


def test_study_to_record_non_recruiting():
    study = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT999", "briefTitle": "Closed"},
            "statusModule": {"overallStatus": "COMPLETED"},
            "descriptionModule": {},
        }
    }
    rec = _study_to_record(study)
    assert rec["status"] == "Not recruiting"
    assert rec["recruiting"] is False
