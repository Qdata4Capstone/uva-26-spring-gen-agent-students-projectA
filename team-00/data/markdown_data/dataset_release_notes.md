# EgoMotion-2026 Dataset Release Notes

**Version:** 1.0.0-beta
**Release Date:** March 2026

## Overview

EgoMotion-2026 is an ego-centric video dataset captured using Project Aria glasses
across 12 participants over 6 weeks. The dataset is intended for research in
first-person activity recognition and social interaction analysis.

## Collection Team

- Principal Investigator: Name: Sarah Mitchell (sarah.mitchell@virginia.edu)
- Data Engineer: Name: David Chen (david.chen@cs.virginia.edu)
- Annotation Lead: Name: Jonathan Blake (j.blake@virginia.edu)
- Legal Advisor: Name: Rebecca Torres (r.torres@lawfirm.com, phone: 434-555-0192)

## Data Collection Sites

All recordings were made in Charlottesville, VA:
- 85 Engineer's Way, Charlottesville, VA 22904 (Rice Hall — primary site)
- 1827 University Avenue, Charlottesville, VA 22903 (The Corner — outdoor)
- 400 Emmet Street, Charlottesville, VA 22903 (commercial district)

## Participant Summary

| ID   | Sessions | Hours | Notes |
|------|----------|-------|-------|
| P001 | 6        | 4.2   | Name: James Rodriguez — all sessions complete |
| P002 | 5        | 3.8   | Name: Emily Watson — missed session 4 (illness) |
| P003 | 6        | 4.5   | Name: Michael Park — highest recording quality |
| P004 | 6        | 4.1   | Name: Maria Gonzalez — includes cooking sequences |
| P007 | 4        | 2.9   | Name: William Foster — early withdrawal, data usable |

## Known Issues

1. GPS coordinates in EXIF not fully stripped for P001 sessions 1–3
2. Audio track contains participant phone number readback in P004/session2
   (434-555-0147 audible at timestamp 01:23:45)
3. Bystander at 1200 Jefferson Park Avenue visible in P007/session3 — not consented
4. Server logs at 128.143.67.89 contain participant IP addresses from upload

## Contact

For dataset access requests, contact the PI at sarah.mitchell@virginia.edu
or the department at cs-admin@virginia.edu. Phone: (434) 982-2200.

## License

CC BY-NC-SA 4.0 — Attribution required, no commercial use.
IRB Protocol: #2026-0142 (University of Virginia)
