# Data Breach Incident Report

**Report ID:** IR-2026-0042
**Date Filed:** February 12, 2026
**Severity:** Medium

## Summary

An unencrypted USB drive containing participant video data was found in the
common area of Rice Hall on February 11, 2026. The drive was recovered by
Name: Karen Phillips (building staff) and returned to the lab.

## Affected Data

The drive contained raw video recordings from 3 participants:
- P001 (Name: James Rodriguez, SSN: 123-45-6789)
- P004 (Name: Maria Gonzalez, email: maria.gonzalez@gmail.com)
- P007 (Name: William Foster, phone: 434-555-0276)

Video files included facial data and location metadata showing recordings
at 1200 Jefferson Park Avenue, Charlottesville, VA 22903 and surrounding areas.

## Response Actions

1. USB drive secured in locked cabinet (Lab 302, Rice Hall)
2. Investigator: Name: Sarah Mitchell notified at sarah.mitchell@virginia.edu
3. IRB notification sent to irb-review@virginia.edu on 02/12/2026
4. IT Security contacted at security@virginia.edu, ticket #SEC-2026-1189
5. Affected participants notified by phone:
   - James Rodriguez: (434) 555-0183
   - Maria Gonzalez: (434) 555-0147
   - William Foster: 434.555.0276

## Remediation

- All portable drives must be encrypted (BitLocker/LUKS) per policy
- Access logs from 128.143.67.89 reviewed — no unauthorized access detected
- Additional monitoring enabled on subnet 192.168.1.0/24
- Name: David Chen tasked with implementing automated encryption checks

## Sign-off

- Report Author: Name: Rebecca Torres, r.torres@lawfirm.com
- Reviewed by: Name: Lisa Nakamura, l.nakamura@virginia.edu
- Date: February 14, 2026
