-- WITH filtered_rows AS (
--     SELECT s.study_fk, s.study_iuid, DATE(s.created_at),
--            STRING_AGG(DISTINCT jsonb_elements_2.hil_finding ->> 'pathology', ', ') AS hil_pathology
--     FROM webhook."VendorResults" vr
--     LEFT JOIN study."Studies" s ON vr.study_iuid = s.study_iuid
--     LEFT JOIN log."ManualReportingVariables" mrv ON mrv.study_fk = s.study_fk 
--     LEFT JOIN webhook."ModelResults" mr ON mr.study_iuid = s.study_iuid,
--            LATERAL jsonb_array_elements(mrv.selected_findings::jsonb) AS jsonb_elements_2(hil_finding)
--     WHERE vendor_fk = '3' AND DATE(s.created_at) between '2024-06-01' and '2024-07-30'
--     AND s.mod_study_fk = '16' AND s.status = 'REPORTABLE'
--     GROUP BY s.study_fk, s.study_iuid, DATE(s.created_at)
-- ),
-- matching_rows AS (
--     SELECT *
--     FROM filtered_rows
--     WHERE hil_pathology ~* '(Humerus Post OP|Scoliosis|Milliary Tuberculosis|Subcutaneous Emphysema|Pneumoperitoneum)'
-- ),
-- additional_rows AS (
--     SELECT *
--     FROM filtered_rows
--     WHERE hil_pathology !~* '(Humerus Post OP|Scoliosis|Milliary Tuberculosis|Subcutaneous Emphysema|Pneumoperitoneum)'
--     LIMIT 1000 - (SELECT COUNT(*) FROM matching_rows)  
-- )
-- SELECT * FROM matching_rows
-- UNION ALL
-- SELECT * FROM additional_rows
-- LIMIT 1000

WITH filtered_rows AS (
    SELECT s.study_fk, s.study_iuid, DATE(s.created_at),
           STRING_AGG(DISTINCT jsonb_elements_2.hil_finding ->> 'pathology', ', ') AS hil_pathology
    FROM study."Studies" s
    LEFT JOIN log."ManualReportingVariables" mrv ON mrv.study_fk = s.study_fk,
           LATERAL jsonb_array_elements(mrv.selected_findings::jsonb) AS jsonb_elements_2(hil_finding)
    WHERE DATE(s.created_at) between '2024-08-01' and '2024-08-12'
    AND s.mod_study_fk = '35' AND s.status = 'REPORTABLE'
    GROUP BY s.study_fk, s.study_iuid, DATE(s.created_at)
),
matching_rows AS (
    SELECT *
    FROM filtered_rows
    WHERE hil_pathology ~* '(Reducing Joint Space with Osteophytes|Osteophytes|Total Knee Replacement|ACL Reconstruction|Fabella|Fracture|Loose Bodies|Post-OP|Prominent Tibial Spike)'
    limit 400
),
additional_rows AS (
    SELECT *
    FROM filtered_rows
    WHERE hil_pathology !~*'(Reducing Joint Space with Osteophytes|Osteophytes|Total Knee Replacement|ACL Reconstruction|Fabella|Fracture|Loose Bodies|Post-OP|Prominent Tibial Spike)'
    LIMIT 500 - (SELECT COUNT(*) FROM matching_rows)  
)
SELECT * FROM matching_rows
UNION ALL
SELECT * FROM additional_rows
LIMIT 500






