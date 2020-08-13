create or replace table project_unicorn.top_organizations as
SELECT org_name, ai_pubs, Grid_ID FROM
(SELECT org_name, count(distinct ds_id) as ai_pubs, ARRAY_AGG(distinct Grid_ID) as Grid_ID FROM `gcp-cset-projects.project_unicorn.grid_ai_pubs_052920` group by org_name order by ai_pubs desc LIMIT 100)
UNION ALL
(SELECT regexp_extract(org_name, "(?i)Microsoft") as org_name, count(distinct ds_id) as ai_pubs, ARRAY_AGG(distinct Grid_ID) as Grid_ID FROM `gcp-cset-projects.project_unicorn.grid_ai_pubs_052920` WHERE org_name LIKE "%Microsoft%" group by org_name)
UNION ALL
(SELECT SUBSTR(org_name, 0, 3) as org_name, count(distinct ds_id) as ai_pubs, ARRAY_AGG(distinct Grid_ID) as Grid_ID FROM `gcp-cset-projects.project_unicorn.grid_ai_pubs_052920` WHERE org_name LIKE "IBM%" group by org_name)
UNION ALL
(SELECT "Google" as org_name, count(distinct ds_id) as ai_pubs, ARRAY_AGG(distinct Grid_ID) as Grid_ID FROM `gcp-cset-projects.project_unicorn.grid_ai_pubs_052920` WHERE org_name LIKE "Google%" or org_name LIKE "Alphabet%" or org_name LIKE "DeepMind%")
UNION ALL
(SELECT regexp_extract(org_name, "(?i)Facebook") as org_name, count(distinct ds_id) as ai_pubs, ARRAY_AGG(distinct Grid_ID) as Grid_ID FROM `gcp-cset-projects.project_unicorn.grid_ai_pubs_052920` WHERE org_name LIKE "%Facebook%" group by org_name)
UNION ALL
(SELECT regexp_extract(org_name, "(?i)Apple") as org_name, count(distinct ds_id) as ai_pubs, ARRAY_AGG(distinct Grid_ID) as Grid_ID FROM `gcp-cset-projects.project_unicorn.grid_ai_pubs_052920` WHERE org_name LIKE "Apple%" group by org_name)
UNION ALL
(SELECT regexp_extract(org_name, "(?i)Amazon") as org_name, count(distinct ds_id) as ai_pubs, ARRAY_AGG(distinct Grid_ID) as Grid_ID FROM `gcp-cset-projects.project_unicorn.grid_ai_pubs_052920` WHERE org_name LIKE "Amazon%" group by org_name)
order by ai_pubs desc