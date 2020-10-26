-- getting the list of top organizations and their grids and AI paper counts
CREATE OR REPLACE TABLE
  project_unicorn.top_organizations AS
  -- get the org names, paper counts, and GRIDs
SELECT
  org_name,
  ai_pubs,
  Grid_ID
FROM (
-- get the org names, count the paper IDs, and aggregate the GRIDs, pulling from the ai publications dataset
  SELECT
    org_name,
    COUNT(DISTINCT ds_id) AS ai_pubs,
    ARRAY_AGG(DISTINCT Grid_ID) AS Grid_ID
  FROM
    `gcp-cset-projects.project_unicorn.grid_ai_pubs_052920`
    -- group by the organization
  GROUP BY
    org_name
    -- order by the count of publications
  ORDER BY
    ai_pubs DESC
    -- only include the top 100
  LIMIT
    100)
    -- add in the 6 companies we care about
UNION ALL (
  SELECT
  -- get the count of microsoft publications
    REGEXP_EXTRACT(org_name, "(?i)Microsoft") AS org_name,
    COUNT(DISTINCT ds_id) AS ai_pubs,
    ARRAY_AGG(DISTINCT Grid_ID) AS Grid_ID
  FROM
    `gcp-cset-projects.project_unicorn.grid_ai_pubs_052920`
  WHERE
    org_name LIKE "%Microsoft%"
  GROUP BY
    org_name)
UNION ALL (
-- get the count of IBM publications
  SELECT
    SUBSTR(org_name, 0, 3) AS org_name,
    COUNT(DISTINCT ds_id) AS ai_pubs,
    ARRAY_AGG(DISTINCT Grid_ID) AS Grid_ID
  FROM
    `gcp-cset-projects.project_unicorn.grid_ai_pubs_052920`
  WHERE
    org_name LIKE "IBM%"
  GROUP BY
    org_name)
UNION ALL (
-- get the count of Google publications
  SELECT
    "Google" AS org_name,
    COUNT(DISTINCT ds_id) AS ai_pubs,
    ARRAY_AGG(DISTINCT Grid_ID) AS Grid_ID
  FROM
    `gcp-cset-projects.project_unicorn.grid_ai_pubs_052920`
  WHERE
    org_name LIKE "Google%"
    OR org_name LIKE "Alphabet%"
    OR org_name LIKE "DeepMind%")
UNION ALL (
-- get the count of Facebook publications
  SELECT
    REGEXP_EXTRACT(org_name, "(?i)Facebook") AS org_name,
    COUNT(DISTINCT ds_id) AS ai_pubs,
    ARRAY_AGG(DISTINCT Grid_ID) AS Grid_ID
  FROM
    `gcp-cset-projects.project_unicorn.grid_ai_pubs_052920`
  WHERE
    org_name LIKE "%Facebook%"
  GROUP BY
    org_name)
UNION ALL (
-- get the count of Apple publications
  SELECT
    REGEXP_EXTRACT(org_name, "(?i)Apple") AS org_name,
    COUNT(DISTINCT ds_id) AS ai_pubs,
    ARRAY_AGG(DISTINCT Grid_ID) AS Grid_ID
  FROM
    `gcp-cset-projects.project_unicorn.grid_ai_pubs_052920`
  WHERE
    org_name LIKE "Apple%"
  GROUP BY
    org_name)
UNION ALL (
-- get the count of Amazon publications
  SELECT
    REGEXP_EXTRACT(org_name, "(?i)Amazon") AS org_name,
    COUNT(DISTINCT ds_id) AS ai_pubs,
    ARRAY_AGG(DISTINCT Grid_ID) AS Grid_ID
  FROM
    `gcp-cset-projects.project_unicorn.grid_ai_pubs_052920`
  WHERE
    org_name LIKE "Amazon%"
  GROUP BY
    org_name)
    -- order the whole thing by the count of AI pubs
ORDER BY
  ai_pubs DESC