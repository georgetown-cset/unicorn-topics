-- getting just the ai publications with their grid info
CREATE OR REPLACE TABLE
  project_unicorn.grid_ai_pubs_052920 AS
  --combine everything together
SELECT
  sc.*,
  org_name,
  country
FROM (
-- get the grid and the paper's digital science id grouped
  SELECT
    DISTINCT Grid_ID,
    ds_id
  FROM (
  -- get the distinct papers with their grid
    SELECT
      DISTINCT ds_id,
      af2 AS Grid_ID
    FROM (
    -- get the affiliation info added from the publication data
      SELECT
        id AS ds_id,
        af.grid_ids AS AffliationId
      FROM
        `gcp-cset-projects.project_unicorn.dimensions_publications_with_abstracts_052920`,
        UNNEST( author_affiliations) AS af),
      UNNEST(AffliationId) AS af2
      -- make sure the paper is an AI paper
    WHERE
      ds_id IN (
      SELECT
        ds_id
      FROM
        gcp-cset-projects.project_unicorn.definitions_brief_082120
      WHERE
        arxiv_scibert_hit = TRUE))
  GROUP BY
    Grid_ID,
    ds_id) sc
LEFT JOIN (
-- add in the GRID info
  SELECT
    id,
    name AS org_name,
    country_name AS country
  FROM
    gcp-cset-projects.project_unicorn.api_grid) gr
ON
  sc.Grid_ID = gr.id