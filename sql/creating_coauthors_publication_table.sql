-- building final table for analysis
CREATE OR REPLACE TABLE
  project_unicorn.coauthors_dimensions_publications_with_abstracts AS
SELECT
-- combining all organizations to be the coauthors
  DISTINCT STRING_AGG(DISTINCT org_name, "; ") AS coauthors,
  -- and their grids
  STRING_AGG(DISTINCT grid, "; ") AS grid,
  -- gathering all the paper data
  id,
  year,
  version_of_record,
  STRING_AGG(DISTINCT detailed_field, "; ") AS detailed_field,
  STRING_AGG(DISTINCT CAST(detailed_field_id AS STRING), "; ") AS detailed_field_id,
  STRING_AGG(DISTINCT field, "; ") AS field,
  STRING_AGG(DISTINCT CAST(field_id AS STRING), "; ") AS field_id,
  journal_title,
  conference_name,
  title,
  abstract
FROM (
-- pull the data from the dataset containing the paper data
  SELECT
    *
  FROM
    `gcp-cset-projects.project_unicorn.dimensions_publications_with_abstracts_top_organizations_052920`) AS abstracts
INNER JOIN (
-- pull just the papers from the orgs we want
  SELECT
    org_name AS name,
    grid AS org_grid
  FROM
    project_unicorn.top_organizations,
    UNNEST(Grid_ID) AS grid) AS orgs
ON
  abstracts.grid = orgs.org_grid
  -- the fields that are unique per paper
GROUP BY
  id,
  year,
  version_of_record,
  journal_title,
  conference_name,
  title,
  abstract