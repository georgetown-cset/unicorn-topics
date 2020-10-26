-- get the ids and a little supplementary data for all the dimensions papers from the top organizations
CREATE OR REPLACE TABLE
  project_unicorn.dimensions_publication_ids_top_organizations_052920 AS (
  SELECT
  -- get all the info we care about for these papers
    unicorn.* EXCEPT (Grid_ID),
    wos_subject,
    mag_subject,
    papers.* EXCEPT (grids),
  FROM (
  -- pull the paper id and their grid values from the dimensions publications table
    SELECT
      id,
      grids
    FROM
      `gcp-cset-projects.project_unicorn.dimensions_publications_with_abstracts_052920`,
      UNNEST(author_affiliations) AS author_affils,
      UNNEST(author_affils.grid_ids) AS grids) AS papers
      -- only pull the papers from top organizations
  INNER JOIN (
    SELECT
      *
    FROM
      project_unicorn.top_organizations,
      UNNEST(Grid_ID) AS gridval) AS unicorn
  ON
    papers.grids = unicorn.gridval
    -- only pull the AI papers; add in some info from the AI paper dataset
  INNER JOIN (
    SELECT
      ds_id,
      wos_subject,
      mag_subject,
      arxiv_scibert_hit
    FROM
      project_unicorn.definitions_brief_082120) AS ai
  ON
    papers.id = ai.ds_id
  WHERE
    arxiv_scibert_hit=TRUE)