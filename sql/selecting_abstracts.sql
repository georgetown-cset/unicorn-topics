-- pull the publications for only the orgs we care about
CREATE OR REPLACE TABLE
  project_unicorn.dimensions_publications_with_abstracts_top_organizations_052920 AS
  -- pull the org_name plus some mag/wos data and all the dimensions data for all the papers we care about
SELECT
  org_name,
  mag_subject,
  wos_subject,
  papers.*
FROM (
-- pull all the dimensions data we care about
  SELECT
    id,
    grid,
    title,
    year,
    version_of_record,
    field_of_reference.second_level.name AS detailed_field,
    field_of_reference.second_level.id AS detailed_field_id,
    field_of_reference.first_level.name AS field,
    field_of_reference.first_level.id AS field_id,
    journal.title AS journal_title,
    conference.name AS conference_name,
    abstract,
  FROM
  -- pull from dimensions
    `gcp-cset-projects.project_unicorn.dimensions_publications_with_abstracts_052920`,
    UNNEST(author_affiliations) AS author_affils,
    UNNEST(author_affils.grid_ids) AS grid,
    UNNEST(`for`) AS field_of_reference) AS papers
  -- Check each paper we're pulling to see if its ds_id matches the ds_id of a paper with a scibert hit
INNER JOIN (
-- add in the info about which orgs we care about
  SELECT
    org_name,
    id AS ds_id,
    mag_subject,
    wos_subject
  FROM
    project_unicorn.dimensions_publication_ids_top_organizations_052920) AS ids
ON
  papers.id = ids.ds_id