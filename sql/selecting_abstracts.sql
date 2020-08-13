create or replace table project_unicorn.dimensions_publications_with_abstracts_top_organizations_052920 as
SELECT org_name, mag_subject, wos_subject, papers.* FROM
(SELECT id, grid, title, year, version_of_record, field_of_reference.second_level.name as detailed_field, field_of_reference.second_level.id as detailed_field_id,  field_of_reference.first_level.name as field, field_of_reference.first_level.id as field_id, journal.title as journal_title, conference.name as conference_name, abstract, FROM `gcp-cset-projects.project_unicorn.dimensions_publications_with_abstracts_052920`, UNNEST(author_affiliations) as author_affils, UNNEST(author_affils.grid_ids) as grid, UNNEST(`for`) as field_of_reference) as papers
-- Check each paper we're pulling to see if its ds_id matches the ds_id of a paper with a scibert hit
INNER JOIN
(SELECT org_name, id as ds_id, mag_subject, wos_subject FROM project_unicorn.dimensions_publication_ids_top_organizations_052920) as ids
ON papers.id = ids.ds_id