create or replace table project_unicorn.coauthors_dimensions_publications_with_abstracts as
SELECT distinct STRING_AGG(DISTINCT org_name, "; ") as coauthors, STRING_AGG(DISTINCT grid, "; ") as grid, id, year, version_of_record, STRING_AGG(DISTINCT detailed_field, "; ") as detailed_field, STRING_AGG(DISTINCT CAST(detailed_field_id as STRING), "; ") as detailed_field_id, STRING_AGG(DISTINCT field, "; ") as field, STRING_AGG(DISTINCT CAST(field_id as STRING), "; ") as field_id, journal_title, conference_name, title, abstract FROM
(SELECT * FROM `gcp-cset-projects.project_unicorn.dimensions_publications_with_abstracts_top_organizations_052920`) as abstracts
INNER JOIN
(SELECT org_name as name, grid as org_grid FROM project_unicorn.top_organizations, UNNEST(Grid_ID) as grid) as orgs
ON abstracts.grid = orgs.org_grid
group by id, year, version_of_record, journal_title, conference_name, title, abstract