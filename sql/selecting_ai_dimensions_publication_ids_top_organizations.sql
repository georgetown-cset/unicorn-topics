create or replace table project_unicorn.dimensions_publication_ids_top_organizations_052920 as
(SELECT unicorn.* EXCEPT (Grid_ID), wos_subject, mag_subject, papers.* EXCEPT (grids), FROM
(SELECT id, grids FROM `gcp-cset-projects.project_unicorn.dimensions_publications_with_abstracts_052920`, UNNEST(author_affiliations) as author_affils, UNNEST(author_affils.grid_ids) as grids) as papers
INNER JOIN
(SELECT * from project_unicorn.top_organizations, UNNEST(Grid_ID) as gridval) as unicorn
ON papers.grids = unicorn.gridval
INNER JOIN
(SELECT ds_id, wos_subject, mag_subject, arxiv_scibert_hit FROM project_unicorn.definitions_brief_082120) as ai
ON papers.id = ai.ds_id
WHERE arxiv_scibert_hit=true)