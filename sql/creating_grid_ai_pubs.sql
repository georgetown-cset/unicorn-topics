create or replace table project_unicorn.grid_ai_pubs_052920 as
select sc.*, org_name, country from
(select distinct Grid_ID, ds_id from (select distinct ds_id , af2 as Grid_ID
 from (select id as ds_id , af.grid_ids as AffliationId from
`gcp-cset-projects.project_unicorn.dimensions_publications_with_abstracts_052920` , UNNEST( author_affiliations) as af),
  UNNEST(AffliationId) as af2 where ds_id in (select ds_id from gcp-cset-projects.project_unicorn.definitions_brief_082120 where arxiv_scibert_hit =  true)) group by Grid_ID, ds_id) sc left join
(select id, name as org_name, country_name as country from gcp-cset-projects.project_unicorn.api_grid) gr ON  sc.Grid_ID = gr.id