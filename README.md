This repository contains contents in two categories, the online appendix file and the experiments performed and their results, as included in the <Anonymized-conference> manuscript. 
1. The online appendix is a PDF file found in the root, named: Online appendix.pdf

2. The experiements contents.
Whereby: we have three folders (scrapping, working_dir, and roles-enhanced). 

**Scrapping & Working_dir folder:** contains the initial context gathered when we obtained job advertisements to generate role profiles for LLM variants; more descriptions on core-scripts/files: 
- Scrapping scripts based on websites, as accesses differ some had to have independent scripts based on access and content structures: energyjobsearch.py, euroclimate_scrapper.py, euroengineer_scrapper.py, rejobs_scrapper.py and scrapping.ipynb.
- Folders: assessments, evaluation on standard contents, ealuation and categorization resulted into the following folders: scrapping/merged_jobs, scrapping/scrapped_jobs, scrapping/standardized_jobs.
- Generating role-profiles and assessments are in working_dir folder:
    - working_dir/src-roles/role-generate.py: for generating the role profiles (3 runs/iterations)
    - working_dir/src-roles/role-embeddings.py: for generating embeddings from each of the generated profiles
    - working_dir/src-roles/role-similarity.py: computing cosine similarity between the embeddings of the related role-profiles
    - the final role-profiles are at: working_dir/final_mds/final-profiles/
- Assessment of similarity between similar role-profiles (target to identify that the second profile in similar role is not a redundant) [path: scrapping/working_dir/src-btn-profile/...]: 
    - inter-role-coherence.ipynb: is the notebook that contains all of he steps taken from role-prifles embeddings, to cosine similarity computations
    - betweeen_profile_similarity_heatmap.png: is the heatmap of the cosine similarity computed using the notebook.

**Roles-enhanced folder:** contains scripts for prompting LLMs and analyzing generated responses, first the variants were prompted using {barrier_mention.py: inputs (variants lists:variants-few-enh.yaml, and variants-few-new-enh.yaml); counter file (context_barrierMention_counter.txt)}. Thereafter analysis (i.e., clustering and the Jaccard similarity index), in this order:
- Working on the data cleaning and analysis starts with:
    1. Original working csv is: context_barrier_mention_all.csv
    2. recovering some rows which the script parser could not write them properly. 
        - Code cell: #cell 2: recovering
        - input:  context_barrier_mention_all.csv
        - A new csv is written: context_barrier_mention_all_recovered.csv
    3. Cleaning of malformed generation using: analysis/clean_barrier.py
        - new outputs: 
            - new_clean: analysis/context_barrier_mention_clean.csv
            - report on cleaned: analysis/context_barrier_mention_repetition_report.csv
    4. Assigning themes by clustering: named #cell 3 in this notebook:
        - input: context_barrier_mention_clean.csv
        - output: analysis/context_barrier_mention_hdbscan.csv
    5. Cluster naming:
        - input: context_barrier_mention_hdbscan.csv
        - GPT naming, output file: context_barrier_mention_hdbscan_gpt5named.csv
        - Human inspection and re-naming of the GPT version, output file: context_barrier_mention_hdbscan_gpt5named_human.csv

    6. Verification inspection report on all rows being filled intact:  # cell 5 in this notebook, the outputs are not for exports just views in the notebook

    7. A latex table contains the information like (cluster_id, cluster_size, share_of_rows,gpt_cluster_name, human_labelled), all these are in the online appendix attachment; here computed in #cell 6

    8. Theme-subthemes mappings for the actual paper contents: 
        - #cell 7a: contains latex exportable table
        - #cell 7b: the Sankey plot on the theme-subthemes mappings

    9. Jaccard similarity index: #cell 8: comparing similar role variants for within-model and between models on similar-role variants

