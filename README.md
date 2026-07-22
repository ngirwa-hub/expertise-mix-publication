This repository contains information on the experiments performed and their results, as included in the paper titled: *Role-Conditioned LLMs for Diverse Synthetic Expert Panels: Expert Elicitation on Barriers to Direct Current Adoption*


Experiment contents.
We have three folders (scrapping, working_dir, and roles-enhanced).

**Scrapping & Working_dir folders:** contain the initial context gathered when we obtained job advertisements to generate role profiles for LLM variants; more descriptions of the core scripts/files are provided below:
- Scraping scripts based on websites: as access requirements differ, some websites had to have independent scripts based on their access and content structures: energyjobsearch.py, euroclimate_scrapper.py, euroengineer_scrapper.py, rejobs_scrapper.py, and scrapping.ipynb.
- Folders:
    - For assessments, evaluation of standard content and categorization resulted in the following folders: scrapping/merged_jobs, scrapping/scrapped_jobs, and scrapping/standardized_jobs.
- Role-profile generation and assessment are in the working_dir folder:
    - working_dir/src-roles/role-generate.py: for generating the role profiles (3 runs/iterations)
    - working_dir/src-roles/role-embeddings.py: for generating embeddings from each of the generated profiles.
    - working_dir/src-roles/role-similarity.py: computing cosine similarity between the embeddings of the related role-profiles.
    - the final role-profiles are at: working_dir/final_mds/llm-generated-roles/
- Assessment of similarity between similar role-profiles (target to identify that the second profile in a similar role is not redundant) [path: working_dir/src-btn-profile/...]:
    - inter-role-coherence.ipynb: is the notebook that contains all of the steps taken, from role-profile embeddings to cosine similarity computations.
    - between_profile_similarity_heatmap.png: is the heatmap of the cosine similarity computed using the notebook.

**Roles-enhanced folder:** contains scripts for prompting LLMs and analyzing generated responses. First, the variants were prompted using {barrier_mention.py: inputs (variant lists: variants-few-enh.yaml and variants-few-new-enh.yaml); counter file (context_barrierMention_counter.txt)}. Thereafter, analysis (i.e., clustering and the Jaccard similarity index) was performed in this order:
- Data cleaning and analysis proceed as follows:
    1. Original working CSV is: expert_responses/context_barrier_mention_all.csv
    2. Recovering some rows that the script parser could not write properly.
        - Code cell: #cell 2: recovering
        - input: expert_responses/context_barrier_mention_all.csv
        - A new CSV is written: expert_responses/context_barrier_mention_all_recovered.csv
    3. Cleaning of malformed generated content using: analysis/clean_barriers.py
        - new outputs:
            - new_clean: analysis/context_barrier_mention_clean.csv
            - report on cleaned content: analysis/context_barrier_mention_repetition_report.csv
    4. Assigning themes by clustering: named #cell 3 in this notebook:
        - input: analysis/context_barrier_mention_clean.csv
        - output: analysis/context_barrier_mention_hdbscan.csv
    5. Cluster naming:
        - input: analysis/context_barrier_mention_hdbscan.csv
        - GPT naming, output file: analysis/context_barrier_mention_hdbscan_gpt5named.csv
        - Author inspection and renaming of the GPT version, output file: analysis/context_barrier_mention_hdbscan_gpt5named_human.csv

    6. Verification that all rows were filled and remained intact: #cell 5 in this notebook. The outputs are not intended for export; they are only views in the notebook outputs.

    7. A LaTeX table contains information such as (cluster_id, cluster_size, share_of_rows, gpt_cluster_name, human_labelled). It is computed in #cell 6, and the output can be seen in: cluster_id_size_name_table.tex

    8. Theme-subtheme mappings for the actual paper content:
        - #cell 7a: contains a LaTeX-exportable table
        - #cell 7b: the Sankey plot on the theme-subthemes mappings (in three different formats to increase quality—.pdf, .png, and .eps). In the paper we used the .eps format.

    9. Jaccard similarity index: #cell 8 compares similar role variants for within-model and between-model similarity. Here the heatmap outputs are in three formats; in the paper we used the .eps format too.
