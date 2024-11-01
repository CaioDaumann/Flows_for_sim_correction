#python apply_flow_to_parquet.py --mcfilespath "/net/scratch_cms3a/daumann/HiggsDNA/v13_samples_2/DY_preEE_v13/" --process "Zee"
#python apply_flow_to_parquet.py --mcfilespath "/net/scratch_cms3a/daumann/PhD/EarlyHgg/flow_Zee_samples/DY_postEE_v13/" --process "Zee" --period "postEE" --outpath "./validate_post_processing_script/"
#python apply_flow_to_parquet.py --mcfilespath "./validate_post_processing_script/jq_zmmg_out.parquet" --process "Zmmg" --period "postEE" --outpath "./validate_post_processing_script/"

# For paper with IC guys !!!! (Zee)
#python apply_flow_to_parquet.py --mcfilespath "/net/scratch_cms3a/daumann/normalizing_flows_project/script_to_prepare_samples_for_paper/splited_parquet/DY_test.parquet" --process "Zee" --period "postEE" --outpath "./paper_with_IC/"

# for zmmg
python apply_flow_to_parquet.py --mcfilespath "/net/scratch_cms3a/daumann/massresdecorrhiggsdna/sigma_syst/jon_high_stat_zmmg/" --process "Zmmg" --period "postEE" --outpath "./paper_with_IC/"