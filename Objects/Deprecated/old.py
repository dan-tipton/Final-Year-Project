    dave_md14 = 0.015 * pow((1 + rev_redshifts), 2.7)/(1 + pow((1 + rev_redshifts)/2.9, 5.6)) 
    dave_sfr = rev_snrd_alt_scaled / kcc_chabrier
    dave_ratio = dave_sfr / dave_md14

    print('For Dave: sfr/md14')
    for idx, r in enumerate(dave_ratio):
        print(f"redshift {rev_redshifts[idx]}, ratio: {dave_ratio[idx]}")

    print('red to green')
    red_to_green = rev_snrd_1000_scaled / rev_sfrd_all
    for idx, r in enumerate(dave_ratio):
        print(f"redshift {rev_redshifts[idx]}, ratio: {red_to_green[idx]}")

    print('cyan to green')
    cyan_to_green = rev_snrd_alt_scaled / rev_sfrd_all
    for idx, r in enumerate(dave_ratio):
        print(f"redshift {rev_redshifts[idx]}, ratio: {cyan_to_green[idx]}")