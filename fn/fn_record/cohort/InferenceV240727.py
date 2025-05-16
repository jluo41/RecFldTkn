import os

import json

import pandas as pd

import numpy as np

OneCohort_Args = {'CohortLabel': 9,
 'CohortName': '20240410_Inference',
 'FolderPath': './_Data/0-Data_Raw/Inference/',
 'SourcePath': './_Data/0-Data_Raw/Inference/inference_test_v240410',
 'Source2CohortName': 'InferenceV240727'}

SourceFile_SuffixList = ['json']

def generate_inv_details(inv_old):
    inv_old = inv_old.copy()
    inv = {}
    inv['patient_id_encoded'] = inv_old['patientId']
    inv['invitation_id_encoded'] = inv_old['invitationId']
    inv['invitation_date'] = inv_old['invitationDate']
    inv['created_date'] = inv_old['createdDate']
    inv['invitation_state'] = inv_old['state']          # <--- not sure. is this available before sending out the message. 
    inv['workflow_step'] = inv_old['workflowStep']   
    inv['invitation_type'] = inv_old['invitationType'] 
    inv['partner_id'] = 'None'
    inv['message_id'] = 'None'
    return inv 


def get_age_bucket(x):
    if x >= 18 and x <= 30:
        return '18-30'
    elif x > 30 and x <= 40:
        return '31-40'
    elif x > 40 and x <= 50:
        return '41-50'
    elif x > 50 and x <= 60:
        return '51-60'
    elif x > 60 and x <= 70:
        return '61-70'
    elif x > 70 and x <= 80:
        return '71-80'
    elif x > 80:
        return '81+'


def generate_rx_details(rx_old, InvID_value, PID_value, invitation_date):
    rx = {}

    ptt = {}
    rx_old = rx_old.copy()
    ######################### make ids
    rx['patient_id_encoded'] = PID_value
    rx['invitation_id_encoded'] = InvID_value
    rx['prescription_id_encoded'] = rx_old['prescriptionId']
    #########################

    ######################### date
    rx['invitation_date'] = invitation_date
    #########################

    # ------------- patients -------------
    
    doy = pd.to_datetime(rx_old['patient']['dateOfBirth']).year 
    age = pd.to_datetime(invitation_date).year - doy 
    age_bucket = get_age_bucket(age)
    age_bucket_by5 = get_age_bucket_by5(age)
    ptt['patient_id_encoded'] = rx['patient_id_encoded']
    ptt['patient_gender'] = rx_old['patient']['gender']

    ptt['patient_age_bucket'] = age_bucket
    ptt['patient_age_by5'] = age_bucket_by5
    ptt['zipcode3'] = str(rx_old['patient']['address']['zipCode'])[:3]
    ptt['zipcode5'] = str(rx_old['patient']['address']['zipCode'])[:5]
    
    ptt['patient_age'] = age
    ptt['state_code'] = None              # <--------- not sure
    ptt['timezone'] = None 
    ptt['UserTimeZoneOffset'] = None 

    

    # ------------- prescriber -------------
    rx['prescriber_npi'] = rx_old['doctor']['npi']

    # ------------- pharmacy -------------
    rx['ncpdp_id'] = rx_old['pharmacy']['ncpdpid']
    rx['pharmacy_zip_code_3'] = str(rx_old['pharmacy']['address']['zipCode'])[:3] 
    rx['pharmacy_zip_code'] = str(rx_old['pharmacy']['address']['zipCode']) # [:3] 
    
    rx['pharmacy_name_rx'] = rx_old['pharmacy']['storeName'] # <--------  not sure
    rx['simple_pharmacy_name_rx'] = None  # <--------- not sure
    rx['specialty_pharmacy'] = None       # <--------- not sure
    rx['legal_business_name'] = None # <--------- not sure

    # ------------- prescription details -------------
    rx['prescription_type'] = rx_old['prescriptionType']
    rx['directions'] = rx_old['directions']
    rx['delivery_type'] = rx_old['deliveryType']
    rx['controlled_substance_code'] = None
    rx['quantity'] = rx_old['quantity']
    rx['days_supply'] = rx_old['daysSupply']
    rx['fill_number'] = rx_old['fillNumber']
    rx['quantity_unit'] = rx_old['quantityUnit']

    rx['refills_available'] = rx_old['refills'] 
    rx['package_size'] = rx_old['medication']['dispensedPackageQuantity'] # None      # <--------- not sure
    rx['package_units'] = None     # <--------- not sure
    

    # ------------- drug / medication -------------
    rx['ndc_id'] = rx_old['medication']['ndc']
    rx['drug_name'] = rx_old['medication']['drugName']
    rx['drug_description'] = rx_old['medication']['drugDescription']
    rx['form'] = rx_old['medication']['form'] # <--------- not sure. 
    
    rx['top_200_branded_drugs'] = rx_old['medication']['top200BrandedDrugs']
    rx['top_50_generic_drugs'] = rx_old['medication']['top50GenericDrugs']
    rx['brand_source'] = rx_old['medication']['brandingSource']
    rx['suppliers'] = rx_old['medication']['suppliers']
    rx['legend_status'] = rx_old['medication']['legendStatus']


    rx['therapeutic_equivalence_id'] = rx_old['medication']['therapeuticEquivalenceId']
    rx['ther_eq_hierarchy_name'] = rx_old['medication']['therEqHierarchyName']
    rx['ther_eq_parent_name'] = rx_old['medication']['therEqParentName']
    rx['ther_eq_ult_parent_name'] = rx_old['medication']['therEqUltParentName']
    rx['ther_eq_ult_child_ind'] = rx_old['medication']['therEqUltChildInd']
    rx['ther_eq_ult_parent_etc_id'] = rx_old['medication']['therEqUltParentEtcId']
    rx['ther_eq_hierarchy_level'] = rx_old['medication']['therEqHierarchyLevel']
    
    rx['strength']               = rx_old['medication']['strength']
    rx['total_package_strength'] = rx_old['medication']['totalPackageStrength'] 
    rx['strength_units']         = rx_old['medication']['strengthUnits']
    
    rx['generic_id'] = rx_old['medication']['genericId']#  None  # genericId

    # ------------- settings -------------
    rx['show_rems_campaigns'] = rx_old['settings']['showREMSCampaigns']
    rx['show_coupon_campaigns'] = rx_old['settings']['showCouponCampaigns']
    rx['show_educational_campaigns'] = rx_old['settings']['showEducationalCampaigns']
    rx['show_internal_campaigns'] = rx_old['settings']['showInternalCampaigns']
    rx['show_target_campaigns'] = rx_old['settings']['showTargetCampaigns']
    rx['show_experimental_campaigns'] = rx_old['settings']['showExperimentalCampaigns']
    rx['send_refill_reminder_messages'] = rx_old['settings']['sendRefillReminderMessages']
    rx['send_renewal_reminder_messages'] = rx_old['settings']['sendRenewalReminderMessages']
    rx['supports_hippo_prices'] = rx_old['settings']['supportsHippoPrices']
    rx['supports_hippo_prices65'] = rx_old['settings']['supportsHippoPrices65']
    rx['supports_hippo_prices_medicare'] = rx_old['settings']['supportsHippoPricesMedicare']
    rx['supports_copay_prices'] = rx_old['settings']['supportsCopayPrices']

    # ------------- source -------------
    rx['source_id'] = rx_old['sourceId']
    rx['customer_type'] = rx_old['customer']['type']
    rx['organization_type'] = rx_old['organization']['type']

    
    # ------------- eligibility -------------
    rx['group_id'] = rx_old['eligibility']['group'] # <--------- not sure
    rx['rx_bin'] = rx_old['eligibility']['rxBin']
    rx['rx_pcn'] = rx_old['eligibility']['rxPcn']
    rx['plan_name'] = rx_old['eligibility']['planName']
    

    # ------------- others -------------
    rx['primary_dispenser_type'] = None 
    rx['primary_dispenser_type_code'] = None 
    rx['dispenser_class_code'] = None 
    rx['dispenser_class'] = None 
    rx['secondary_dispenser_type_code'] = None 
    rx['tertiary_dispenser_type_code'] = None 

    # ------------- others, added in 2024-07-27 ---------
    rx['record_locator_encoded'] = None 
    rx['start_date'] = None 
    rx['campaign_content_id'] = None 
    rx['insurance_start_date'] = None 


    # ------------- others -------------
    rx['content_id'] = None 
    rx['campaign_id'] = None 
    rx['CmpRmd'] = 0
    rx['CmpLearn'] = 0
    rx['CmpSave'] = 0
    
    return rx, ptt


def fill_missing_keys(input_form, template):
    # Create a copy of the template to preserve the original structure
    filled_form = template.copy()
    
    # Function to recursively fill missing keys based on the template
    def recurse_fill(current_form, current_template):
        if isinstance(current_template, dict):
            # Ensure current_form is a dictionary
            if not isinstance(current_form, dict):
                current_form = {}
            for key in current_template:
                if isinstance(current_template[key], dict):
                    # If the template value is a dictionary, recurse into it
                    current_form[key] = recurse_fill(current_form.get(key, {}), current_template[key])
                elif isinstance(current_template[key], list) and current_template[key] and isinstance(current_template[key][0], dict):
                    # If the template value is a list of dictionaries, process each dictionary
                    if key in current_form and isinstance(current_form[key], list):
                        # Ensure each element in the input list conforms to the template
                        current_form[key] = [recurse_fill(elem, current_template[key][0]) for elem in current_form[key]]
                    else:
                        # If the key is missing or not a list in the input, initialize it with a list containing a filled template dict
                        current_form[key] = [recurse_fill({}, current_template[key][0])]
                else:
                    # Set the key to None if it is not present in the input and not a dict or list of dicts
                    current_form[key] = current_form.get(key, None)
            return current_form
        elif isinstance(current_template, list) and current_template and isinstance(current_template[0], dict):
            # If the top-level template itself is a list of dictionaries
            if isinstance(current_form, list):
                return [recurse_fill(elem, current_template[0]) for elem in current_form]
            else:
                return [recurse_fill({}, current_template[0])]
        else:
            # Return None for unexpected types or if the template does not contain a dict or list of dicts
            return None

    # Call the recursive function starting with the entire form
    filled_form = recurse_fill(input_form, filled_form)
    return filled_form


def get_InferenceEntry(OneCohort_Args, 
                      SourceFile_List, 
                      get_RawName_from_SourceFile):
    
    # SourceFile_List = SourceFileList_or_InferenceEntry
    
    ## 1. sample and template
    Inference_EntryPath = {} # RawName_to_dfRawPath
    # assert len(SourceFile_List) == 1
    for file_path in SourceFile_List:
        RawName = get_RawName_from_SourceFile(file_path, OneCohort_Args)
        Inference_EntryPath[RawName] = file_path
    
    Inference_Entry = {}
    RawName = 'sample'
    inference_form_path = Inference_EntryPath[RawName]
    with open(inference_form_path, 'r') as json_file:
        inference_form = json.load(json_file)
    Inference_Entry['inference_form'] = inference_form

    RawName = 'template'
    template_form_path = Inference_EntryPath[RawName]
    with open(template_form_path, 'r') as json_file:
        template_form = json.load(json_file)
    Inference_Entry['template_form'] = template_form
    return Inference_Entry


def get_RawName_from_SourceFile(file_path, OneCohort_Args):
    # RawName = file_path.split('_df_')[0].split('/')[-1] # split('.')[0]
    # RawName = file_path.split('_df')[0].split('/')[-1] # split('.')[0]
    RawName = file_path.split('inference_form_')[-1].split('.json')[0] # split('.')[0]
    return RawName


def process_Source_to_Raw(OneCohort_Args, 
                          SourceFileList_or_InferenceEntry, 
                          get_RawName_from_SourceFile,
                          SPACE):
    

    # 1. prepare inference_form
    if type(SourceFileList_or_InferenceEntry) == list:
        SourceFile_List = SourceFileList_or_InferenceEntry
        Inference_Entry = get_InferenceEntry(OneCohort_Args, 
                                             SourceFile_List, 
                                             get_RawName_from_SourceFile)
        
    else:
        Inference_Entry = SourceFileList_or_InferenceEntry

    assert 'template_form' in Inference_Entry
    assert 'inference_form' in Inference_Entry
    inference_form = Inference_Entry['inference_form']
    template_form = Inference_Entry['template_form']
    inference_form = fill_missing_keys(inference_form, template_form)



    # ----------- current invitation ------------
    invitation = inference_form['invitation'].copy()

    # ----------- in the future, these information should be saved in invitation ------------
    invitation['patient'] = inference_form['patient']
    invitation['eligibility'] = inference_form['eligibility']
    invitation['prescriptions'] = inference_form['prescriptions']
    invitation_data = [invitation]+ [i['invitation'] for i in inference_form.get('previous_invitations', [])]


    inv_list = []
    rx_list  = []
    # egm_list = []
    for inv_detail in invitation_data:
        inv = generate_inv_details(inv_detail)
        inv_list.append(inv)

        InvID_value = inv['invitation_id_encoded']
        PID_value = inv['patient_id_encoded']
        invitation_date = inv['invitation_date']

        eligibility = inv_detail['eligibility']
        patient = inv_detail['patient']

        for rx_loc_id, rx_detail in enumerate(inv_detail['prescriptions']):
            rx_detail['eligibility'] = eligibility
            rx_detail['patient'] = patient
            rx, ptt = generate_rx_details(rx_detail, InvID_value, PID_value, invitation_date)
            rx_list.append(rx)

        ## TODO: historical engagement data

    df_inv = pd.DataFrame(inv_list)
    df_rx  = pd.DataFrame(rx_list)
    df_ptt = pd.DataFrame([ptt])
    # df_egm = pd.DataFrame(egm_list) # TODO


    old_names = ['patient_id_encoded', 'patient_gender', 'patient_age_bucket', 'patient_age_by5', 
       'zipcode3', 'zipcode5', 'patient_age', 'state_code', 'timezone',
       'UserTimeZoneOffset']

    new_names = ['patient_id_encoded', 'patient_gender', 'patient_age_bucket',  'patient_age_by5', 
        'patient_zipcode_3', 'patient_zipcode_5', 'patient_age', 'state_code', 'timezone',
        'UserTimeZoneOffset']

    df_ptt.rename(columns = dict(zip(old_names, new_names)), inplace = True)

    selected_names = ['patient_id_encoded', 'patient_gender', 'patient_age_bucket',  'patient_age_by5', 
        'patient_zipcode_3', 'patient_zipcode_5', 
        'patient_age', 
        # 'state_code', 'timezone',
        # 'UserTimeZoneOffset'
        ]

    df_ptt = df_ptt[selected_names].reset_index(drop = True)
    # df_ptt


    df = df_ptt
    df = df.groupby('patient_id_encoded').first().reset_index() # .value_counts()
    df = df.rename(columns = {
        'patient_zipcode_3': 'zipcode3', 
        'patient_zipcode_5': 'zipcode5', 
        })

    if 'zipcode3' in df.columns:
        df['zipcode3'] = df['zipcode3'].astype(str).apply(lambda x: x.split('.')[0])
    if 'zipcode5' in df.columns:
        df['zipcode5'] = df['zipcode5'].astype(str).apply(lambda x: x.split('.')[0])


    if 'timezone' not in df.columns and 'zipcode3' in df.columns:
        ##### everytime, it will read it. 
        # print('here')
        zipcode3_geo_path = os.path.join(SPACE['DATA_EXTERNAL'], 'zipcode3/zipcode3_to_geo.pkl')
        df_zip3 = pd.read_pickle(zipcode3_geo_path)
        # print(df_zip3)
        df = pd.merge(df, df_zip3[['zipcode3', 'state', 'timezone', 'UserTimeZoneOffset']], how = 'left', on = 'zipcode3')

    elif 'timezone' not in df.columns and 'zipcode5' in df.columns:
        print('here2')
        zipcode3_geo_path = os.path.join(SPACE['DATA_EXTERNAL'], 'zipcode5/zipcode5_to_geo.pkl')
        df_zip3 = pd.read_pickle(zipcode3_geo_path)
        df = pd.merge(df, df_zip3[['zipcode5', 'state', 'timezone', 'UserTimeZoneOffset']], how = 'left', on = 'zipcode5')
    else:
        print('here3')
        df['timezone'] = None 
        df['UserTimeZoneOffset'] = None 

    df = df.rename(columns = {'state': 'state_code'})
    df_ptt = df



    RawName_to_dfRaw = {
        'invitation': df_inv, 
        'Rx': df_rx, 
        'Ptt': df_ptt, 
        # 'Engagement': df_egm # TODO
    }

    RawName_to_dfRaw = {k: v for k, v in RawName_to_dfRaw.items() if len(v) > 0}
    return RawName_to_dfRaw


def get_age_bucket_by5(x):
    ageby5 = round(x / 5, 0) * 5
    ageby5 = int(ageby5)
    return ageby5


MetaDict = {
	"OneCohort_Args": OneCohort_Args,
	"SourceFile_SuffixList": SourceFile_SuffixList,
	"generate_inv_details": generate_inv_details,
	"get_age_bucket": get_age_bucket,
	"generate_rx_details": generate_rx_details,
	"fill_missing_keys": fill_missing_keys,
	"get_InferenceEntry": get_InferenceEntry,
	"get_RawName_from_SourceFile": get_RawName_from_SourceFile,
	"process_Source_to_Raw": process_Source_to_Raw,
	"get_age_bucket_by5": get_age_bucket_by5
}