import pandas as pd
import os
import sys


conceptToUri = {
    'S':...,
    'K':...,
    'T':...,
    'L':...,
    'S4':...,
    'S5':...,
    'S2':...,
    'S1':...
}

ESCO_DIR = "../../../esco/"
files = os.listdir(ESCO_DIR)
esco = pd.read_csv(ESCO_DIR + "skills_en.csv")
print("ESCO contains :", len(esco), "entities.") #13896
esco = esco[['conceptUri', 'preferredLabel',  'altLabels', 'description']]

relationskill = pd.read_csv(ESCO_DIR + "broaderRelationsSkillPillar_en.csv")
skillshierarchy = pd.read_csv(ESCO_DIR + "skillsHierarchy_en.csv")

uritoexclude = ['http://data.europa.eu/esco/isced-f/06', ## Information and Communication Technologies
                'http://data.europa.eu/esco/skill/869fc2ce-478f-4420-8766-e1f02cec4fb2', ## managament skills S4
                'http://data.europa.eu/esco/skill/243eb885-07c7-4b77-ab9c-827551d83dc4', ## Working with computer S5
                'http://data.europa.eu/esco/skill/8267ecb5-c976-4b6a-809b-4ceecb954967', ## Thinking skills and Competences T2
                'http://data.europa.eu/esco/skill/0a2d70ee-d435-4965-9e96-702b2fb65740', ## Information skills S2
                'http://data.europa.eu/esco/skill/dc06de9f-dd3a-4f28-b58f-b01b5ae72ab8', ## Communication, Collaboration S1
                'http://data.europa.eu/esco/skill/b94686e3-cce5-47a2-a8d8-402a0d0ed44e', ## Core skills T1
        ]

relationskill1 = relationskill[relationskill['conceptUri'].isin(uritoexclude)]
print("Number of ConceptURI dans la liste > ", len(relationskill1)) 
relationskill2 = relationskill[relationskill['broaderUri'].isin(uritoexclude)]
print(len(relationskill2)) # 2: conceptUri = http://data.europa.eu/esco/isced-f/061, http://data.europa.eu/esco/isced-f/068
esco1 = esco[esco['conceptUri'].isin(uritoexclude)]
print("Number of matches in ESCO > ", len(esco1)) # 0
skillshierarchy1 = skillshierarchy[skillshierarchy['Level 0 URI'].isin(uritoexclude)]
print("Number of matches in Level0 hierarchy = (S, K, T, L) > ", len(skillshierarchy1)) # 0
skillshierarchy2 = skillshierarchy[skillshierarchy['Level 1 URI'].isin(uritoexclude)]
print("Number of matches in Level1 hierarchy = (Sn, Kn, Tn, Ln) > ", len(skillshierarchy2)) 
#
#  Level 0 URI = 'http://data.europa.eu/esco/skill/c46fcb45-5c14-4ffa-abed-5a43f104bb22'
# Level 2 URI = http://data.europa.eu/esco/isced-f/061, http://data.europa.eu/esco/isced-f/068
skillshierarchy3 = skillshierarchy[skillshierarchy['Level 2 URI'].isin(uritoexclude)]
print("Number of matches in Level2 hierarchy = (Si.j, Ki.j, Ti.j, Li.j) > ", len(skillshierarchy3)) 
skillshierarchy4 = skillshierarchy[skillshierarchy['Level 3 URI'].isin(uritoexclude)]
print("Number of matches in Level3 hierarchy = (Si.j.k, Ki.j.k, Ti.j.k, Li.j.k) > ", len(skillshierarchy3)) 



for level in ['Level 0 URI', 'Level 1 URI', 'Level 2 URI', 'Level 3 URI']:
    print(f"Matching is ESCO conceptUri at level : {level.split(' ')[1]} > {len(esco[esco['conceptUri'].isin(list(skillshierarchy[level]))])}")
    print(f"Matching is ESCO broaderUri at level : {level.split(' ')[1]} > {len(esco[esco['conceptUri'].isin(list(skillshierarchy[level]))])}")
# esco['conceptUri'] looks like this: http://data.europa.eu/esco/skill/000f1d3d-220f...

# conclusion: esco['conceptUri'] are all in relationskill['conceptUri'] --> can deduce broaderURi

for level in ['Level 0 URI', 'Level 1 URI', 'Level 2 URI', 'Level 3 URI']:
    print(f"Matching in relation skill broaderUri at Level {level.split(' ')[1]} > {len(relationskill[relationskill['broaderUri'].isin(list(skillshierarchy[level]))])}")

# conclusion: the broaderUri mostly corresponds to skillshierarchy['Level 3 URI']

# the uritoexclude = 'http://data.europa.eu/esco/isced-f/06' is in skillshierarchy['Level 1 URI']
# The associated skillshierarchy['Level 3 URI'] are:
broaderskill_toexclude = list(skillshierarchy[skillshierarchy['Level 1 URI'].isin(uritoexclude)]['Level 3 URI'])

skillshierarchy['Level 3 URI'] = skillshierarchy['Level 3 URI'].fillna('')
# these two lists are the same, so the pattern is right

# get conceptUri of broader skills to exclude:
conceptUri_toexclude = list(relationskill[relationskill['broaderUri'].isin(broaderskill_toexclude)]['conceptUri'])

esco_filtered = esco[esco['conceptUri'].isin(conceptUri_toexclude)]


# merge skillshierarchy with esco
# 1/ get broader URi
esco_filtered_augmented = pd.merge(esco_filtered, relationskill, left_on='conceptUri', right_on='conceptUri', how='left')
print(len(esco_filtered_augmented)) # 3281
esco_filtered_augmented = esco_filtered_augmented[['conceptUri', 'preferredLabel', 'altLabels', 'description', 'broaderUri']]
# After merging we get duplicated conceptURi
# 2/ get skill levels
esco_filtered_augmented2 = pd.merge(esco_filtered_augmented, skillshierarchy, left_on='broaderUri', right_on='Level 3 URI', how='left')
# remove NAs
esco_filtered_augmented2.isnull().sum()
esco_filtered_augmented2.dropna(subset=['Level 0 URI'], inplace=True)



esco_filtered_augmented2 = esco_filtered_augmented2[['conceptUri', 'preferredLabel', 'altLabels', 'description', 'Level 0 preferred term', 'Level 1 preferred term', 'Level 2 preferred term', 'Level 3 preferred term']]

esco_filtered_augmented2.rename(columns={'preferredLabel': 'Type Level 4', 'description': 'Definition', 'Level 1 preferred term': 'Type Level 1', 'Level 2 preferred term': 'Type Level 2', 'Level 3 preferred term': 'Type Level 3', 'conceptUri': 'Source', 'Level 0 preferred term': 'Dimension'}, inplace=True)
esco_filtered_augmented2["unique_id"] = [1000+ i for i in range(len(esco_filtered_augmented2))]
esco_filtered_augmented2["name"] = esco_filtered_augmented2["Type Level 4"]
print("Number of skills : ", len(esco_filtered_augmented2.index))


