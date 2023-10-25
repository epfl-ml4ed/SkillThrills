import sys

sys.path.append("../recommendation")  # Add the src directory to the path

from matchings import skill_skill_similarity

provided_level = 4
required_level = 4

similarity = skill_skill_similarity(provided_level, required_level)

print(f"Similarity: {similarity:.2f}")  # This should be 1, the maximum possible


provided_level = 4
required_level = 3

similarity = skill_skill_similarity(provided_level, required_level)

print(
    f"Similarity: {similarity:.2f}"
)  # This should be 1, the maximum possible because the provided level is higher than the required level


provided_level = 2
required_level = 4

similarity = skill_skill_similarity(provided_level, required_level)

print(
    f"Similarity: {similarity:.2f}"
)  # This should be 0.5, the provided level is half of the required level
