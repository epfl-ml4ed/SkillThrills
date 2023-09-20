# %%
import chardet

# %%
with open("CVTest_V4.csv", "rb") as file:
    result = chardet.detect(file.read())

encoding = result["encoding"]
confidence = result["confidence"]

print(f"Detected encoding: {encoding} (confidence: {confidence:.2f})")
