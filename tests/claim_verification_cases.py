TEST_CASES = [
    {
        "claim": "Diabetes causes high blood glucose.",
        "evidence": "Diabetes is a condition that causes high blood glucose.",
        "expected": "RELEVANT",
    },
    {
        "claim": "Common symptoms include frequent urination.",
        "evidence": "People with diabetes may urinate more often.",
        "expected": "PARTIAL",
    },
    {
        "claim": "Diabetes is caused by a fictional blue virus.",
        "evidence": "Diabetes involves problems with insulin and blood glucose regulation.",
        "expected": "UNRELATED",
    },
    {
        "claim": "Type 2 diabetes can involve insulin resistance.",
        "evidence": "In type 2 diabetes, the body does not use insulin properly.",
        "expected": "PARTIAL",
    },
]
