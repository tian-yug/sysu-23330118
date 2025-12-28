
# AHNS Model Experiment Summary Report

## Experiment Overview
This experiment aims to reproduce the AHNS (Adaptive Hardness Negative Sampling) model and compare it with baseline models BPRMF and LightGCN.

## Key Findings

### 1. Performance Comparison
From the experimental results:

**ml-1m Dataset**:
- BPRMF: NDCG@20 = 0.3672
- LightGCN: NDCG@20 = 0.3521 (4.1% lower than BPRMF)
- AHNS-v1: NDCG@20 = 0.3676 (0.1% higher than BPRMF)
- AHNS-v2: NDCG@20 = 0.3641 (0.8% lower than BPRMF)

**Grocery Dataset**:
- BPRMF: NDCG@20 = 0.3265
- LightGCN: NDCG@20 = 0.3311 (1.4% higher than BPRMF)
- AHNS-v1: NDCG@20 = 0.3267 (0.06% higher than BPRMF)
- AHNS-v2: NDCG@20 = 0.3254 (0.3% lower than BPRMF)

### 2. Computational Efficiency Analysis
**Training Time Comparison**:
- AHNS-v1 on ml-1m: 3882 seconds, 24 times slower than BPRMF
- AHNS-v2 (with smaller candidate pool M=16): 1565 seconds, but with slightly lower performance
- LightGCN: Moderate training time, but best performance on Grocery dataset

### 3. Conclusion
1. **AHNS did not achieve expected results**: Compared to the 2-8% improvement reported in the original paper, our reproduction showed very limited improvement (0.06-0.1%)
2. **High computational cost**: AHNS has high computational complexity, especially with candidate pool size M=32
3. **LightGCN performs better on sparse data**: On the Grocery dataset, LightGCN achieved the best performance

## Recommended Improvements
1. Reduce candidate pool size to balance computational cost and performance
2. Perform more detailed hyperparameter tuning
3. Try combining AHNS ideas with other models

## Detailed Results
Please refer to the detailed results tables and charts in the attachments.

---

*Report generated: 2025-12-27*
    