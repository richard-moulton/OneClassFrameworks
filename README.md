# OneClassFrameworks
Three frameworks developed as part of my Master's thesis; each uses knowledge of the majority class's sub-concept structure to improve one-class classifier performance in data streams.

Designed to be compatible with the MOA (Massive Online Analysis) 17.06 release. MOA [1] is a Java-based, open source framework for data stream mining. More details can be found on its website (https://moa.cms.waikato.ac.nz/) and it can be found on GitHub as well (https://github.com/waikato/moa).

This code has a DOI: [![DOI](https://zenodo.org/badge/139605381.svg)](https://zenodo.org/badge/latestdoi/139605381)


FILES INCLUDED

classifiers/oneclass

[1] **OCComplete** has complete knowledge of the sub-concept structure for both the initialization and online portions;

[2] **OCFuzzy** has complete knowledge of the sub-concept structure for the initialization portion only; and

[3] **OCCluster** knows a sub-concept structure exists, but has no specific knowledge about it for either portion.

NOTE: Each of these frameworks produces an anomaly score as its output: the higher the score the more firmly the framework believes that instance to be. This is not converted into a vote, which means that MOA's built-in evaluation functions cannot be used.

core

[1] **FixedLengthList** is a utility class that extends ArrayList to produce a list that is restricted to an argument, fixed length; and

[2] **SMOTE** is an implementation of Chawla et al.'s Synthetic Minority Oversampling Technique [2].

REFERENCES

[1] Bifet, A., Holmes, G., Kirkby, R. & Pfahringer, B., Moa: Massive online analysis. J. Mach. Learn. Res. 11, 1601–1604 (2010).

[2] Chawla, N. V., Bowyer, K. W., Hall, L. O., & Kegelmeyer,  W. P., SMOTE: Synthetic minority over-sampling technique. J. Artif. Intell. Res., 16, 321–357 (2002).
