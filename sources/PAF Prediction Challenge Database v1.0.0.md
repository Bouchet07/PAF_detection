---
title: "PAF Prediction Challenge Database v1.0.0"
source: "https://physionet.org/content/afpdb/1.0.0/"
author:
  - "[[George Moody]]"
published: 2001-03-01
created: 2026-05-17
description: "ECG recordings created for use in the Computers in Cardiology Challenge 2001, a competition with the goal of developing automated methods for predicting paroxysmal atrial fibrillation."
tags:
  - "clippings"
---
## PAF Prediction Challenge Database

---

> [!secondary] Secondary
> **When using this resource, please cite the original publication:**
> 
> [Moody GB, Goldberger AL, McClennen S, Swiryn SP. Predicting the Onset of Paroxysmal Atrial Fibrillation: The Computers in Cardiology Challenge 2001. Computers in Cardiology 28:113-116 (2001).](http://www.cinc.org/archives/2001/pdf/113.pdf)
> 
> **Please include the standard citation for PhysioNet:** [(show more options)](#citationModalPlatform)  
> Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R.,... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation \[Online\]. 101 (23), pp. e215–e220. RRID:SCR\_007345.
> 
> ##### Cite
> 
> | APA | Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P. C., Mark, R.,... & Stanley, H. E. (2000). PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation \[Online\]. 101 (23), pp. e215–e220. RRID:SCR\_007345. |
> | --- | --- |
> | MLA | Goldberger, A., et al. "PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation \[Online\]. 101 (23), pp. e215–e220." (2000). RRID:SCR\_007345. |
> | CHICAGO | Goldberger, A., L. Amaral, L. Glass, J. Hausdorff, P. C. Ivanov, R. Mark, J. E. Mietus, G. B. Moody, C. K. Peng, and H. E. Stanley. "PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation \[Online\]. 101 (23), pp. e215–e220." (2000). RRID:SCR\_007345. |
> | HARVARD | Goldberger, A., Amaral, L., Glass, L., Hausdorff, J., Ivanov, P.C., Mark, R., Mietus, J.E., Moody, G.B., Peng, C.K. and Stanley, H.E., 2000. PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation \[Online\]. 101 (23), pp. e215–e220. RRID:SCR\_007345. |
> | VANCOUVER | Goldberger A, Amaral L, Glass L, Hausdorff J, Ivanov PC, Mark R, Mietus JE, Moody GB, Peng CK, Stanley HE. PhysioBank, PhysioToolkit, and PhysioNet: Components of a new research resource for complex physiologic signals. Circulation \[Online\]. 101 (23), pp. e215–e220. RRID:SCR\_007345. |

### Abstract

This database of two-channel ECG recordings has been created for use in the Computers in Cardiology Challenge 2001, an open competition with the goal of developing automated methods for predicting paroxysmal atrial fibrillation (PAF). See the challenge [announcement](https://physionet.org/challenge/2001/) for information about the competition, and see [Predicting Onset of Atrial Fibrillation](https://physionet.org/content/afpdb/1.0.0/paf.shtml) for a brief overview of the clinical problem, its significance, and suggestions for further reading on the subject.

### Data Description

The database is divided into a *learning set* (records with names of the form n *\** and p *\**) and a *test set* (records with names of the form t *\**).

The learning set consists of 50 record sets. Each record set contains two 30-minute records with consecutive record names (e.g., p15 and p16), and two 5-minute \`\`continuation'' records with names ending in c (e.g., p15c and p16c). All four records in each record set are excerpts of longer continuous ECG recordings of a single subject; the 50 record sets come from 48 different subjects.

The records with names beginning with p come from subjects who have paroxysmal atrial fibrillation (PAF). The second (even-numbered) record in each pair of 30-minute records contains the ECG immediately preceding an episode of PAF, which can be verified by examining the like-numbered continuation record. Thus, for example, record p16 immediately precedes the episode of PAF in record p16c. The first (odd-numbered) record of the set (for example, record p15) contains 30 minutes of the ECG during a period that is distant from any episode of PAF (there is no PAF during the 45-minute period before the beginning or after the end of the 30-minute record). The corresponding 5-minute continuation record (e.g., record p15c) shows that (at least!) the minutes immediately following the \`\`PAF-distant'' record do not contain PAF. **Note:** Please be aware that a few of the 30-minute records in this group may contain very short bursts of PAF that escaped notice while the learning set was being compiled.

The records with names beginning with n come from subjects who do not have documented atrial fibrillation, either during the period from which the records were excerpted or at any other time. The subjects include healthy controls, patients referred for long-term ambulatory ECG monitoring, and patients in intensive care units.

The test set is similarly constructed of 50 record sets (from 50 different subjects); unlike the learning set, there are no continuation records. The test set records are named t01, t02,... t100. As in the learning set, pairs of consecutively numbered records come from the same long-term ECG recording of a single subject. Approximately half of the record sets in the test set come from subjects with PAF; part 1 of the challenge is to identify these record sets, and part 2 is to identify which record in each pair immediately precedes PAF.

Several files are associated with each record. The files with names of the form *\**.dat contain the digitized ECGs (16 bits per sample, least significant byte first in each pair, 128 samples per signal per second, samples from each channel alternating, nominally 200 A/D units per millivolt). The.hea files are (text) header files that specify the names and formats of the associated signal files; these header files are needed by the software available from this site. The.qrs files are machine-generated (binary) annotation files, provided for the convenience of those who do not wish to use their own QRS detectors. Please note that the.qrs files are unaudited and contain errors. You may use these annotations in uncorrected form if you wish to investigate methods of PAF prediction that are robust with respect to small numbers of QRS detection errors, or you may ignore these annotations entirely and work directly from the signal files.

### Acknowledgements

Special thanks to Steven Swiryn of Northwestern University, who provided many of the recordings excerpted here.

### Release Info

**Update (16 March 2001):** Four of the records in the learning set have been replaced; these are p02, n24, n47, n48. Thanks to Isaac Henry, Christoph Maier, Joseph Mietus, and Juan Millet for their timely and valuable feedback on this database.

---

## Files

Total uncompressed size: 191.4 MB.
