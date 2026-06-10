---
title: "MIT-BIH Atrial Fibrillation Database v1.0.0"
source: "https://physionet.org/content/afdb/1.0.0/"
author:
  - "[[George Moody]]"
  - "[[Roger Mark]]"
published: 2000-11-04
created: 2026-05-17
description:
tags:
  - "clippings"
---
## MIT-BIH Atrial Fibrillation Database

**,**

Published: Nov. 4, 2000. Version: 1.0.0

---

> [!primary] Primary
> **MIT-BIH Atrial Fibrillation Database expanded** *(Nov. 4, 2000, midnight)*
> 
> Previously unreleased signal files for the [MIT-BIH Atrial Fibrillation Database](https://doi.org/10.13026/C2MW2D) have been added to PhysioNet.

> [!secondary] Secondary
> **When using this resource, please cite the original publication:**
> 
> [Moody GB, Mark RG. A new method for detecting atrial fibrillation using R-R intervals. Computers in Cardiology. 10:227-230 (1983).](http://ecg.mit.edu/george/publications/afib-cinc-1983.pdf)
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

This database includes 25 long-term ECG recordings of human subjects with atrial fibrillation (mostly paroxysmal).

### Data Description

Of these, 23 records include the two ECG signals (in the.dat files); records 00735 and 03665 are represented only by the rhythm (.atr) and unaudited beat (.qrs annotation files.

The individual recordings are each 10 hours in duration, and contain two ECG signals each sampled at 250 samples per second with 12-bit resolution over a range of ±10 millivolts. The original analog recordings were made at Boston's Beth Israel Hospital (now the Beth Israel Deaconess Medical Center) using ambulatory ECG recorders with a typical recording bandwidth of approximately 0.1 Hz to 40 Hz. The rhythm annotation files (with the suffix.atr) were prepared manually; these contain rhythm annotations of types (AFIB (atrial fibrillation), (AFL (atrial flutter), (J (AV junctional rhythm), and (N (used to indicate all other rhythms). (The original rhythm annotation files, still available in the [old](https://physionet.org/content/afdb/1.0.0/old/) directory, used AF, AFL, J, and N to mark these rhythms; the atr annotations in this directory have been revised for consistency with those used for the [MIT-BIH Arrhythmia Database](https://physionet.org/content/afdb/mitdb/).) Beat annotation files (with the suffix.qrs) were prepared using an automated detector and have not been corrected manually. For some records, manually corrected beat annotation files (with the suffix.qrsc) are available. (The.qrs annotations may be useful for studies of methods for automated AF detection, where such methods must be robust with respect to typical QRS detection errors. The.qrsc annotations may be preferred for basic studies of AF itself, where QRS detection errors would be confounding.) Note that in both.qrs and.qrsc files, no distinction is made among beat types (all beats are labelled as if normal).

### Release Info

Until November 2000, only one of the signal files (for record 04936) was available. The original 9-track tapes from 1983 for these records have now been read to produce the other signal files in this directory. In a few cases (04043, 08405, and 08434) isolated data blocks from the original tapes were unreadable. In these cases, the missing data, corresponding to 10.24 seconds for each missing block, have been replaced with a flat segment of samples with amplitudes of zero. See [notes.txt](https://physionet.org/content/afdb/1.0.0/notes.txt) for details.

In March 2014, Henian Xia reported that the.qrs annotations were not aligned with the ECG waveforms in record 07859, and investigation showed that 2569 samples appeared to be missing from the beginning of the record. The original annotation file has been renamed 07859.qrs-, and a realigned version of it (without its initial 14 annotations) is available as 07859.qrs. A correctly aligned, manually reviewed annotation file was also produced at that time, and is now available here as 07859.qrsc.

---

## Files

Total uncompressed size: 605.9 MB.