---
title: "Long Term AF Database v1.0.0"
source: "https://physionet.org/content/ltafdb/1.0.0/"
author:
  - "[[George Moody]]"
  - "[[Steven Swiryn]]"
published: 2001-10-30
created: 2026-06-10
description: "This database includes 84 long-term ECG recordings of subjects with paroxysmal or sustained atrial fibrillation (AF). Each record contains two simultaneously recorded ECG signals digitized at 128 Hz with 12-bit resolution over a 20 mV range; record durations vary but are typically 24 to 25 hours."
tags:
  - "clippings"
---
## Long Term AF Database

**,**

Published: Oct. 30, 2008. Version: 1.0.0

---

> [!primary] Primary
> **New annotations for the Long-Term AF Database** *(July 23, 2012, 6:45 p.m.)*
> 
> A complete set of over 9 million reference beat and rhythm annotations for the Long-Term AF Database has been contributed by MEDICALgorithmics (Warsaw, Poland). The Long-Term AF Database, a collection of 84 long-term ECG recordings (typically 24 to 25 hours each) of subjects with paroxysmal or sustained atrial fibrillation, was contributed to PhysioNet in 2008 by Steven Swiryn and his colleagues at Northwestern University. Michael Tadeusiak of MEDICALgorithmics coordinated the annotation development as a PhysioNet project.
> 
> **Long Term AF Database** *(Oct. 30, 2008, midnight)*
> 
> The [Long Term AF Database](https://physionet.org/content/ltafdb/), a collection of 84 two-lead 24-hour ECG recording of subjects with paroxysmal and sustained atrial fibrillation, is now available.

> [!secondary] Secondary
> **When using this resource, please cite the original publication:**
> 
> [Petrutiu S, Sahakian AV, Swiryn S. Abrupt changes in fibrillatory wave characteristics at the termination of paroxysmal atrial fibrillation in humans. Europace 9:466-470 (2007).](http://europace.oxfordjournals.org/cgi/content/full/9/7/466)
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

This database includes 84 long-term ECG recordings of subjects with paroxysmal or sustained atrial fibrillation (AF). Each record contains two simultaneously recorded ECG signals digitized at 128 Hz with 12-bit resolution over a 20 mV range; record durations vary but are typically 24 to 25 hours.

### Data Description

Two sets of annotations are available here:

- **The qrs annotations** were produced by an automated QRS detector; in these annotation files, all detected beats (including occasional ventricular ectopic beats) are labelled N, detected artifacts are labelled '|', and AF terminations are labelled T.
- **The atr annotations** were obtained by manual review of the output of an automated ECG analysis system (see below); in these annotation files, all detected beats are labelled by type, and rhythm changes are also annotated (see the [summary tables](https://physionet.org/content/ltafdb/1.0.0/tables.shtml) for details).

The [AF Termination Challenge Database](https://physionet.org/physiobank/database/aftdb/) consists of 80 one-minute excerpts of a subset of these records (those numbered 00 through 75). As a guide to selecting the excerpts, “T” were inserted manually in the qrs annotation files of this subset of the records, to mark spontaneous terminations of AF episodes with durations of at least one minute.

The original recordings were collected and contributed to PhysioNet by Steven Swiryn and his colleagues at Northwestern University. They were digitized and automatically annotated at Boston’s Beth Israel Deaconess Medical Center. Steven Swiryn and George Moody annotated the AF terminations.

The atr annotations were developed on PhysioNetWorks and contributed to PhysioNet by MEDICALgorithmics Ltd (Warsaw, Poland). The recordings have been automatically annotated using MEDICALgorithmics’ PocketECG system algorithm (FDA 510(k): K112921) and manually verified by an experienced team of MEDICALgorithmics' ECG technicians. The reference annotation development project was coordinated by Michal Tadeusiak.

Thanks to Mariano Llamedo Soria for reporting an error in the original version of 20.hea, and for providing a correction incorporated in the current version.

### Acknowledgments

In addition, when referencing the atr annotations of this database, please acknowledge their creators (MEDICALgorithmics Ltd, Warsaw, Poland; efforts coordinated by Michal Tadeusiak).

### Further reading

Segments from this database were used in:

Moody GB. [Spontaneous Termination of Atrial Fibrillation: A Challenge from PhysioNet and Computers in Cardiology 2004](http://physionet.org/challenge/2004/challenge-2004.pdf), *Computers in Cardiology* **31**:101-104 (2004).

---

## Files

Total uncompressed size: 3.4 GB.