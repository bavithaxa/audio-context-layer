# Audio Context Layer for Audio Question Answering



## Overview



This project explores an interpretable pipeline for answering natural language questions about audio.



Instead of mapping audio directly to an answer, the system uses three stages:



Audio â†’ Audio Perception â†’ Timestamped Audio Context â†’ Question Answering



The main contribution is the intermediate audio context layer. It converts detected sounds into a structured, timestamped JSON representation that can be inspected before the language model reasons over it.



## Problem



Given an audio recording and a natural language question, the system should answer questions such as:



- What sounds are present?

- How many times does a sound occur?

- Which sound happened first?

- How long did an event last?

- Which event lasted longer?

- Could one event be related to another?



The system only uses information represented in the detected audio context and reports when the available evidence is insufficient.



## Dataset



ESC-50 was used as the source audio collection. A smaller synthetic scene dataset was created from it.



- 150 audio scenes

- 100 train scenes

- 25 validation scenes

- 25 test scenes

- Scene duration: 20 to 30 seconds

- 5 to 7 events per scene

- Controlled overlap and repeated events

- 1,200 QA pairs

- 8 questions per scene



Question types:



- Apparent

- Counting

- Temporal

- Duration

- Causal reasoning

- Comparison

- Yes/no



The original ESC-50 audio and generated WAV scenes are not included in the repository. The manifests and generation scripts are included for reproducibility.



## Architecture



1\. \*\*Perception\*\*  

&#x20;  Audio is processed by an audio event detection model.



2\. \*\*Context layer\*\*  

&#x20;  Detected events are converted into timestamped JSON containing labels, timestamps, duration, confidence, and event IDs.



3\. \*\*Question answering\*\*  

&#x20;  A language model receives the structured context and the question and returns an answer with confidence and evidence event IDs.



Example output:



```json

{

&#x20; "answer": "The siren occurred first.",

&#x20; "confidence": 0.92,

&#x20; "evidence\_event\_ids": \["event\_001", "event\_003"]

}

