# CV_animal_video_detection
Overview of the methods used to detect animals in heterogeneous environments on video data 

# Исследование современных подходов к детекции животных на видео

Для глубокого понимания текущего состояния методов детекции животных на видеоданных в условиях естественной среды была проведена обзорная работа по литературе, проектам и доступным датасетам. Все найденные материалы я разделил на три основные категории: **научные исследования**, **проекты**, и **датасеты**.

## Research

Основные научные исследования по теме включают анализ подходов глубокого обучения для детекции животных, изучение видеодатасетов, а также методы многообъектного отслеживания. Вот ключевые источники:

- [A Comprehensive Review of Deep Learning Approaches for Animal Detection on Video Data](https://www.researchgate.net/publication/376179857_A_Comprehensive_Review_of_Deep_Learning_Approaches_for_Animal_Detection_on_Video_Data)
- [A Video Dataset of Cattle Visual Behaviors](https://arxiv.org/pdf/2305.16555v1)
- [Multi-Object Tracking in Heterogeneous environments (MOTHe) for animal video recordings | PeerJ](https://peerj.com/articles/15573/)
- [MOTHe: Multi-Object Tracking in Heterogeneous environments - bioRxiv](https://www.biorxiv.org/content/10.1101/2020.01.10.899989v2.full)
- [MOTHe GUI на GitHub](https://github.com/tee-lab/MOTHe-GUI)

Эти исследования охватывают как теоретические аспекты использования глубоких нейронных сетей в задачах детекции животных, так и практические примеры с конкретными видеодатасетами. В частности, статья по **MOTHe** предлагает эффективные методы для отслеживания животных на сложных фонах в условиях естественной среды.

## Projects

Для практического применения в задачах детекции животных были найдены несколько проектов, реализующих модели на основе нейронных сетей. Эти проекты предлагают готовые решения для распознавания различных животных с помощью TensorFlow и других инструментов.

- [Animal Detection Using Tensorflow Object Detection API](https://medium.com/@mdtausifc/animal-detection-using-tensorflow-object-detection-api-859d6bd368dc) — статья с Medium
- [CMDTausif/Animal-Detection-Using-Tensorflow-Object-Detection-API на GitHub](https://github.com/CMDTausif/Animal-Detection-Using-Tensorflow-Object-Detection-API?source=post_page-----859d6bd368dc--------------------------------)
- [Animal detection using neural network in TensorFlow | Updates-Matter](https://stackforgeeks.com/blog/animal-detection-using-neural-network-in-tensorflow)
- [burnpiro/farm-animal-tracking: Farm Animal Tracking (FAT)](https://github.com/burnpiro/farm-animal-tracking)
- [iNaturalist - датасеты животных](https://github.com/inaturalist)

Эти проекты ориентированы на детекцию животных с использованием библиотек TensorFlow и OpenCV. Реализованные подходы позволяют эффективно обрабатывать видеопотоки и обучать модели для различных задач распознавания.

## Datasets

Для создания и тестирования моделей в задачи детекции животных критически важно использование качественных видеодатасетов, охватывающих различные виды животных и условия съемки. Вот некоторые из самых значимых и обширных видеодатасетов:

- [Video dataset of sheep activity for animal behavioral analysis via deep learning - PubMed](https://pubmed.ncbi.nlm.nih.gov/38328501/)
- [Video Dataset of Sheep Activity (Grazing, Running, Sitting) - Mendeley Data](https://data.mendeley.com/datasets/h5ppwx6fn4/1)
- [CVB: A Video Dataset of Cattle Visual Behaviors | DeepAI](https://deepai.org/publication/cvb-a-video-dataset-of-cattle-visual-behaviors)
- [iWildCam 2022 - LILA BC](https://lila.science/datasets/iwildcam-2022/)
- [Snapshot Serengeti - LILA BC](https://lila.science/datasets/snapshot-serengeti)
- [Caltech Camera Traps](https://beerys.github.io/CaltechCameraTraps/)

На этих платформах собраны видеодатасеты, предназначенные для анализа поведения животных и детекции в природной среде. Они включают активность различных животных (овцы, крупный рогатый скот, дикие животные) и обеспечивают необходимую разнообразность фона, условий освещения и типов поведения.


