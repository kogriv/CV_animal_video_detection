# CV_animal_video_detection
Overview of the methods used to detect animals in heterogeneous environments on video data 

# Исследование современных подходов к детекции животных на видео

Для погружения в задачу детекции животных на видеоданных и получения понимания о существующих подходах и методах в этой области я провел обзор научных исследований, готовых проектов и доступных датасетов. Поскольку я не являюсь узким специалистом в компьютерном зрении, мне кажется важным сосредоточиться на изучении существующих наработок и лучших решений, разработанных экспертами. Разделив материалы на три категории — научные исследования, проекты и датасеты — я смог глубже разобраться в теме, определить возможные подходы и оценить доступные инструменты.

## Research

Некоторые исследования по теме включают анализ подходов глубокого обучения для детекции животных, изучение видеодатасетов, а также методы многообъектного отслеживания. Вот эти источники:

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

- [Video dataset of sheep activity for animal behavioral analysis via deep learning - PubMed](https://pubmed.ncbi.nlm.nih.gov/38328501/)
- [Video dataset of sheep activity for animal behavioral analysis via deep learning - ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2352340924000015?via%3Dihub)
- [Video Dataset of Sheep Activity (Grazing, Running, Sitting) - Mendeley Data](https://data.mendeley.com/datasets/h5ppwx6fn4/1)
- [Video Dataset of Sheep Activity (Standing and Walking) - Mendeley Data](https://data.mendeley.com/datasets/w65pvb84dg/1)
- [CVB: A Video Dataset of Cattle Visual Behaviors | DeepAI](https://deepai.org/publication/cvb-a-video-dataset-of-cattle-visual-behaviors)
- [dtnguyen0304/sawit: A Small-Sized Animal Wild Image Dataset with Annotations](https://github.com/dtnguyen0304/sawit?tab=readme-ov-file)
- [Search | Kaggle](https://www.kaggle.com/search?q=animal+video+dataset+in%3Adatasets)
- [Version](https://universe.roboflow.com/ai-bd/animals_video2/dataset/1)
- [Machine Learning Datasets | Papers With Code](https://paperswithcode.com/datasets?q=animals&v=lst&o=match&mod=videos&page=1)
- [DeformingThings4D Dataset | Papers With Code](https://paperswithcode.com/dataset/deformingthings4d)
- [TGIF Dataset | Papers With Code](https://paperswithcode.com/dataset/tgif)
- [Animal Kingdom Dataset | Papers With Code](https://paperswithcode.com/dataset/animal-kingdom)
- [Camouflaged Animal Dataset Dataset | Papers With Code](https://paperswithcode.com/dataset/camouflaged-animal-dataset)
- [Causal Motion Segmentation in Moving Camera Videos](http://vis-www.cs.umass.edu/motionSegmentation/)
- [Lindenthal Camera Traps Dataset | Papers With Code](https://paperswithcode.com/dataset/lindenthal-camera-traps)
- [Lindenthal Camera Traps - LILA BC](https://lila.science/datasets/lindenthal-camera-traps/)
- [MGif Dataset | Papers With Code](https://paperswithcode.com/dataset/mgif)
- [TGIF-QA Dataset | Papers With Code](https://paperswithcode.com/dataset/tgif-qa)
- [Dynamic Replica Dataset | Papers With Code](https://paperswithcode.com/dataset/dynamic-replica)
- [3D-POP Dataset | Papers With Code](https://paperswithcode.com/dataset/3d-pop)
- [DogCentric Activity Dataset | Papers With Code](https://paperswithcode.com/dataset/dogcentric-activity)
- [SLT-Net: Implicit Motion Handling for Video Camouflaged Object Detection](https://xueliancheng.github.io/SLT-Net-project/)
- [MoCA-Mask Dataset | Papers With Code](https://paperswithcode.com/dataset/moca-mask)
- [Desert Lion Conservation Camera Traps - LILA BC](https://lila.science/datasets/desert-lion-conservation-camera-traps/)
- [iWildCam 2022 - LILA BC](https://lila.science/datasets/iwildcam-2022/)
- [Data Sets - LILA BC](https://lila.science/datasets)
- [Amur Tiger Re-identification - LILA BC](https://lila.science/datasets/atrw)
- [CaltechCameraTraps | Caltech Camera Trap dataset page](https://beerys.github.io/CaltechCameraTraps/)
- [visipedia/inat_comp: iNaturalist competition details](https://github.com/visipedia/inat_comp?tab=readme-ov-file)
- [visipedia/inat_comp: iNaturalist competition details](https://github.com/visipedia/inat_comp/tree/master)
- [Snapshot Serengeti - LILA BC](https://lila.science/datasets/snapshot-serengeti)


На этих платформах собраны видеодатасеты, предназначенные для анализа поведения животных и детекции в природной среде. Они включают активность различных животных (овцы, крупный рогатый скот, дикие животные) и обеспечивают необходимую разнообразность фона, условий освещения и типов поведения.

# Решение задачи детекции зайцев на видеопотоке: вопросы и ответы

## Задача
Система должна детектировать определенные объекты (зайцев) в видеопотоке, фиксируя их в условиях естественной среды обитания — леса и поля. Основные сложности включают переменное освещение, погодные условия и разнообразие растительности.

---

## Вопрос 1: Выбор алгоритмов детекции
**Оптимальные алгоритмы: современные методы на базе глубокого обучения** — способны справляться с изменяющимися условиями освещения и фона.

### Рекомендуемые алгоритмы:
- **YOLOv5 или SSD**: высокая скорость обработки, подходят для реального времени. Подход: однопроходная детекция объектов ([A Comprehensive Review of Deep Learning Approaches for Animal Detection on Video Data](https://www.researchgate.net/publication/376179857_A_Comprehensive_Review_of_Deep_Learning_Approaches_for_Animal_Detection_on_Video_Data)).

**Недостатки:** YOLO и SSD могут терять точность при детекции небольших объектов на дальнем расстоянии или при плотной растительности, что стоит учитывать при настройке модели.

- **Faster R-CNN и Mask R-CNN**: более высокая точность, возможность сегментации контуров зайцев. Подходят для сложных условий ([MOTHe GUI](https://github.com/tee-lab/MOTHe-GUI)).  

**Недостатки:** Сегментационные сети медленнее и требовательны к вычислительным ресурсам, поэтому их лучше использовать, если скорость не критична или в случаях, когда возможна предобработка кадров перед подачей в систему реального времени.

- **EfficientDet**: эффективен для детекции, экономит вычислительные ресурсы, но требует GPU для реального времени.

### Возможный общий подход к решению:

- Начать с использования YOLOv5 или EfficientDet для детекции объектов в кадре.
- Для устойчивого отслеживания объектов можно дополнить подход Kalman-фильтром или алгоритмом Хангариана для предсказания положения в следующем кадре, как реализовано в MOTHe. Это обеспечит непрерывность отслеживания объектов между кадрами и устранит «дрожание» детекции.
- При необходимости сегментации, если частично видимые объекты или контрастные формы животных создают ложные срабатывания, использовать Mask R-CNN.

### Дополнительные техники:

- Аугментация данных: Добавить вариации (яркость, контраст, шум) к тренировочному набору, чтобы обученная модель могла справляться с изменяющимся освещением и погодными условиями.
- Активное обучение: Постепенно добавлять в обучающий набор сложные для модели кадры, чтобы повысить ее устойчивость к нестандартным условиям.

---

## Вопрос 2: Архитектура модели
Для устойчивой детекции на фоне сложных условий предпочтительны архитектуры, сочетающие пространственные  (формы и контуры объектов) и временные (движение) признаки.

### Рекомендованные архитектуры:
- **YOLOv5 и YOLOv7**: слои свертки, residual-блоки, объединяющий слой для детекции объектов.  
    - Особенности архитектуры: YOLO использует слои свёртки для выделения иерархических признаков и выходной слой, позволяющий делать предсказания по классам и границам объектов в одном проходе. Этот подход делает YOLO быстрым и оптимальным для задач реального времени.
    - Оптимальные слои: Сверточные слои с разными размерами рецептивных полей, residual-блоки для улучшения обучения, а также final output layer, объединяющий детекцию объектов в одном кадре.
- **EfficientDet с BiFPN**: с объединением признаков на разных уровнях для работы с объектами разного размера.
    - Особенности архитектуры: EfficientDet сочетает архитектуры EfficientNet для экстракции признаков с BiFPN (Bidirectional Feature Pyramid Network), что делает его экономичным с точки зрения вычислительных ресурсов и устойчивым к изменяющимся условиям.
    - Оптимальные слои: BiFPN для эффективного объединения признаков на разных уровнях (для работы с объектами разного размера и на разной глубине), scaling layers для адаптации модели под задачи с разными разрешениями изображений.
- **Mask R-CNN**: слой RPN для выделения областей интереса, ROIAlign для точной сегментации.
    - Особенности архитектуры: Mask R-CNN расширяет Faster R-CNN, добавляя слой для сегментации объектов, что позволяет точно выделить границы объектов. Это полезно, если требуется точная идентификация границ зайцев среди растительности.
    - Оптимальные слои: Region Proposal Network (RPN) для нахождения потенциальных регионов объектов, ROIAlign для более точного извлечения признаков из этих регионов и upsampling layers для улучшения разрешения сегментирующей маски.
- **3D-CNN (3D Convolutional Neural Networks)**: позволяют учитывать временные зависимости для более точного распознавания движения.
    - Особенности архитектуры: Использует трехмерные сверточные слои, которые одновременно работают с пространственными и временными данными, что позволяет учитывать движение объектов между кадрами. Подходит для задач отслеживания, где важны как форма, так и движение.
    - Оптимальные слои: 3D сверточные слои и 3D max-pooling слои для объединения временных данных в видеопотоке.
- **ConvLSTM (Convolutional LSTM)**:
    - Особенности архитектуры: ConvLSTM объединяет CNN с Long Short-Term Memory (LSTM), что позволяет захватывать временные зависимости. LSTM помогает запоминать движение и изменения, устойчивы к фоновому шуму, переменам освещения и погодным условиям.
    - Оптимальные слои: Convolutional LSTM для обработки пространственно-временных признаков, обеспечивая детекцию с учетом движения объектов. Использование нескольких ConvLSTM слоев поможет модели лучше различать случайные шумы и реальные перемещения зайцев.

### Оптимальные слои и их применение

- Сверточные слои (Convolutional Layers):
    Используются для экстракции признаков с разных уровней сложности. Начальные слои будут отвечать за общие текстуры и формы, а глубокие слои за более специфические признаки, отличающие зайцев от фона.

- Residual-блоки (Residual Blocks):
    Улучшают качество обучения в глубоких сетях за счет прямой передачи данных между слоями, минимизируя исчезновение градиента. В YOLO и EfficientDet они повышают точность детекции, особенно для мелких объектов.

- BiFPN (Bidirectional Feature Pyramid Network):
    Уникальный слой, который используется в EfficientDet для объединения признаков с разных уровней пирамиды. Подходит для улучшения распознавания объектов на фоне сложных условий, так как позволяет учитывать детали на разных масштабах.

- ROIAlign (Region of Interest Alignment):
    Используется в Mask R-CNN для точного выделения регионов интереса (ROI). Слой выравнивает признаки для более точной сегментации и хорошо подходит для распознавания объектов со сложной структурой или на неструктурированном фоне.

- Kalman Filter / Hungarian Algorithm для постпроцессинга:
    Эти алгоритмы помогают улучшить отслеживание объектов между кадрами, сглаживая траектории зайцев и уменьшая количество пропущенных объектов.

---

## Вопрос 3: Подготовка данных
### Организация сбора и разметки данных:
1. **Сбор данных**: запись видео с камеры, дополнение данными из открытых источников (например, [iNaturalist](https://github.com/visipedia/inat_comp)).
2. **Разметка**: ручная разметка или полуавтоматическая разметка с предобученными моделями (LabelImg, CVAT).
3. **Аугментация данных**: изменение яркости, контраста, добавление шума, что повышает устойчивость к условиям среды.

### Решение сложностей:
- **Сложное освещение и фоновый шум**: аугментация и использование сегментации (Mask R-CNN).
- **Частичное перекрытие объектов**: использование моделей, способных учитывать мелкие детали и сложный фон.

---

## Вопрос 4: Оптимизация и снижение погрешности
### Методы минимизации ложных срабатываний (FP) и пропусков (FN):
1. **Многоуровневая детекция и сегментация**: Mask R-CNN для точных границ, FPN для объектов разного масштаба.
2. **Аугментация и временные признаки**: улучшает распознавание в условиях сложного фона (ConvLSTM, 3D-CNN).
3. **Постобработка**: фильтрация детекции по вероятности, контекстная фильтрация для устранения шума (фильтр Калмана).
4. **Дообучение и активное обучение**: добавление сложных примеров для повышения устойчивости модели.

---

## Вопрос 5: Производительность
### Методы оптимизации:
1. **Легкие архитектуры (YOLOv5-Tiny, EfficientDet)**: подходит для работы на ограниченных ресурсах, в том числе CPU.
2. **Техники оптимизации модели**: квантование, прунинг, distillation (TensorFlow Lite, ONNX).
3. **Оптимизация инференса**: использование TensorRT для GPU, микробатчинг.
4. **Ресурсоэффективная обработка видео**: снижение разрешения, использование ROI для фокусировки на интересующих областях.

### Аппаратные решения:
- **NVIDIA Jetson, Google Coral**: подходят для работы в полевых условиях, поддерживают оптимизации, такие как квантование и ускоренный инференс.

---

## Вопрос 6: Постобработка и интеграция
### Постобработка для повышения качества:
1. **Отслеживание объектов и фильтрация траекторий**: фильтр Калмана, алгоритм Хангариана для точной идентификации.
2. **Контекстуальные фильтры и маски**: исключение зон без объектов интереса, например, небо или дорога.

### Интеграция в видеопоток:
1. **Потоковая обработка**: использование OpenCV или GStreamer, многопоточность для разделения этапов обработки.
2. **Вывод результатов и уведомления**: сохранение результатов в JSON/CSV, интеграция с аналитикой, настройка уведомлений.
3. **Масштабируемость**: динамическая регулировка частоты обработки, использование Docker для гибкого развертывания.

---

Эти подходы и методы обеспечат высокую точность и производительность системы детекции зайцев, учитывая сложные условия среды, и позволят эффективно интегрировать её в реальный видеопоток.


