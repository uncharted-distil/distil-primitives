from distutils.core import setup

from setuptools import find_packages

import subprocess

try:
    subprocess.run(['pip', 'install', '-r', 'build_requirements.txt'])
except Exception as e:
    print(e)

setup(
    name='distil-primitives',
    version='0.1.0',
    description='Distil primitives as a single library',
    packages=find_packages(),
    keywords=['d3m_primitive'],
    install_requires=[
        # Commented out packages are required but handled by
        # upstream versions :/
        #'scikit-learn==0.20.2',
        #'torchvision==0.2.2.post3',
        #'fastai==1.0.52',
        'joblib',
        'scipy==1.2.1',
        'numpy==1.15.4',
        'pandas==0.23.4',
        'sklearn_pandas==1.8.0',
        'torch==1.0.1.post2',
        #'pytorch-pretrained-bert==0.4.0', has print statements on import that break d3m annotation validation
        'sklearn_pandas==1.8.0',
        'tensorflow-gpu==1.12.2',
        'frozendict>=1.2',
        'nose==1.3.7',
        #'python-prctl==1.7', Needs to be installed in the primitive install section
        'fastdtw==0.3.2',
        'networkx==2.2.0',
        'resampy==0.2.1',
        'soundfile==0.10.2',
        'sgm @ git+https://github.com/nowfred/sgm.git@v1.0.3#egg=sgm',
        'basenet @ git+https://github.com/bkj/basenet.git@903756540b89809ef458f35257287b937b333417',
        'rescal @ git+https://github.com/cdbethune/rescal.py.git@af2091c1d5521c987edd3be41627b9c563582fe8',
        'pytorch-pretrained-bert @ git+https://github.com/cdbethune/pytorch-pretrained-BERT.git@6639b53a34760208700127e24fbbae93fb03429a'
        # Can cause errors with pretrained-bert: https://github.com/NVIDIA/apex/issues/156
        #'apex @ git+https://github.com/NVIDIA/apex.git@47e3367fcd6636db6cd549bbb385a6e06a3861d0'

    ],
    entry_points={
        'd3m.primitives': [
            'data_transformation.community_detection.DistilCommunityDetection = distil.primitives.community_detection:DistilCommunityDetectionPrimitive',
            'data_transformation.link_prediction.DistilLinkPrediction = distil.primitives.link_prediction:DistilLinkPredictionPrimitive',
            'data_transformation.vertex_nomination.DistilVertexNomination = distil.primitives.vertex_nomination:DistilVertexNominationPrimitive',
            'data_transformation.load_single_graph.DistilSingleGraphLoader = distil.primitives.load_single_graph:DistilSingleGraphLoaderPrimitive',
            'data_transformation.seeded_graph_matcher.DistilSeededGraphMatcher = distil.primitives.seeded_graph_matcher:DistilSeededGraphMatchingPrimitive',
            'data_transformation.load_graphs.DistilGraphLoader = distil.primitives.load_graphs:DistilGraphLoaderPrimitive',
            'data_transformation.data_cleaning.DistilReplaceSingletons = distil.primitives.replace_singletons:ReplaceSingletonsPrimitive',
            'data_transformation.imputer.DistilCategoricalImputer = distil.primitives.categorical_imputer:CategoricalImputerPrimitive',
            'data_transformation.data_cleaning.DistilEnrichDates = distil.primitives.enrich_dates:EnrichDatesPrimitive',
            'learner.random_forest.DistilEnsembleForest = distil.primitives.ensemble_forest:EnsembleForestPrimitive',
            'classification.bert_classifier.DistilBertPairClassification = distil.primitives.bert_classification:BertPairClassificationPrimitive',
            'learner.collaborative_filtering_link_prediction.DistilCollaborativeFiltering = distil.primitives.collaborative_filtering:CollaborativeFilteringPrimitive',
            'data_transformation.one_hot_encoder.DistilOneHotEncoder = distil.primitives.one_hot_encoder:OneHotEncoderPrimitive',
            'data_transformation.encoder.DistilBinaryEncoder = distil.primitives.binary_encoder:BinaryEncoderPrimitive',
            'data_transformation.encoder.DistilTextEncoder = distil.primitives.text_encoder:TextEncoderPrimitive',
            'data_transformation.missing_indicator.DistilMissingIndicator = distil.primitives.missing_indicator:MissingIndicatorPrimitive',
            'data_transformation.data_cleaning.DistilRaggedDatasetLoader = distil.primitives.ragged_dataset_loader:RaggedDatasetLoaderPrimitive',
            'data_transformation.data_cleaning.DistilTimeSeriesReshaper = distil.primitives.timeseries_reshaper:TimeSeriesReshaperPrimitive',
            'learner.random_forest.DistilTimeSeriesNeighboursPrimitive = distil.primitives.timeseries_neighbours:TimeSeriesNeighboursPrimitive',
            'learner.text_classifier.DistilTextClassifier = distil.primitives.text_classifier:TextClassifierPrimitive',
            'feature_extraction.image_transfer.DistilImageTransfer = distil.primitives.image_transfer:ImageTransferPrimitive',
            'feature_extraction.audio_transfer.DistilAudioTransfer = distil.primitives.audio_transfer:AudioTransferPrimitive',
            'data_preprocessing.audio_loader.DistilAudioDatasetLoader = distil.primitives.audio_loader:AudioDatasetLoaderPrimitive',
            'clustering.k_means.DistilKMeans = distil.primitives.k_means:KMeansPrimitive'
        ],
    }
)
