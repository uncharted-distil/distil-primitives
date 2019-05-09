from distutils.core import setup

from setuptools import find_packages

setup(
    name='DistilPrimitives',
    version='0.1.0',
    description='Distil primitives as a single library (temporary)',
    packages=find_packages(),
    keywords=['d3m_primitive'],
    install_requires=[
        'scipy==1.2.0',
        'scikit-learn==0.20.2',
        'numpy==1.15.4',
        'pandas==0.23.4',
        'sklearn_pandas==1.8.0',
        'torch==1.0.0',
        'torchvision==0.2.2.post3',
        'pytorch-pretrained-bert==0.4.0',
        'sklearn_pandas==1.8.0',
        'tensorflow-gpu==1.12.0',
        'fastai==1.0.52',
        'frozendict>=1.2',
        'cython==0.29.3',
        'nose==1.3.7',
        'joblib==0.13.0',
        'fastdtw==0.3.2',
        'networkx==2.2.0',
        'resampy==0.2.1',
        'soundfile==0.10.2',
        'sgm @ git+https://github.com/nowfred/sgm.git#egg=sgm',
        'common_primitives @ git+https://gitlab.com/datadrivendiscovery/common-primitives.git#egg=common_primitives',
        'basenet @ git+https://github.com/bkj/basenet.git@903756540b89809ef458f35257287b937b333417',
        'rescal @ git+https://github.com/mnick/rescal.py.git@69dddaa9157fc7bd24d5d7ecf0308cc412403c17'
        'd3m==2019.4.4',

    ],
    entry_points={
        'd3m.primitives': [
            'data_transformation.link_prediction.ExlineLinkPrediction = exline.primitives.link_prediction:ExlineLinkPredictionPrimitive',
            'data_transformation.vertex_nomination.ExlineVertexNomination = exline.primitives.vertex_nomination:ExlineVertexNominationPrimitive',
            'data_transformation.load_single_graph.ExlineSingleGraphLoader = exline.primitives.load_single_graph:ExlineSingleGraphLoaderPrimitive',
            'data_transformation.seeded_graph_matcher.ExlineSeededGraphMatcher = exline.primitives.seeded_graph_matcher:ExlineSeededGraphMatchingPrimitive',
            'data_transformation.load_graphs.ExlineGraphLoader = exline.primitives.load_graphs:ExlineGraphLoaderPrimitive',
            'data_transformation.imputer.ExlineSimpleImputer = exline.primitives.simple_imputer:SimpleImputerPrimitive',
            'data_transformation.data_cleaning.ExlineReplaceSingletons = exline.primitives.replace_singletons:ReplaceSingletonsPrimitive',
            'data_transformation.imputer.ExlineCategoricalImputer = exline.primitives.categorical_imputer:CategoricalImputerPrimitive',
            'data_transformation.data_cleaning.ExlineEnrichDates = exline.primitives.enrich_dates:EnrichDatesPrimitive',
            'learner.random_forest.ExlineEnsembleForest = exline.primitives.ensemble_forest:EnsembleForestPrimitive',
            'learner.random_forest.ExlineBertClassification = exline.primitives.bert_classification:BertClassificationPrimitive',
            'learner.random_forest.ExlineCollaborativeFiltering = exline.primitives.collaborative_filtering:CollaborativeFilteringPrimitive',
            'data_transformation.standard_scaler.ExlineStandardScaler = exline.primitives.standard_scaler:StandardScalerPrimitive',
            'data_transformation.one_hot_encoder.ExlineOneHotEncoder = exline.primitives.one_hot_encoder:OneHotEncoderPrimitive',
            'data_transformation.encoder.ExlineBinaryEncoder = exline.primitives.binary_encoder:BinaryEncoderPrimitive',
            'data_transformation.encoder.ExlineTextEncoder = exline.primitives.text_encoder:TextEncoderPrimitive',
            'data_transformation.column_parser.ExlineSimpleColumnParser = exline.primitives.simple_column_parser:SimpleColumnParserPrimitive',
            'data_transformation.missing_indicator.ExlineMissingIndicator = exline.primitives.missing_indicator:MissingIndicatorPrimitive',
            'data_transformation.data_cleaning.ExlineZeroColumnRemover = exline.primitives.zero_column_remover:ZeroColumnRemoverPrimitive',
            'data_transformation.data_cleaning.ExlineRaggedDatasetLoader = exline.primitives.ragged_dataset_loader:RaggedDatasetLoaderPrimitive',
            'data_transformation.data_cleaning.ExlineTimeSeriesReshaper = exline.primitives.timeseries_reshaper:TimeSeriesReshaperPrimitive',
            'learner.random_forest.ExlineTimeSeriesNeighboursPrimitive = exline.primitives.timeseries_neighbours:TimeSeriesNeighboursPrimitive',
            'data_transformation.encoder.ExlineTextClassifier = exline.primitives.text_classifier:TextClassifierPrimitive',
            'data_transformation.encoder.ExlineImageTransfer = exline.primitives.image_transfer:ImageTransferPrimitive',
            'data_transformation.encoder.ExlineAudioTransfer = exline.primitives.audio_transfer:AudioTransferPrimitive',
            'data_transformation.encoder.ExlineTextReader = exline.primitives.text_reader:TextReaderPrimitive',
            'clustering.k_means.ExlineKMeans = exline.primitives.k_means:KMeansPrimitive'
        ],
    }
)
