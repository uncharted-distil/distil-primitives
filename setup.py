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
        #'fastai==1.0.52',
        'joblib',
        'scikit-learn>=0.20.2,<=0.21.3',
        'scipy>=1.2.1,<=1.3.1',
        'numpy>=1.15.4,<=1.17.3',
        'pandas>=0.23.4,<=0.25.2',
        'torch>=1.3.1',
        'torchvision>=0.4.2',
        #'pytorch-pretrained-bert==0.4.0', has print statements on import that break d3m annotation validation
        #'sklearn_pandas==1.8.0',
        'sklearn_pandas @ git+https://github.com/cdbethune/sklearn-pandas.git@c009c3a5a26f883f759cf123c0f5a509b1df013b',
        'tensorflow-gpu==2.0.0',
        'frozendict>=1.2',
        'nose==1.3.7',
        #'python-prctl==1.7', Needs to be installed in the primitive install section
        'fastdtw==0.3.2',
        'networkx==2.4',
        'resampy==0.2.1',
        'soundfile==0.10.2',
        'pillow==6.2.1',
        'sgm @ git+https://github.com/nowfred/sgm.git@v1.0.3#egg=sgm',
        'basenet @ git+https://github.com/bkj/basenet.git@903756540b89809ef458f35257287b937b333417',
        'rescal @ git+https://github.com/cdbethune/rescal.py.git@af2091c1d5521c987edd3be41627b9c563582fe8',
        'pytorch-pretrained-bert @ git+https://github.com/cdbethune/pytorch-pretrained-BERT.git@fb5a42d9de9385b146ba585d6d47ec36d27dcbca#egg=pytorch-pretrained-bert',
        'ShapExplainers @ git+https://github.com/NewKnowledge/shap_explainers.git@8c2e824f3a8bce411898cf26c749549240f3bf9f#egg=ShapExplainers',
        # Can cause errors with pretrained-bert: https://github.com/NVIDIA/apex/issues/156
        #'apex @ git+https://github.com/NVIDIA/apex.git@47e3367fcd6636db6cd549bbb385a6e06a3861d0',
        'torchvggish @ git+https://github.com/harritaylor/torchvggish.git@f5ec66cb05029ddfdb9971f343d79408fac44c70#egg=torchvggish',


    ],
    entry_points={
        'd3m.primitives': [
            'community_detection.community_detection.DistilCommunityDetection = distil.primitives.community_detection:DistilCommunityDetectionPrimitive',
            'link_prediction.link_prediction.DistilLinkPrediction = distil.primitives.link_prediction:DistilLinkPredictionPrimitive',
            'vertex_nomination.vertex_nomination.DistilVertexNomination = distil.primitives.vertex_nomination:DistilVertexNominationPrimitive',
            'data_transformation.load_single_graph.DistilSingleGraphLoader = distil.primitives.load_single_graph:DistilSingleGraphLoaderPrimitive',
            'data_transformation.load_edgelist.DistilEdgeListLoader = distil.primitives.load_edgelist:DistilEdgeListLoaderPrimitive',
            'graph_matching.seeded_graph_matching.DistilSeededGraphMatcher = distil.primitives.seeded_graph_matching:DistilSeededGraphMatchingPrimitive',
            'data_transformation.load_graphs.DistilGraphLoader = distil.primitives.load_graphs:DistilGraphLoaderPrimitive',
            'data_transformation.data_cleaning.DistilReplaceSingletons = distil.primitives.replace_singletons:ReplaceSingletonsPrimitive',
            'data_transformation.imputer.DistilCategoricalImputer = distil.primitives.categorical_imputer:CategoricalImputerPrimitive',
            'data_transformation.data_cleaning.DistilEnrichDates = distil.primitives.enrich_dates:EnrichDatesPrimitive',
            'learner.random_forest.DistilEnsembleForest = distil.primitives.ensemble_forest:EnsembleForestPrimitive',
            'classification.bert_classifier.DistilBertPairClassification = distil.primitives.bert_classifier:BertPairClassificationPrimitive',
            'collaborative_filtering.collaborative_filtering_link_prediction.DistilCollaborativeFiltering = distil.primitives.collaborative_filtering_link_prediction:CollaborativeFilteringPrimitive',
            'data_transformation.one_hot_encoder.DistilOneHotEncoder = distil.primitives.one_hot_encoder:OneHotEncoderPrimitive',
            'data_transformation.encoder.DistilBinaryEncoder = distil.primitives.binary_encoder:BinaryEncoderPrimitive',
            'data_transformation.encoder.DistilTextEncoder = distil.primitives.text_encoder:TextEncoderPrimitive',
            'data_preprocessing.data_cleaning.DistilTimeSeriesFormatter = distil.primitives.time_series_formatter:TimeSeriesFormatterPrimitive',
            'classification.text_classifier.DistilTextClassifier = distil.primitives.text_classifier:TextClassifierPrimitive',
            'feature_extraction.image_transfer.DistilImageTransfer = distil.primitives.image_transfer:ImageTransferPrimitive',
            'feature_extraction.audio_transfer.DistilAudioTransfer = distil.primitives.audio_transfer:AudioTransferPrimitive',
            'data_preprocessing.audio_reader.DistilAudioDatasetLoader = distil.primitives.audio_reader:AudioDatasetLoaderPrimitive',
            'clustering.k_means.DistilKMeans = distil.primitives.k_means:KMeansPrimitive',
            'data_transformation.data_cleaning.OutputDataframe = distil.primitives.output_dataframe:OutputDataframePrimitive',
            'feature_selection.mutual_info_classif.DistilMIRanking = distil.primitives.mi_ranking:MIRankingPrimitive',
            'data_transformation.list_to_dataframe.DistilListEncoder = distil.primitives.list_to_dataframe:ListEncoderPrimitive',
        ],
    }
)
