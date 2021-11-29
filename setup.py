#
#   Copyright © 2021 Uncharted Software Inc.
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.

from distutils.core import setup

from setuptools import find_packages

import subprocess

try:
    subprocess.run(["pip", "install", "-r", "build_requirements.txt"])
except Exception as e:
    print(e)

with open("version.py") as f:
    exec(f.read())

setup(
    name="distil-primitives",
    version=__version__,
    description="Distil primitives as a single library",
    packages=find_packages(),
    keywords=["d3m_primitive"],
    license="Apache-2.0",
    install_requires=[
        "d3m",  # d3m best-practice moving forward is to remove the version (simplifies updates)
        # shared d3m versions - need to be aligned with core package
        "scikit-learn==0.22.2.post1",
        "scipy==1.4.1",
        "numpy==1.18.2",
        "pandas>=1.1.3",
        "torch>=1.4.0",  # validated up to 1.7.0
        "networkx==2.4",
        "pillow==7.1.2",
        # additional dependencies
        "joblib>=0.13.2",
        "torchvision>=0.5.0",  # validated up to 0.8.0
        #'pytorch-pretrained-bert==0.4.0', has print statements on import that break d3m annotation validation
        #'sklearn_pandas==1.8.0', use fork to address bugs
        "sklearn_pandas @ git+https://github.com/cdbethune/sklearn-pandas.git@c009c3a5a26f883f759cf123c0f5a509b1df013b",
        "frozendict>=1.2",
        # 'nose>=1.3.7', Needs to be installed in the primitive install section
        #'python-prctl==1.7', Needs to be installed in the primitive install section
        "fastdtw>=0.3.2",
        "resampy>=0.2.1",
        "soundfile>=0.10.2",
        "sgm @ git+https://github.com/nowfred/sgm.git@v1.0.3#egg=sgm",
        "basenet @ git+https://github.com/bkj/basenet.git@903756540b89809ef458f35257287b937b333417",
        "rescal @ git+https://github.com/cdbethune/rescal.py.git@af2091c1d5521c987edd3be41627b9c563582fe8",
        "pytorch-pretrained-bert @ git+https://github.com/cdbethune/pytorch-pretrained-BERT.git@fb5a42d9de9385b146ba585d6d47ec36d27dcbca#egg=pytorch-pretrained-bert",
        "scikit-image<=0.17.2",
        "shap>=0.29",
        # Can cause errors with pretrained-bert: https://github.com/NVIDIA/apex/issues/156
        #'apex @ git+https://github.com/NVIDIA/apex.git@47e3367fcd6636db6cd549bbb385a6e06a3861d0',
        "torchvggish @ git+https://github.com/harritaylor/torchvggish.git@f5ec66cb05029ddfdb9971f343d79408fac44c70#egg=torchvggish",
        "nonneg_rescal @ git+https://github.com/bkj/nonneg_rescal.git@7c07617aa025b6417286aba766c6c7b0a3eb6c24#egg=nonneg_rescal",
        "lz4>=3.1.0",
    ],
    extras_require={
        "cpu": ["tensorflow==2.2.0"],
        "gpu": ["tensorflow-gpu==2.2.0"],
    },
    entry_points={
        "d3m.primitives": [
            "community_detection.parser.DistilCommunityDetection = distil.primitives.community_detection:DistilCommunityDetectionPrimitive",
            "link_prediction.link_prediction.DistilLinkPrediction = distil.primitives.link_prediction:DistilLinkPredictionPrimitive",
            "vertex_nomination.seeded_graph_matching.DistilVertexNomination = distil.primitives.vertex_nomination:DistilVertexNominationPrimitive",
            "data_transformation.load_single_graph.DistilSingleGraphLoader = distil.primitives.load_single_graph:DistilSingleGraphLoaderPrimitive",
            "data_transformation.load_edgelist.DistilEdgeListLoader = distil.primitives.load_edgelist:DistilEdgeListLoaderPrimitive",
            "graph_matching.seeded_graph_matching.DistilSeededGraphMatcher = distil.primitives.seeded_graph_matching:DistilSeededGraphMatchingPrimitive",
            "data_transformation.load_graphs.DistilGraphLoader = distil.primitives.load_graphs:DistilGraphLoaderPrimitive",
            "data_transformation.replace_singletons.DistilReplaceSingletons = distil.primitives.replace_singletons:ReplaceSingletonsPrimitive",
            "data_transformation.imputer.DistilCategoricalImputer = distil.primitives.categorical_imputer:CategoricalImputerPrimitive",
            "data_transformation.enrich_dates.DistilEnrichDates = distil.primitives.enrich_dates:EnrichDatesPrimitive",
            "learner.random_forest.DistilEnsembleForest = distil.primitives.ensemble_forest:EnsembleForestPrimitive",
            "classification.bert_classifier.DistilBertPairClassification = distil.primitives.bert_classifier:BertPairClassificationPrimitive",
            "classification.linear_svc.DistilRankedLinearSVC = distil.primitives.ranked_linear_svc:RankedLinearSVCPrimitive",
            "collaborative_filtering.link_prediction.DistilCollaborativeFiltering = distil.primitives.collaborative_filtering_link_prediction:CollaborativeFilteringPrimitive",
            "data_transformation.one_hot_encoder.DistilOneHotEncoder = distil.primitives.one_hot_encoder:OneHotEncoderPrimitive",
            "data_transformation.encoder.DistilBinaryEncoder = distil.primitives.binary_encoder:BinaryEncoderPrimitive",
            "data_transformation.encoder.DistilTextEncoder = distil.primitives.text_encoder:TextEncoderPrimitive",
            "data_transformation.satellite_image_loader.DistilSatelliteImageLoader = distil.primitives.satellite_image_loader:DataFrameSatelliteImageLoaderPrimitive",
            "data_transformation.time_series_formatter.DistilTimeSeriesFormatter = distil.primitives.time_series_formatter:TimeSeriesFormatterPrimitive",
            "classification.text_classifier.DistilTextClassifier = distil.primitives.text_classifier:TextClassifierPrimitive",
            "feature_extraction.image_transfer.DistilImageTransfer = distil.primitives.image_transfer:ImageTransferPrimitive",
            "feature_extraction.audio_transfer.DistilAudioTransfer = distil.primitives.audio_transfer:AudioTransferPrimitive",
            "data_transformation.audio_reader.DistilAudioDatasetLoader = distil.primitives.audio_reader:AudioDatasetLoaderPrimitive",
            "clustering.k_means.DistilKMeans = distil.primitives.k_means:KMeansPrimitive",
            "data_transformation.list_to_dataframe.DistilListEncoder = distil.primitives.list_to_dataframe:ListEncoderPrimitive",
            "data_transformation.column_parser.DistilColumnParser = distil.primitives.column_parser:ColumnParserPrimitive",
        ],
    },
)
