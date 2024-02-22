from CSR_Synthesizer.synthesizers.ctgan import CTGANSynthesizer
from CSR_Synthesizer.synthesizers.tvae import TVAESynthesizer
from CSR_Synthesizer.synthesizers.smote import SMOTE
from CSR_Synthesizer.synthesizers.CSR_Synthesizer import CSR_Synthesizer
from CSR_Synthesizer.synthesizers.CSR_Synthesizer import CSRE_SynthesizerNrows
from CSR_Synthesizer.synthesizers.CSR_Synthesizer import SRE_Synthesizer
from CSR_Synthesizer.synthesizers.CSR_Synthesizer import SR_Synthesizer
from CSR_Synthesizer.synthesizers.CSR_Synthesizer import CRE_Synthesizer
from CSR_Synthesizer.synthesizers.CSR_Synthesizer import CR_Synthesizer
from CSR_Synthesizer.synthesizers.CSR_Synthesizer import CSE_Synthesizer
from CSR_Synthesizer.synthesizers.CSR_Synthesizer import CS_Synthesizer
from CSR_Synthesizer.synthesizers.CSR_Synthesizer import CR_SynthesizerNrows
from CSR_Synthesizer.synthesizers.CSR_Synthesizer import CRE_SynthesizerNrows
from CSR_Synthesizer.synthesizers.CSR_Synthesizer import CS_SynthesizerNrows
from CSR_Synthesizer.synthesizers.CSR_Synthesizer import CSE_SynthesizerNrows

__all__ = (
    'CTGANSynthesizer',
    'TVAESynthesizer',
    'SMOTE',
    'CSR_Synthesizer',
    'CSRE_SynthesizerNrows',
    'SRE_Synthesizer',
    'SR_Synthesizer',
    'CRE_Synthesizer',
    'CR_Synthesizer',
    'CSE_Synthesizer',
    'CS_Synthesizer',
    'CR_SynthesizerNrows',
    'CRE_SynthesizerNrows',
    'CS_SynthesizerNrows',
    'CSE_SynthesizerNrows'
)

def get_all_synthesizers():
    return {
        name: globals()[name]
        for name in __all__
    }
