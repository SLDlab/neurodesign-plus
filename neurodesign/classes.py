from __future__ import annotations

import copy
import math
import shutil
import warnings
import random
import zipfile
from collections import Counter
from io import BytesIO
from pathlib import Path

import numpy as np
import scipy
import scipy.linalg
import sklearn.cluster
from numpy import transpose as t
from rich import print
from rich.progress import (
    BarColumn,
    MofNCompleteColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)
from scipy.special import gamma

from neurodesign import generate, report


def progress_bar(text: str, color: str = "green") -> Progress:
    """Return a rich progress bar instance."""
    return Progress(
        TextColumn(f"[{color}]{text}"),
        SpinnerColumn("dots"),
        TimeElapsedColumn(),
        BarColumn(),
        MofNCompleteColumn(),
        TaskProgressColumn(),
        TimeRemainingColumn(),
    )


class Design:
    """
    This class represents an experimental design for an fMRI experiment.

    :param order: The stimulus order.
    :type order: list of integers
    :param ITI: The ITI's between all stimuli.
    :type ITI: list of floats
    :param experiment: The experimental setup.
    :type experiment: experiment object
    :param onsets: The onsets of all stimuli.
    :type onsets: list of floats
    """

    def __init__(self, order, ITI, experiment, onsets=None):

        self.order = order
        self.ITI = ITI
        self.onsets = onsets
        self.Fe = 0
        self.Fd = 0

        self.experiment = experiment

        # assert whether design is valid
        if self.experiment.ITI is not None:
            self.ITI = self.experiment.ITI
        if len(self.ITI) != experiment.n_trials:
            raise ValueError("length of design (ITI's) does not comply with experiment")
        if len(self.order) != experiment.n_trials:
            raise ValueError("length of design (orders) does not comply with experiment")

    def check_maxrep(self, maxrep):
        """Check whether design does not exceed maximum repeats within design.

        :param maxrep: How many times should a stimulus maximally be repeated.
        :type maxrep: integer
        :returns repcheck: Boolean indicating maximum repeats are respected
        """
        for stim in range(self.experiment.n_stimuli):
            repcheck = "".join(str(e) for e in [stim] * maxrep) not in "".join(
                str(e) for e in self.order
            )
            if not repcheck:
                break

        return repcheck

    def check_hardprob(self):
        """Check whether frequencies of stimuli 
           are **exactly** the prespecified frequencies.

        :returns probcheck: Boolean indicating probabilities are respected
        """
        obscnt = Counter(self.order).values()
        obsprob = np.round(obscnt / np.sum(obscnt), decimals=2)
        if len(self.experiment.P) != len(obsprob):
            return False

        close = np.isclose(np.array(self.experiment.P), np.array(obsprob), atol=0.001)
        return np.sum(close) == len(obsprob)

    def crossover(self, other, seed=1234):
        """Crossover design with other design and create offspring.

        :param other: The design with which the design will be mixed
        :type other: design object
        :param seed: The seed with which the change point will be sampled.
        :type seed: integer or None
        :returns offspring: List of two offspring designs.
        """
        # check whether designs are compatible
        assert len(self.order) == len(other.order)

        np.random.seed(seed)
        changepoint = np.random.choice(len(self.order), 1)[0]

        offspringorder1 = None
        offspringorder2 = None

        #Making sure the order doesn't change for the crossover
        if self.experiment.order_fixed:
            offspringorder1 = self.order
            offspringorder2 = other.order
        else: 
            offspringorder1 = list(self.order)[:changepoint] + list(other.order)[changepoint:]
            offspringorder2 = list(other.order)[:changepoint] + list(self.order)[changepoint:]

        offspring1 = Design(
            order=offspringorder1, ITI=self.ITI, experiment=self.experiment
        )
        offspring2 = Design(
            order=offspringorder2, ITI=other.ITI, experiment=self.experiment
        )

        return [offspring1, offspring2]

    def mutation(self, q, seed=1234):
        """Mutate q% of the stimuli with another stimulus.

        :param q: The percentage of stimuli that should be mutated
        :type q: float
        :param seed: The seed with which the mutation points are sampled.
        :type seed: integer or None
        :returns mutated: Mutated design
        """
        np.random.seed(seed)
        mut_ind = np.random.choice(
            len(self.order), int(len(self.order) * q), replace=False
        )
        mutated = copy.copy(self.order)

        if not self.experiment.order_fixed:
            for mut in mut_ind:
                np.random.seed(seed)
                mut_stim = np.random.choice(self.experiment.n_stimuli, 1, replace=True)[0]
                mutated[mut] = mut_stim

        offspring = Design(order=mutated, ITI=self.ITI, experiment=self.experiment)

        return offspring


    def designmatrix(self):
        """Expand from order of stimuli to a fMRI timeseries."""
        # ITIs to onsets
        orderli = list(self.order)
        ITIli = list(self.ITI)
        if self.experiment.restnum > 0:
            if self.experiment.all_stim_durations is None: 
                # Old Package Code
                ITIli = [
                    y + self.experiment.trial_duration if not x == "R" else y
                    for x, y in zip(orderli, ITIli)
                ]
                onsets = np.cumsum(ITIli) - self.experiment.trial_duration

                self.onsets = [y for x, y in zip(orderli, onsets) if not x == "R"]
            else: 
                for x in np.arange(0, self.experiment.n_trials, self.experiment.restnum)[1:][
                    ::-1
                ]:
                    orderli.insert(x, "R")
                    ITIli.insert(x, self.experiment.restdur)
                
                #Modified by Atharv Umap
                #Calculate the ITI_li based on the rest numbers and varied trial duration
                ITIli_new = [] 
                onsets = [] 
                for i, (x, y) in enumerate(zip(orderli, ITIli)):
                    if not x == "R":
                        ITIli_new.append(y+self.experiment.all_stim_durations[i])
                    else: 
                        ITIli_new.append(y)

                ITIli_cumsum = np.cumsum(ITIli_new)
                # Subtracts the cumulative sum by the respsective trial durations while excluding the rest trials. 
                for i, (x, y) in enumerate(zip(orderli, ITIli_cumsum)):
                    if not x == "R":
                        onsets.append(y - self.experiment.all_stim_durations[i])

                self.onsets = onsets
        else:
            if self.experiment.all_stim_durations is None: 
                #Old Package Code 
                ITIli = np.array(self.ITI) + self.experiment.trial_duration
                self.onsets = np.cumsum(ITIli) - self.experiment.trial_duration
            else: 
                #Modified by Atharv Umap
                ITIli_new = [y+x for x, y in zip(self.experiment.all_stim_durations, ITIli)]
                ITIli_cumsum = np.cumsum(ITIli_new)
                
                onsets_temp = [] 
                for x, y in zip(self.experiment.all_stim_durations, list(ITIli_cumsum)):
                    onsets_temp.append(y - x)
                self.onsets = onsets_temp

        stimonsets = [x + self.experiment.t_pre for x in self.onsets]

        # round onsets to resolution
        self.ITI, x = _round_to_resolution(self.ITI, self.experiment.resolution)
        onsetX, XindStim = _round_to_resolution(stimonsets, self.experiment.resolution)

        if self.experiment.all_stim_durations is None: 
            stim_duration_tp = int(self.experiment.stim_duration / self.experiment.resolution)

            # find indices in resolution scale of stimuli
            assert np.max(XindStim) <= self.experiment.n_tp
            assert np.max(XindStim) + stim_duration_tp <= self.experiment.n_tp

            # create design matrix in resolution scale (=deltasM in Kao toolbox)
            X_X = np.zeros([self.experiment.n_tp, self.experiment.n_stimuli])
            for stimulus in range(self.experiment.n_stimuli):
                for dur in range(stim_duration_tp):
                    X_X[np.array(XindStim) + dur, int(stimulus)] = [
                        1 if z == stimulus else 0 for z in self.order
                    ]
        else: 
            stim_duration_tp = self.experiment.all_stim_durations / self.experiment.resolution
            stim_duration_tp = [int(x) for x in stim_duration_tp] 

            # find indices in resolution scale of stimuli
            try:
                assert np.max(XindStim) <= self.experiment.n_tp

                #Modified to add multiple different stimuli_duration
                assert np.max(XindStim) + stim_duration_tp[-1] <= self.experiment.n_tp
            except Exception as e:
                print(XindStim)
                print(stim_duration_tp[-1])
                print(self.experiment.n_tp)
                assert np.max(XindStim) <= self.experiment.n_tp
                assert np.max(XindStim) + stim_duration_tp[-1] <= self.experiment.n_tp

            X_X = np.zeros([self.experiment.n_tp, self.experiment.n_stimuli])
            #indexing and traversing through the order
            for i, stim in enumerate(self.order):
                #the current onset
                onset = XindStim[i]
                #the duration between the onsets
                dur = stim_duration_tp[i]
                #labeling the binary values of the indices with 
                for j in range(dur):
                    t_idx = onset + j
                    if t_idx < self.experiment.n_tp:
                        X_X[t_idx, stim] = 1
    
        # deconvolved matrix in resolution units
        deconvM = np.zeros(
            [
                self.experiment.n_tp,
                int(self.experiment.laghrf * self.experiment.n_stimuli),
            ]
        )
        for stim in range(self.experiment.n_stimuli):
            for j in range(int(self.experiment.laghrf)):
                deconvM[j:, self.experiment.laghrf * stim + j] = X_X[
                    : (self.experiment.n_tp - j), stim
                ]

        # downsample and whiten deconvM
        idxX = [
            int(x)
            for x in np.arange(
                0, self.experiment.n_tp, self.experiment.TR / self.experiment.resolution
            )
        ]

        if len(idxX) - self.experiment.white.shape[0] == 1:
            idxX = idxX[: self.experiment.white.shape[0]]

        deconvMdown = deconvM[idxX, :]
        Xwhite = np.dot(np.dot(t(deconvMdown), self.experiment.white), deconvMdown)

        # convolve design matrix
        X_Z = np.zeros([self.experiment.n_tp, self.experiment.n_stimuli])
        for stim in range(self.experiment.n_stimuli):
            X_Z[:, stim] = deconvM[
                :, (stim * self.experiment.laghrf) : ((stim + 1) * self.experiment.laghrf)
            ].dot(self.experiment.basishrf)

        X_Z = X_Z[idxX, :]
        X_X = X_X[idxX, :]
        Zwhite = t(X_Z) * self.experiment.white * X_Z

        self.X = Xwhite
        self.Z = Zwhite
        self.Xconv = X_Z
        self.Xnonconv = X_X
        self.CX = self.experiment.CX
        self.C = self.experiment.C

        return self



    def FeCalc(self, Aoptimality=True):
        """
        Compute estimation efficiency.

        :param Aoptimality: Kind of optimality to optimize, A- or D-optimality
        :type Aoptimality: boolean
        """
        try:
            invM = scipy.linalg.inv(self.X)
        except scipy.linalg.LinAlgError:
            try:
                invM = scipy.linalg.pinv(self.X)
            except np.linalg.linalg.LinAlgError:
                invM = np.nan

        invM = np.array(invM)
        st1 = np.dot(self.CX, invM)
        CMC = np.dot(st1, t(self.CX))
        if Aoptimality is True:
            self.Fe = float(self.CX.shape[0] / np.matrix.trace(CMC))
        else:
            self.Fe = float(np.linalg.det(CMC) ** (-1 / len(self.C)))
        self.Fe = self.Fe / self.experiment.FeMax
        return self

    def FdCalc(self, Aoptimality=True):
        """
        Compute detection power.

        :param Aoptimality: Kind of optimality to optimize: A- or D-optimality
        :type Aoptimality: boolean
        """
        try:
            invM = scipy.linalg.inv(self.Z)
        except scipy.linalg.LinAlgError:
            try:
                invM = scipy.linalg.pinv(self.Z)
            except np.linalg.linalg.LinAlgError:
                invM = np.nan

        invM = np.array(invM)
        CMC = np.matrix(self.C) * invM * np.matrix(t(self.C))
        if Aoptimality is True:
            self.Fd = float(len(self.C) / np.matrix.trace(CMC))
        else:
            self.Fd = float(np.linalg.det(CMC) ** (-1 / len(self.C)))
        self.Fd = self.Fd / self.experiment.FdMax
        return self

    def FcCalc(self, confoundorder=3):
        """
        Compute confounding efficiency.

        :param confoundorder: To what order should confounding be protected
        :type confoundorder: integer
        """
        Q = np.zeros(
            [self.experiment.n_stimuli, self.experiment.n_stimuli, confoundorder]
        )
        for n in range(len(self.order)):
            for r in np.arange(1, confoundorder + 1):
                if n > (r - 1):
                    Q[self.order[n], self.order[n - r], r - 1] += 1
        Qexp = np.zeros(
            [self.experiment.n_stimuli, self.experiment.n_stimuli, confoundorder]
        )
        for si in range(self.experiment.n_stimuli):
            for sj in range(self.experiment.n_stimuli):
                for r in np.arange(1, confoundorder + 1):
                    Qexp[si, sj, r - 1] = (
                        self.experiment.P[si]
                        * self.experiment.P[sj]
                        * (self.experiment.n_trials + 1)
                    )
        Qmatch = np.sum(abs(Q - Qexp))
        self.Fc = Qmatch
        self.Fc = 1 - self.Fc / self.experiment.FcMax
        return self

    def FfCalc(self):
        """Compute efficiency of frequencies."""
        trialcount = Counter(self.order)
        Pobs = [trialcount[x] for x in range(self.experiment.n_stimuli)]
        self.Ff = np.sum(
            abs(
                np.array(Pobs)
                - np.array(self.experiment.n_trials * np.array(self.experiment.P))
            )
        )
        self.Ff = 1 - self.Ff / self.experiment.FfMax
        return self

    def FCalc(self, weights, Aoptimality=True, confoundorder=3):
        """
        Compute weighted average of efficiencies.

        :param weights: Weights given to each of the efficiency metrics in this order:
                        Estimation, Detection, Frequencies, Confounders.
        :type weights: list of floats
        """
        if weights[0] > 0:
            self.FeCalc(Aoptimality)
        if weights[1] > 0:
            self.FdCalc(Aoptimality)
        self.FfCalc()
        self.FcCalc(confoundorder)
        matr = np.array([self.Fe, self.Fd, self.Ff, self.Fc])
        self.F = np.sum(weights * matr)
        return self


class Experiment:
    """
    This class represents an fMRI experiment.

    :param TR: The repetition time.
    :type  TR: float

    :param P: The probabilities of each trialtype.
    :type  P: ndarray

    :param C: The contrast matrix.  Example: np.array([[1,-1,0],[0,1,-1]])
    :type  C: ndarray

    :param rho: AR(1) correlation coefficient
    :type  rho: float

    :param n_stimuli: The number of stimuli (or conditions) in the experiment.
    :type  n_stimuli: integer

    :param n_trials: The number of trials in the experiment.
                     Either specify n_trials **or** duration.
    :type  n_trials: integer

    :param duration: The total duration (seconds) of the experiment.
                     Either specify n_trials **or** duration.
    :type  duration: float

    :param resolution: the maximum resolution of design matrix
    :type  resolution: float

    :param stim_duration: duration (seconds) of stimulus
    :type  stim_duration: float

    :param t_pre: duration (seconds) of trial part before stimulus presentation
                  (eg. fixation cross)
    :type  t_pre: float

    :param t_post: duration (seconds) of trial part after stimulus presentation
    :type  t_post: float

    :param maxrep: maximum number of repetitions
    :type  maxrep: integer or None

    :param hardprob: can the probabilities differ from the nominal value?
    :type  hardprob: boolean

    :param confoundorder: The order to which confounding is controlled.
    :type  confoundorder: integer

    :param restnum: Number of trials between restblocks
    :type  restnum: integer

    :param restdur: duration (seconds) of the rest blocks
    :type  restdur: float

    :param ITImodel: Which model to sample from.
                     Possibilities: "fixed","uniform","exponential"
    :type  ITImodel: string

    :param ITImin: The minimum ITI (required with "uniform" or "exponential")
    :type  ITImin: float

    :param ITImean: The mean ITI (required with "fixed" or "exponential")
    :type  ITImean: float

    :param ITImax: The max ITI (required with "uniform" or "exponential")
    :type  ITImax: float

    """

    def __init__(
        self,
        TR: float,
        P,
        C,
        rho: float,
        stim_duration,
        n_stimuli: int,
        stimuli_durations=None, 
        ITI = None, #Adding a custom ITI
        conditional_ITI=None, #Adding a new input for ITI's that can be controlled by the user.
        ITImodel=None,
        ITImin=None,
        ITImax=None,
        ITImean=None,
        restnum=0,
        restdur=0,
        t_pre=0,
        t_post=0,
        n_trials: int | None = None,
        duration=None,
        resolution=0.1,
        FeMax=1,
        FdMax=1,
        FcMax=1,
        FfMax=1,
        maxrep=None,
        hardprob=False,
        confoundorder=3,
        order = None, 
        order_probabilities = None,
        order_keys = None, 
        order_length = None, 
        order_fixed = False,
        trial_max = None
    ):
        self.TR = TR
        self.P = P
        self.C = np.array(C)
        self.rho = rho
        self.n_stimuli = n_stimuli
        self.t_pre = t_pre
        self.t_post = t_post
        self.n_trials = n_trials
        self.duration = duration
        self.resolution = resolution
        self.stim_duration = stim_duration


        #We will calculate all stimuli durations based on the given values
        self.all_stim_durations = None
        
        #Modification
        #Working wiht multiple stimuli durations 
        if stimuli_durations is not None: 
            assert len(stimuli_durations) == n_stimuli, "Must specify a duration for each stimulus"
            assert trial_max is not None, "Must provide a trial_max given stimuli_durations"
            self.trial_max = trial_max
            self.stimuli_durations = stimuli_durations 
        else:
            self.trial_max = stim_duration
            self.stimuli_durations = None
        
        self.order_probabilities = order_probabilities
        self.order_keys = order_keys
        self.order_length = order_length
        self.order_fixed = order_fixed

        #Adding the custom order 
        if order is not None:
            self.order = order 
            self.order_fixed = True
        elif order_probabilities is not None:  
            order = self.sample_from_probabilities(order_probabilities, order_keys, order_length)


        self.maxrep = maxrep
        self.hardprob = hardprob
        self.confoundorder = confoundorder


        #Addding a conditional ITI where you can have a number of stimulus and it can have different ITI's
        if conditional_ITI is not None: 
            self.conditional_ITI = conditional_ITI
        else: 
            self.conditional_ITI = None
        # else: 
        #     self.conditional_ITI = {
        #         (0, 1): {"model": "exponential", "mean": 2, "min": 1},
        #         (1, 2): {"model": "fixed", "mean": 4},
        #         "default": {"model": "exponential", "mean": 3, "min": 1}
        #     }


        self.ITImodel = ITImodel
        self.ITImin = ITImin
        self.ITImean = ITImean
        self.ITImax = ITImax
        self.ITIlam = None

        #Generating a default ITI for the experiment (if not provided)
        if ITI is not None:
            assert len(ITI) == n_trials, "ITI length must be the same "
            self.ITI = ITI
        elif self.conditional_ITI is not None and order is not None: 
            self.ITI = self.generate_iti(order, self.conditional_ITI)
        else: 
            self.ITI, ITIlam = generate.iti(
                ntrials=self.n_trials,
                model=self.ITImodel,
                min=self.ITImin,
                max=self.ITImax,
                mean=self.ITImean,
                lam=self.ITIlam,
                seed= np.random.randint(10000),
                resolution=self.resolution,
            )
            if ITIlam:
                self.ITIlam = ITIlam

        self.restnum = restnum
        self.restdur = restdur

        self.FeMax = FeMax
        self.FdMax = FdMax
        self.FcMax = FcMax
        self.FfMax = FfMax

        # make sure resolution is a divisor of TR (up to )
        if not np.isclose(self.TR % self.resolution, 0):
            self.resolution = _find_new_resolution(self.TR, self.resolution)
            warnings.warn(
                "the resolution is adjusted to be a multiple of the TR. "
                f"New resolution: {self.resolution}"
            )

        self.countstim()
        self.CreateTsComp()
        self.CreateLmComp()
        self.max_eff()

    def max_eff(self):
        """Compute maximum efficiency for Confounding and Frequency efficiency."""
        NulDesign = Design(
            order=[np.argmin(self.P)] * self.n_trials,
            ITI=[0] + [self.ITImean] * (self.n_trials - 1),
            experiment=self,
        )
        NulDesign.designmatrix()
        NulDesign.FcCalc(self.confoundorder)
        self.FcMax = 1 - NulDesign.Fc
        NulDesign.FfCalc()
        self.FfMax = 1 - NulDesign.Ff

        return self

    def countstim(self):
        """Compute some arguments depending on other arguments."""
        self.trial_duration = self.trial_max + self.t_pre + self.t_post

        if self.ITImodel == "uniform":
            self.ITImean = (self.ITImax + self.ITImin) / 2

        if self.stimuli_durations is None: 
            #Specific n_trials and duration conditions to avoid updates when both are provided (pre_calculated)
            #If both are provided no further calculations required
            if self.n_trials is not None and self.duration == None:
                ITIdur = self.n_trials * self.ITImean
                TRIALdur = self.n_trials * self.trial_duration
                duration = ITIdur + TRIALdur
                if self.restnum > 0:
                    duration = duration + (
                        np.floor(self.n_trials / self.restnum) * self.restdur
                    )
                self.duration = duration
            elif self.duration is not None and self.n_trials == None:
                self.n_trials = self._compute_n_trials()
        else:
            assert self.n_trials is not None, "Must have n_trials provided with variable stimuli durations"
            if self.order_fixed: 
                self.calculate_all_stimuli()
                TRIALdur = sum(self.all_stim_durations)
            else: 
                self.all_stim_durations = [(self.trial_max + self.t_pre + self.t_post)] * len(self.n_trials)
                TRIALdur = self.n_trials * self.trial_max
                
            self.duration = self.calculate_duration(self.ITI, self.all_stim_durations)


    #Computes the n_trials given the duration
    def _compute_n_trials(self):
        if self.restnum == 0:
            return int(self.duration / (self.ITImean + self.trial_duration))

        # duration of block between rest
        blockdurNR = self.restnum * (self.ITImean + self.trial_duration)

        # duration of block including rest
        blockdurWR = blockdurNR + self.restdur

        # number of blocks
        blocknum = np.floor(self.duration / blockdurWR)
        n_trials = blocknum * self.restnum

        remain = self.duration - (blocknum * blockdurWR)
        if remain >= blockdurNR:
            n_trials = n_trials + self.restnum
        else:
            extratrials = np.floor(remain / (self.ITImean + self.trial_duration))
            n_trials = n_trials + extratrials

        return int(n_trials)

    def CreateTsComp(self):
        """Compute the number of scans and timpoints (in seconds and resolution units)."""
        self.n_scans = int(np.ceil(self.duration / self.TR))  # number of scans
        # number of timepoints  (in resolution)
        self.n_tp = int(np.ceil(self.duration / self.resolution))
        self.r_scans = np.arange(0, self.duration, self.TR)
        self.r_tp = np.arange(0, self.duration, self.resolution)

        return self

    def CreateLmComp(self):
        """Generate components for the linear model.
        - hrf,
        - whitening matrix
        - autocorrelation matrix
        - CX
        """
        # hrf
        self.canonical()

        # contrasts
        # expand contrasts to resolution
        self.CX = np.array(np.kron(self.C, np.eye(self.laghrf)))
        assert self.CX.shape[0] == self.C.shape[0] * self.laghrf
        assert self.CX.shape[1] == self.n_stimuli * self.laghrf

        # drift
        self.S = self.drift(np.arange(0, self.n_scans))  # [tp x 1]
        assert self.S.shape == (3, self.n_scans)
        self.S = np.matrix(self.S)

        # square of the whitening matrix
        base = [1 + self.rho**2, -1 * self.rho] + [0] * (self.n_scans - 2)
        self.V2 = scipy.linalg.toeplitz(base)
        # set first and last to 1
        self.V2[0, 0] = 1
        self.V2[self.n_scans - 1, self.n_scans - 1] = 1
        self.V2 = np.matrix(self.V2)

        self.white = (
            self.V2
            - self.V2
            * t(self.S)
            * np.linalg.pinv(self.S * self.V2 * t(self.S))
            * self.S
            * self.V2
        )

        return self

    def canonical(self):
        """Generate the canonical hrf.

        :param resolution: resolution to sample the canonical hrf
        :type resolution: float
        """
        # translated from spm_hrf
        p = [6, 16, 1, 1, 6, 0, 32]
        dt = self.resolution
        s = np.array(range(int(np.ceil(p[6] / dt))))
        # HRF sampled at resolution
        hrf = (
            self.spm_Gpdf(s, p[0] / p[2], dt / p[2])
            - self.spm_Gpdf(s, p[1] / p[3], dt / p[3]) / p[4]
        )
        self.basishrf = hrf / np.sum(hrf)
        s  # duration of the HRF
        self.durhrf = p[6]
        # length of the HRF parameters in resolution scale
        self.laghrf = int(np.ceil(self.durhrf / self.resolution))
        assert self.laghrf == len(s)

        return self

    @staticmethod
    def drift(s, deg=3):
        """Compute a drift component."""
        S = np.ones([deg, len(s)])
        s = np.array(s)
        tmpt = np.array(2.0 * s / float(len(s) - 1) - 1)
        S[1] = tmpt
        for k in np.arange(2, deg):
            S[k] = ((2.0 * k - 1.0) / k) * tmpt * S[k - 1] - ((k - 1) / float(k)) * S[
                k - 2
            ]
        return S

    @staticmethod
    def spm_Gpdf(s, h, l):
        """Generate gamma pdf."""
        s = np.array(s)
        res = (h - 1) * np.log(s) + h * np.log(l) - l * s - np.log(gamma(h))
        return np.exp(res)
    

    #Calculating the new duration and all_stimuli lengths and trial_durations based on new order
    def calculate_all_stimuli(self):
        self.all_stim_durations = []
        for i in range(len(self.order)):
            stimuli = self.order[i]
            key = self.stimuli_durations[stimuli]
            
            if isinstance(key, dict): 
                params = key

                if params["model"] == "fixed":
                    self.all_stim_durations.append(params["mean"])
                elif params["model"] == "exponential":
                    val = np.random.exponential(scale=params["mean"])
                    if "min" in params:
                        val = max(val, params["min"])
                    self.all_stim_durations.append(val)
                elif params["model"] == "uniform":
                    self.all_stim_durations.append(np.random.uniform(params["min"], params["max"]))
                elif params["model"] == "gaussian":
                    mean = params.get("mean", 0)
                    std = params.get("std", 1)
                    val = np.random.normal(loc=mean, scale=std)
                    if "min" in params:
                        val = max(val, params["min"])
                    if "max" in params:
                        val = min(val, params["max"])

                    self.all_stim_durations.append(val)

            else: 
                self.all_stim_durations.append(key)
        
        assert len(self.all_stim_durations) == len(self.order)

        self.all_stim_durations = [d + self.t_pre + self.t_post for d in self.all_stim_durations]
        # print(sum(self.all_stim_durations))
        self.duration = self.calculate_duration(self.ITI, self.all_stim_durations)

    #Added functions
    #Generates ITI with the first value being the default then the order. (ensures n_trials and ITI length is the same)
    def generate_iti(self, order, conditional_iti):
        ITI = []

        for i in range(len(order)):
            stim_prev = order[i - 1] if i > 0 else None
            stim_curr = order[i]

            if stim_prev is not None or stim_curr is not None: 
            # Determine key for condition-based ITI
                key = (stim_prev, stim_curr) if stim_prev is not None else "default"
                params = conditional_iti.get(key, conditional_iti.get("default"))

                # Generate ITI based on parameters
                if params["model"] == "fixed":
                    ITI.append(params["mean"])
                elif params["model"] == "exponential":
                    val = np.random.exponential(scale=params["mean"])
                    if "min" in params:
                        val = max(val, params["min"])
                    if "max" in params:
                        val = min(val, params['max'])
                    ITI.append(val)
                elif params["model"] == "uniform":
                    ITI.append(np.random.uniform(params["min"], params["max"]))
                elif params["model"] == "gaussian":
                    mean = params.get("mean", 0)
                    std = params.get("std", 1)
                    val = np.random.normal(loc=mean, scale=std)
                    if "min" in params:
                        val = max(val, params["min"])
                    if "max" in params:
                        val = min(val, params["max"])

                    self.all_stim_durations.append(val)
        return ITI

    #Generates/calculates the duration based on stimuli_duration and ITI
    def calculate_duration(self, ITI, dur):
        total_sum = sum(dur)
        total_sum = total_sum + sum(ITI)  
        return total_sum
    
    #Generates an order based on a given probability distribution of certain keys
    def sample_from_probabilities(self, prob, key, length):
        # random.choices picks elements from key with weights = prob_array\
        samples = random.choices(key, weights=prob, k=length)
        merged = [item for sublist in samples for item in sublist]
        return merged[:self.n_trials]


class Optimisation:
    """Represent the population of experimental designs for fMRI.

    :param experiment: The experimental setup of the fMRI experiment.
    :type  experiment: experiment

    :param G: The size of the generation
    :type  G: integer

    :param R: with which rate are the orders generated from ['blocked','random','mseq']
    :type  R: list

    :param q: percentage of mutations
    :type  q: float

    :param weights: weights attached to Fe, Fd, Ff, Fc
    :type  weights: list

    :param I: number of immigrants
    :type  I: integer

    :param preruncycles: number of prerun cycles (to find maximum Fe and Fd)
    :type  preruncycles: integer

    :param cycles: number of cycles
    :type  cycles: integer

    :param seed: seed
    :type  seed: integer

    :param Aoptimality: optimises A-optimality if true, else D-optimality
    :type  Aoptimality: boolean

    :param convergence: after how many stable iterations is there convergence
    :type  convergence: integer

    :param folder: folder to save output
    :type  folder: string

    :param outdes: number of designs to be saved
    :type  outdes: integer

    :param optimisation: The type of optimisation - 'GA' or 'simulation'
    :type  optimisation: string
    """

    def __init__(
        self,
        experiment: experiment,
        weights: list[float],
        preruncycles: int,
        cycles: int,
        seed: int | None = None,
        I: int = 4,
        G: int = 20,
        R: list[float] | None = None,
        q: float = 0.01,
        Aoptimality: bool = True,
        folder: str | Path | None = None,
        outdes: int = 3,
        convergence: int = 1000,
        optimisation: str = "GA",
    ):

        self.exp = experiment
        self.G = G
        self.R = [0.4, 0.4, 0.2] if R is None else R
        self.q = q
        self.weights = weights
        self.I = I
        self.preruncycles = preruncycles
        self.cycles = cycles
        self.convergence = convergence
        self.Aoptimality = Aoptimality
        self.outdes = outdes
        self.folder = Path(folder).absolute() if folder else None
        self.optimisation = optimisation
        self.seed = seed or np.random.randint(10000)
        self.designs = []
        self.optima = []
        self.bestdesign = None
        self.cov = None

    def change_seed(self):
        """Change the seed."""
        self.seed = self.seed + 1000 if self.seed < 4 * 10**9 else 1
        return self

    def check_develop(self, design, weights=None):
        """Check and develop a design to the population.

        Function will check design against strict options and develop the design if valid.

        :param design: Design to be added to population.
        :type design:  esign object

        :param weights: weights for efficiency calculation.
        :type  weights: list of floats, summing to 1
        """
        # weights

        if weights is None:
            weights = self.weights

        # check maxrep, hardprob, every stimulus at least once
        if self.exp.maxrep is not None and not design.check_maxrep(self.exp.maxrep):
            return False
        if self.exp.hardprob and not design.check_hardprob():
            return False
        if len(np.unique(design.order)) < self.exp.n_stimuli:
            return False

        # develop

        out = design.designmatrix()
        if out is False:
            return False
        design.FCalc(
            weights, confoundorder=self.exp.confoundorder, Aoptimality=self.Aoptimality
        )
        return False if np.isnan(design.F) else design

    def add_new_designs(self, weights=None, R=None):
        """Generate the population.

        :param experiment: The experimental setup of the fMRI experiment.
        :type experiment: experiment
        :param weights: weights for efficiency calculation.
        :type weights: list of floats, summing to 1
        :param seed: The seed for random processes.
        :type seed: integer or None
        """
        # weights
        if weights is None:
            weights = self.weights

        if not R:
            R = np.round(np.array(self.R) * self.G).tolist()

        if self.exp.n_stimuli in [6, 10] and R[2] > 0:
            warnings.warns(
                "for this number of conditions/stimuli, "
                "there are no msequences possible.\n"
                "Replaced by random designs."
            )
            R[1] = R[1] + R[2]
            R[2] = 0

        NDes = 0
        self.change_seed()

        while NDes < np.sum(R):
            self.change_seed()
            ind = np.sum(NDes >= np.cumsum(R))
            ordertype = ["blocked", "random", "msequence"][ind]

            #Modified by Atharv Umap
            order = None

            if self.exp.order_fixed: 
                order = self.exp.order
            elif self.exp.order_probabilities is None:
                order = generate.order(
                self.exp.n_stimuli,
                self.exp.n_trials,
                self.exp.P,
                ordertype=ordertype,
                seed=self.seed,
                )
            else:
                order = self.exp.sample_from_probabilities(self.exp.order_probabilities, self.exp.order_keys, self.exp.order_length)
        
            #If conditional_ITI not provided use default ITI calculation, else use custom calculation.
            ITI = []
            if self.exp.conditional_ITI is None: 
                ITI, ITIlam = generate.iti(
                    ntrials=self.exp.n_trials,
                    model=self.exp.ITImodel,
                    
                    min=self.exp.ITImin,
                    max=self.exp.ITImax,
                    mean=self.exp.ITImean,
                    lam=self.exp.ITIlam,
                    seed=self.seed,
                    resolution=self.exp.resolution,
                )
                if ITIlam:
                    self.exp.ITIlam = ITIlam
            else: 
                ITI = self.exp.generate_iti(order, self.exp.conditional_ITI)

            des = None
            #des = Design(order=order, ITI=np.array(ITI), experiment=self.exp)
            if self.exp.conditional_ITI is not None or self.exp.order_probabilities is not None:
                new_exp = Experiment(
                    TR = self.exp.TR,
                    P = self.exp.P,
                    C = self.exp.C,
                    rho = self.exp.rho,
                    stim_duration = self.exp.stim_duration,
                    n_stimuli = self.exp.n_stimuli,
                    stimuli_durations = self.exp.stimuli_durations, 
                    ITI = ITI, 
                    conditional_ITI = self.exp.conditional_ITI, #Adding a new input for ITI's that can be controlled by the user.
                    ITImodel= self.exp.ITImodel,
                    ITImin= self.exp.ITImin,
                    ITImax= self.exp.ITImax,
                    ITImean= self.exp.ITImean,
                    restnum= self.exp.restnum,
                    restdur= self.exp.restdur,
                    t_pre=0,
                    t_post=0,
                    n_trials= self.exp.n_trials,
                    resolution= self.exp.resolution,
                    FeMax= self.exp.FeMax,
                    FdMax= self.exp.FdMax,
                    FcMax= self.exp.FcMax ,
                    FfMax= self.exp.FfMax,
                    maxrep= self.exp.maxrep,
                    hardprob= self.exp.hardprob,
                    confoundorder= self.exp.confoundorder,
                    order = order, 
                    order_probabilities = self.exp.order_probabilities,
                    order_keys = self.exp.order_keys, 
                    order_length = self.exp.order_length,
                    order_fixed = self.exp.order_fixed,
                    trial_max= self.exp.trial_max
                )
               
                des = Design(order=order, ITI=np.array(ITI), experiment=new_exp)
            else:
                des = Design(order=order, ITI=np.array(ITI), experiment=self.exp)
            
            fulldes = self.check_develop(des, weights)

            if fulldes is False:
                continue
            self.designs.append(fulldes)
            NDes += 1

        return self

    def _clean_designs(self, weights):
        n = 0
        rm = 0
        while n == 0: 
            orders = [x.order for x in self.designs]
            cors = np.corrcoef(orders)
            isone = np.isclose(cors, 1.0)
            if len(isone) == 1:
                n = 1
            else:
                np.fill_diagonal(isone, 0)
                if np.sum(isone) == 0:
                    n = 1
                else:
                    ind = np.where(isone)
                    remove = ind[1][ind[0] == ind[0][0]]
                    self.designs = [
                        des for ind, des in enumerate(self.designs) if ind not in remove
                    ]
                    rm = rm + len(remove)
                
                

        self.add_new_designs(R=[0, rm, 0], weights=weights)

        return self

    def _mutation(self, weights, seed):
        # Mutation:
        # if: Best design: stay untouched
        # elif Correlation between all is > 0.8: mutate with 20% mutations
        # else: mutate with 5% mutations
        # for all: if conditions are not fulfilled: not mutated

        signals = [x.Xconv for x in self.designs]
        efficiencies = [x.F for x in self.designs]

        cors = self.pearsonr(signals, self.exp.n_stimuli)
        mncor = np.mean(cors)

        for idx in range(len(self.designs)):
            design = self.designs[idx]

            if design.F == np.max(efficiencies):
                offspring = design

            elif mncor > 0.6:
                offspring = design.mutation(0.2, seed=seed)
                offspring = self.check_develop(offspring, weights)

            else:
                offspring = design.mutation(self.q, seed=seed)
                offspring = self.check_develop(offspring, weights)

            if offspring is False:
                continue
            else:
                self.designs[idx] = offspring

        return self

    def _crossover(self, weights, seed):
        # select designs with F>median(F):
        crossind = range(len(self.designs))

        nparents = len(crossind)
        npairs = int(nparents / 2.0)

        np.random.seed(seed)
        CouplingRnd = np.random.choice(nparents, size=(npairs * 2), replace=False)
        CouplingRnd = [crossind[x] for x in CouplingRnd]
        CouplingRnd = [
            [CouplingRnd[i], CouplingRnd[i + 1]] for i in np.arange(0, npairs * 2, 2)
        ]

        count = 0

        for couple in CouplingRnd:
            baby1, baby2 = self.designs[couple[0]].crossover(
                self.designs[couple[1]], seed=seed
            )
            for baby in [baby1, baby2]:
                baby = self.check_develop(baby, weights)
                if baby is False:
                    continue
                self.designs.append(baby)
                count = count + 1

        return self

    def _immigration(self, weights, noim):
        R = np.ceil(np.array(self.R) * noim).tolist()
        self.add_new_designs(R=R, weights=weights)

        return self

    def to_next_generation(self, weights=None, seed=1234, optimisation=None):
        """Go from one generation to the next.

        :param weights: weights for efficiency calculation.
        :type weights: list of floats, summing to 1

        :param seed: The seed for random processes.
        :type seed: integer or None

        :param optimisation: The type of optimisation - 'GA' or 'simulation'
        :type optimisation: string
        """
        if optimisation is None:
            optimisation = self.optimisation

        # weights
        if weights is None:
            weights = self.weights

        #If the order is fixed, we don't want to perform crossover or mutation on the designs
        if not self.exp.order_fixed: 
            self._clean_designs(weights)
            if optimisation == "GA":
                self._mutation(weights, seed)
                self._crossover(weights, seed)
                self._immigration(weights, noim=self.I)

            elif optimisation == "simulation":
                self._immigration(weights, noim=self.I)
            else:
                print("Unknown optimisation type")
        else: 
            self._immigration(weights, noim=self.I)

        # inspect efficiencies
        efficiencies = [x.F for x in self.designs]
        maximum = np.max(efficiencies)
        self.optima.append(maximum)
        bestind = [ind for ind, val in enumerate(efficiencies) if val == maximum][0]
        self.bestdesign = self.designs[bestind]

        # append best designs to lists

        # check convergence
        gen = len(self.optima)
        if gen > 1000 and self.optima[-1] > self.optima[gen - 1000]:
            self.finished = True
            

        # select best G
        cutoff = np.sort(efficiencies)[::-1][self.G]
        self.designs = [des for des in self.designs if des.F >= cutoff]

        return self

    def clear(self):
        """Clear results between optimalisations (maximum Fe, Fd or opt)."""
        self.designs = []
        self.optima = []
        self.finished = False
        self.change_seed()

        if self.bestdesign:
            bestdes = Design(
                order=self.bestdesign.order, ITI=self.bestdesign.ITI, experiment=self.exp
            )
            bestdes = self.check_develop(bestdes)
            if bestdes is not False:
                self.designs.append(bestdes)
            self.bestdesign = None

        return self

    def optimise(self):
        """Run design optimization."""
        if self.exp.FcMax == 1 and self.exp.FfMax == 1:
            self.exp.max_eff()

        if self.exp.FeMax == 1 and self.weights[0] > 0:
            # add new designs
            self.clear()
            self.add_new_designs(weights=[1, 0, 0, 0])
            with progress_bar(text="Optimizing") as progress:
                task = progress.add_task(
                    description="optimize", total=len(range(self.preruncycles))
                )
                for _ in range(self.preruncycles):
                    self.to_next_generation(seed=self.seed, weights=[1, 0, 0, 0])
                    progress.update(task, advance=1)
                    if self.finished:
                        continue
            self.exp.FeMax = np.max(self.bestdesign.F)

        if self.exp.FdMax == 1 and self.weights[1] > 0:
            self.clear()
            self.add_new_designs(weights=[0, 1, 0, 0])
            with progress_bar(text="Optimizing") as progress:
                task = progress.add_task(
                    description="optimize", total=len(range(self.preruncycles))
                )
                for _ in range(self.preruncycles):
                        self.to_next_generation(seed=self.seed, weights=[0, 1, 0, 0])
                        progress.update(task, advance=1)
                        if self.finished:
                            continue
            self.exp.FdMax = np.max(self.bestdesign.F)

        # clear all attributes
        self.clear()
        self.add_new_designs()

        # loop
        with progress_bar(text="Optimizing") as progress:
            task = progress.add_task(
                description="optimize", total=len(range(self.cycles))
            )
            for _ in range(self.cycles):
                self.to_next_generation(seed=self.seed)
                progress.update(task, advance=1)
                if self.finished:
                    continue

        return self

    def evaluate(self):
        # select designs: best from k-means clusters
        shape = self.bestdesign.Xconv.shape
        des = np.zeros([np.prod(shape), len(self.designs)])
        efficiencies = np.array([x.F for x in self.designs])
        for d in range(len(self.designs)):
            hrf = []
            for stim in range(shape[1]):
                hrf = hrf + self.designs[d].Xconv[:, stim].tolist()
            des[:, d] = hrf
        clus = sklearn.cluster.k_means(des.T, self.outdes, random_state=self.seed)[1]
        out = []
        des = []
        cl = []
        first = 0
        for c in range(self.outdes):
            ids = np.where(clus == c)[0]
            id_ordered = ids[np.flipud(np.argsort(efficiencies[ids]))]
            out.append(first)
            for d in id_ordered:
                cl.append(c)
                des.append(self.designs[d])
                first = first + 1
        self.designs = des
        self.out = out
        self.clus = cl

        signals = [x.Xconv for x in self.designs]
        co = self.pearsonr(signals, 3)
        self.cov = co

        return self

    def download(self):
        if not self.folder:
            raise ValueError("No folder defined to download output.")

        if self.cov is None:
            self.evaluate()

        # empty folder
        if self.folder.exists():
            files = self.folder.glob("**/design_*")
            for f in files:
                shutil.rmtree(f)
        else:
            self.folder.mkdir(parents=True, exist_ok=True)

        reportfile = "report.pdf"
        report.make_report(self, self.folder / reportfile)

        files = []

        for des in range(self.outdes):

            (self.folder / f"design_{str(des)}").mkdir(parents=True)

            design = self.designs[self.out[des]]

            for stim in range(self.exp.n_stimuli):

                onsetsfile = Path(f"design_{str(des)}") / f"stimulus_{str(stim)}.txt"

                onsubsets = [
                    str(x)
                    for x in np.array(design.onsets)[np.array(design.order) == stim]
                ]
                with open(self.folder / onsetsfile, "w+") as f:
                    for line in onsubsets:
                        f.write(line)
                        f.write("\n")
                files.append(onsetsfile)

            itifile = Path(f"design_{str(des)}") / "ITIs.txt"

            with open(self.folder / itifile, "w+") as f:
                for line in design.ITI:
                    f.write(str(line))
                    f.write("\n")
            files.append(itifile)

        files.append(reportfile)

        # zip up
        zip_subdir = "OptimalDesign"
        self.zip_filename = f"{zip_subdir}.zip"
        self.file = BytesIO()
        zf = zipfile.ZipFile(self.file, "w")

        for fpath in files:
            zf.write(self.folder / fpath, Path(zip_subdir) / fpath)

        zf.close()

        return self

    @staticmethod
    def pearsonr(signals, nstim):
        varcov = np.zeros([len(signals), len(signals)])
        for sig1 in range(len(signals)):
            for sig2 in range(sig1, len(signals)):
                cors = np.diag(
                    np.corrcoef(t(signals[sig1]), t(signals[sig2]))[nstim:, :nstim]
                )
                varcov[sig1, sig2] = np.mean(cors)
                varcov[sig2, sig1] = np.mean(cors)
        return varcov


def _find_new_resolution(TR, res):
    n = TR * 1000.0
    # find divisors of TR*1000
    large_divisors = []
    for i in range(1, int(math.sqrt(n) + 1)):
        if n % i == 0:
            large_divisors.append(i)
            if i * i != n:
                large_divisors.append(int(n / i))
    sorted = np.sort(large_divisors)
    # closest to res
    resdivisor = TR / float(res)
    difs = np.abs(resdivisor - sorted)
    minind = np.where(difs == np.min(difs))[0]
    divisor = sorted[minind][0]
    newres = TR / divisor
    return newres


def _round_to_resolution(inmat, res):
    out = res * np.floor(np.array(inmat) / res)
    ind = out / res
    ind = [int(x) for x in ind]
    return out, ind



