import TrainingData.Frontend
import TrainingData.InfoTree.TacticInvocation.Basic

open Lean Core Elab IO Meta Term Command Tactic System



def length_score (cs : CompilationStep) : IO Float :=
    return InfoTree.tactics_new cs.trees |>.length |>.toFloat
