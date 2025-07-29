import ImProver.get_prompts.utils

open Lean Core Elab IO Meta Term Command Tactic

inductive Version : Type
  | optimized
  | unoptimized
deriving Inhabited, BEq, Repr

instance :ToString Version where
  toString := fun v =>
    match v with
    | Version.optimized   => "optimized"
    | Version.unoptimized => "unoptimized"

def versionAttrImpl : ParametricAttributeImpl Version where
  name := `version
  descr := "version tag"
  getParam := fun _ stx => do
  if stx.getNumArgs < 2 then
    throwError "Expected argument"
  match stx[1]![0]! with
  | `(optimized)   => pure Version.optimized
  | `(unoptimized) => pure Version.unoptimized
  -- | _ => pure Version.auto
  | _              => throwError s!"Expected 'optimized' or 'unoptimized', got {stx}, {stx[1]!},{stx[1]![0]!}"

initialize versionAttribute : ParametricAttribute Version ←
  registerParametricAttribute versionAttrImpl


def nameAttrImpl : ParametricAttributeImpl Name where
  name := `improver_example
  descr := "name tag"
  getParam := fun _ stx => do
  if stx.getNumArgs < 2 then
    throwError "Expected argument"
  pure stx[1]![0]!.getId.toString.toName


initialize nameAttribute : ParametricAttribute Name ←
  registerParametricAttribute nameAttrImpl
