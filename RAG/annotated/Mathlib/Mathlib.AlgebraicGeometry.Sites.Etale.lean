/-- Big étale site: the étale pretopology on the category of schemes. -/
def etalePretopology : Pretopology Scheme.{u} :=
  pretopology @IsEtale


/-- Big étale site: the étale topology on the category of schemes. -/
abbrev etaleTopology : GrothendieckTopology Scheme.{u} :=
  etalePretopology.toGrothendieck


lemma zariskiTopology_le_etaleTopology : zariskiTopology ≤ etaleTopology := by
  /-
    ⊢ LE.le AlgebraicGeometry.Scheme.zariskiTopology AlgebraicGeometry.Scheme.etal …
  -/
  apply grothendieckTopology_le_grothendieckTopology
  /-
    case hPQ
    ⊢ LE.le @AlgebraicGeometry.IsOpenImmersion @AlgebraicGeometry.IsEtale
  -/
  intro X Y f hf
  /-
    case hPQ
    X Y : AlgebraicGeometry.Scheme
    f : Quiver.Hom X Y
    hf : AlgebraicGeometry.IsOpenImmersion f
    ⊢ AlgebraicGeometry.IsEtale f
  -/
  infer_instance
  /-
    🎉 no goals
  -/


