instance factTestable {p : Prop} [Testable p] : Testable (Fact p) where
  run cfg min := do
    let h ← runProp p cfg min
    pure <| iff fact_iff h


instance Fact.printableProp {p : Prop} [PrintableProp p] : PrintableProp (Fact p) where
  printProp := printProp p


