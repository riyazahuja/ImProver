instance : HasExplicitPullbacks (fun Y ↦ TotallyDisconnectedSpace Y) where
  hasProp _ _ := { hasProp :=
    show TotallyDisconnectedSpace {_xy : _ | _} from inferInstance}


instance : HasExplicitFiniteCoproducts.{w, u} (fun Y ↦ TotallyDisconnectedSpace Y) where
  hasProp _ := { hasProp :=
    show TotallyDisconnectedSpace (Σ (_a : _), _) from inferInstance}


/-- A one-element space is terminal in `Profinite` -/
abbrev isTerminalPUnit : IsTerminal (Profinite.of PUnit.{u + 1}) := CompHausLike.isTerminalPUnit


