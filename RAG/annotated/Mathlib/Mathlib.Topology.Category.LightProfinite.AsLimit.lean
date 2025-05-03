/-- The functor `ℕᵒᵖ ⥤ FintypeCat` whose limit is isomorphic to `S`. -/
abbrev fintypeDiagram : ℕᵒᵖ ⥤ FintypeCat := S.toLightDiagram.diagram


/-- An abbreviation for `S.fintypeDiagram ⋙ FintypeCat.toProfinite`. -/
abbrev diagram : ℕᵒᵖ ⥤ LightProfinite := S.fintypeDiagram ⋙ FintypeCat.toLightProfinite


/--
A cone over `S.diagram` whose cone point is isomorphic to `S`.
(Auxiliary definition, use `S.asLimitCone` instead.)
-/
def asLimitConeAux : Cone S.diagram :=
  let c : Cone (S.diagram ⋙ lightToProfinite) := S.toLightDiagram.cone
  let hc : IsLimit c := S.toLightDiagram.isLimit
  liftLimit hc


/-- An auxiliary isomorphism of cones used to prove that `S.asLimitConeAux` is a limit cone. -/
def isoMapCone : lightToProfinite.mapCone S.asLimitConeAux ≅ S.toLightDiagram.cone :=
  let c : Cone (S.diagram ⋙ lightToProfinite) := S.toLightDiagram.cone
  let hc : IsLimit c := S.toLightDiagram.isLimit
  liftedLimitMapsToOriginal hc


/--
`S.asLimitConeAux` is indeed a limit cone.
(Auxiliary definition, use `S.asLimit` instead.)
-/
def asLimitAux : IsLimit S.asLimitConeAux :=
  let hc : IsLimit (lightToProfinite.mapCone S.asLimitConeAux) :=
    S.toLightDiagram.isLimit.ofIsoLimit S.isoMapCone.symm
  isLimitOfReflects lightToProfinite hc


/-- A cone over `S.diagram` whose cone point is `S`. -/
def asLimitCone : Cone S.diagram where
  pt := S
  π := {
    app := fun n ↦ (lightToProfiniteFullyFaithful.preimageIso <|
      (Cones.forget _).mapIso S.isoMapCone).inv ≫ S.asLimitConeAux.π.app n
                                 /-
                                   S : LightProfinite
                                   x✝² x✝¹ : Opposite Nat
                                   x✝ : Quiver.Hom x✝² x✝¹
                                   ⊢ Eq (CategoryTheory.CategoryStruct.comp (((CategoryTheory.Functor.const (Oppo …
                                 -/
    naturality := fun _ _ _ ↦ by simp only [Category.assoc, S.asLimitConeAux.w]; rfl }
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/


/-- `S.asLimitCone` is indeed a limit cone. -/
def asLimit : IsLimit S.asLimitCone := S.asLimitAux.ofIsoLimit <|
  Cones.ext (lightToProfiniteFullyFaithful.preimageIso <|
                                                      /-
                                                        S : LightProfinite
                                                        x✝ : Opposite Nat
                                                        ⊢ Eq (S.asLimitConeAux.π.app x✝) (CategoryTheory.CategoryStruct.comp (lightToP …
                                                      -/
    (Cones.forget _).mapIso S.isoMapCone) (fun _ ↦ by rw [← @Iso.inv_comp_eq]; rfl)
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


/-- A bundled version of `S.asLimitCone` and `S.asLimit`. -/
def lim : Limits.LimitCone S.diagram := ⟨S.asLimitCone, S.asLimit⟩


/-- The projection from `S` to the `n`th component of `S.diagram`. -/
abbrev proj (n : ℕ) : S ⟶ S.diagram.obj ⟨n⟩ := S.asLimitCone.π.app ⟨n⟩


lemma lightToProfinite_map_proj_eq (n : ℕ) : lightToProfinite.map (S.proj n) =
    (lightToProfinite.obj S).asLimitCone.π.app _ := by
  /-
    S : LightProfinite
    n : Nat
    ⊢ Eq (lightToProfinite.map (S.proj n)) ((lightToProfinite.obj S).asLimitCone.π …
  -/
  simp only [toCompHausLike_obj, Functor.comp_obj, toCompHausLike_map, coe_of]
  /-
    S : LightProfinite
    n : Nat
    ⊢ Eq (S.proj n) ((Profinite.asLimitCone (CompHausLike.of (fun X => TotallyDisc …
  -/
  let c : Cone (S.diagram ⋙ lightToProfinite) := S.toLightDiagram.cone
  /-
    S : LightProfinite
    n : Nat
    c : CategoryTheory.Limits.Cone (S.diagram.comp lightToProfinite) := S.toLightD …
    ⊢ Eq (S.proj n) ((Profinite.asLimitCone (CompHausLike.of (fun X => TotallyDisc …
  -/
  let hc : IsLimit c := S.toLightDiagram.isLimit
  /-
    S : LightProfinite
    n : Nat
    c : CategoryTheory.Limits.Cone (S.diagram.comp lightToProfinite) := S.toLightD …
    hc : CategoryTheory.Limits.IsLimit c := S.toLightDiagram.isLimit
    ⊢ Eq (S.proj n) ((Profinite.asLimitCone (CompHausLike.of (fun X => TotallyDisc …
  -/
  exact liftedLimitMapsToOriginal_inv_map_π hc _
  /-
    🎉 no goals
  -/


lemma proj_surjective (n : ℕ) : Function.Surjective (S.proj n) := by
  /-
    S : LightProfinite
    n : Nat
    ⊢ Function.Surjective ⇑(S.proj n)
  -/
  change Function.Surjective (lightToProfinite.map (S.proj n))
  /-
    S : LightProfinite
    n : Nat
    ⊢ Function.Surjective ⇑(lightToProfinite.map (S.proj n))
  -/
  rw [lightToProfinite_map_proj_eq]
  /-
    S : LightProfinite
    n : Nat
    ⊢ Function.Surjective ⇑((lightToProfinite.obj S).asLimitCone.π.app ((CategoryT …
  -/
  exact DiscreteQuotient.proj_surjective _
  /-
    🎉 no goals
  -/


/-- An abbreviation for the `n`th component of `S.diagram`. -/
abbrev component (n : ℕ) : LightProfinite := S.diagram.obj ⟨n⟩


/-- The transition map from `S_{n+1}` to `S_n` in `S.diagram`. -/
abbrev transitionMap (n : ℕ) :  S.component (n+1) ⟶ S.component n :=
  S.diagram.map ⟨homOfLE (Nat.le_succ _)⟩


/-- The transition map from `S_m` to `S_n` in `S.diagram`, when `m ≤ n`. -/
abbrev transitionMapLE {n m : ℕ} (h : n ≤ m) : S.component m ⟶ S.component n :=
  S.diagram.map ⟨homOfLE h⟩


lemma proj_comp_transitionMap (n : ℕ) :
    S.proj (n + 1) ≫ S.diagram.map ⟨homOfLE (Nat.le_succ _)⟩ = S.proj n :=
  S.asLimitCone.w (homOfLE (Nat.le_succ n)).op


lemma proj_comp_transitionMap' (n : ℕ) : S.transitionMap n ∘ S.proj (n + 1) = S.proj n := by
  /-
    S : LightProfinite
    n : Nat
    ⊢ Eq (Function.comp ⇑(S.transitionMap n) ⇑(S.proj (HAdd.hAdd n 1))) ⇑(S.proj n)
  -/
  rw [← S.proj_comp_transitionMap n]
  /-
    S : LightProfinite
    n : Nat
    ⊢ Eq (Function.comp ⇑(S.transitionMap n) ⇑(S.proj (HAdd.hAdd n 1))) ⇑(Category …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma proj_comp_transitionMapLE {n m : ℕ} (h : n ≤ m) :
    S.proj m ≫ S.diagram.map ⟨homOfLE h⟩ = S.proj n :=
  S.asLimitCone.w (homOfLE h).op


lemma proj_comp_transitionMapLE' {n m : ℕ} (h : n ≤ m) :
    S.transitionMapLE h ∘ S.proj m  = S.proj n := by
  /-
    S : LightProfinite
    n m : Nat
    h : LE.le n m
    ⊢ Eq (Function.comp ⇑(S.transitionMapLE h) ⇑(S.proj m)) ⇑(S.proj n)
  -/
  rw [← S.proj_comp_transitionMapLE h]
  /-
    S : LightProfinite
    n m : Nat
    h : LE.le n m
    ⊢ Eq (Function.comp ⇑(S.transitionMapLE h) ⇑(S.proj m)) ⇑(CategoryTheory.Categ …
  -/
  rfl
  /-
    🎉 no goals
  -/


lemma surjective_transitionMap (n : ℕ) : Function.Surjective (S.transitionMap n) := by
  /-
    S : LightProfinite
    n : Nat
    ⊢ Function.Surjective ⇑(S.transitionMap n)
  -/
  apply Function.Surjective.of_comp (g := S.proj (n + 1))
  /-
    S : LightProfinite
    n : Nat
    ⊢ Function.Surjective (Function.comp ⇑(S.transitionMap n) ⇑(S.proj (HAdd.hAdd  …
  -/
  simpa only [proj_comp_transitionMap'] using S.proj_surjective n
  /-
    🎉 no goals
  -/


lemma surjective_transitionMapLE {n m : ℕ} (h : n ≤ m) :
    Function.Surjective (S.transitionMapLE h) := by
  /-
    S : LightProfinite
    n m : Nat
    h : LE.le n m
    ⊢ Function.Surjective ⇑(S.transitionMapLE h)
  -/
  apply Function.Surjective.of_comp (g := S.proj m)
  /-
    S : LightProfinite
    n m : Nat
    h : LE.le n m
    ⊢ Function.Surjective (Function.comp ⇑(S.transitionMapLE h) ⇑(S.proj m))
  -/
  simpa only [proj_comp_transitionMapLE'] using S.proj_surjective n
  /-
    🎉 no goals
  -/


