/-- The type of points of a complete lattice `L`, where a *point* of a complete lattice is,
by definition, a frame homomorphism from `L` to `Prop`. -/
abbrev PT := FrameHom L Prop


/-- The frame homomorphism from a complete lattice `L` to the complete lattice of sets of
points of `L`. -/
@[simps]
def openOfElementHom : FrameHom L (Set (PT L)) where
  toFun u := {x | x u}
                     /-
                       L : Type u_1
                       inst✝ : CompleteLattice L
                       a b : L
                       ⊢ Eq ((fun u => setOf fun x => x u) (Min.min a b)) (Min.min ((fun u => setOf f …
                     -/
  map_inf' a b := by simp [Set.setOf_and]
                     /-
                       🎉 no goals
                     -/
                 /-
                   L : Type u_1
                   inst✝ : CompleteLattice L
                   ⊢ Eq ({ toFun := fun u => setOf fun x => x u, map_inf' := ⋯ }.toFun Top.top) T …
                 -/
  map_top' := by simp
                 /-
                   🎉 no goals
                 -/
                    /-
                      L : Type u_1
                      inst✝ : CompleteLattice L
                      S : Set L
                      ⊢ Eq ({ toFun := fun u => setOf fun x => x u, map_inf' := ⋯, map_top' := ⋯ }.t …
                    -/
  map_sSup' S := by ext; simp [Prop.exists_iff]
                         /-
                           🎉 no goals
                         -/


/-- The topology on the set of points of the complete lattice `L`. -/
instance instTopologicalSpace : TopologicalSpace (PT L) where
  IsOpen s := ∃ u, {x | x u} = s
                        /-
                          L : Type u_1
                          inst✝ : CompleteLattice L
                          ⊢ Eq (setOf fun x => x Top.top) Set.univ
                        -/
  isOpen_univ := ⟨⊤, by simp⟩
                        /-
                          🎉 no goals
                        -/
                     /-
                       L : Type u_1
                       inst✝ : CompleteLattice L
                       ⊢ ∀ (s t : Set (Locale.PT L)), (fun s => Exists fun u => Eq (setOf fun x => x  …
                     -/
  isOpen_inter := by rintro s t ⟨u, rfl⟩ ⟨v, rfl⟩; use u ⊓ v; simp_rw [map_inf]; rfl
                                                                                 /-
                                                                                   🎉 no goals
                                                                                 -/
  isOpen_sUnion S hS := by
    /-
      L : Type u_1
      inst✝ : CompleteLattice L
      S : Set (Set (Locale.PT L))
      hS : ∀ (t : Set (Locale.PT L)), Membership.mem S t → (fun s => Exists fun u => …
      ⊢ (fun s => Exists fun u => Eq (setOf fun x => x u) s) S.sUnion
    -/
    choose f hf using hS
    /-
      L : Type u_1
      inst✝ : CompleteLattice L
      S : Set (Set (Locale.PT L))
      f : (t : Set (Locale.PT L)) → Membership.mem S t → L
      hf : ∀ (t : Set (Locale.PT L)) (a : Membership.mem S t), Eq (setOf fun x => x  …
      ⊢ Exists fun u => Eq (setOf fun x => x u) S.sUnion
    -/
    use ⨆ t, ⨆ ht, f t ht
    /-
      case h
      L : Type u_1
      inst✝ : CompleteLattice L
      S : Set (Set (Locale.PT L))
      f : (t : Set (Locale.PT L)) → Membership.mem S t → L
      hf : ∀ (t : Set (Locale.PT L)) (a : Membership.mem S t), Eq (setOf fun x => x  …
      ⊢ Eq (setOf fun x => x (iSup fun t => iSup fun ht => f t ht)) S.sUnion
    -/
    simp_rw [map_iSup, iSup_Prop_eq, setOf_exists, hf, sUnion_eq_biUnion]
    /-
      🎉 no goals
    -/


/-- Characterization of when a subset of the space of points is open. -/
lemma isOpen_iff (U : Set (PT L)) : IsOpen U ↔ ∃ u : L, {x | x u} = U := Iff.rfl


/-- The covariant functor `pt` from the category of locales to the category of
topological spaces, which sends a locale `L` to the topological space `PT L` of homomorphisms
from `L` to `Prop` and a locale homomorphism `f` to a continuous function between the spaces
of points. -/
def pt : Locale ⥤ TopCat where
  obj L := ⟨PT L.unop, inferInstance⟩
                                                          /-
                                                            L : Type u_1
                                                            inst✝ : CompleteLattice L
                                                            X✝ Y✝ : Locale
                                                            f : Quiver.Hom X✝ Y✝
                                                            ⊢ ∀ (s : Set ↑((fun L => { α := Locale.PT ↑(Opposite.unop L), str := inferInst …
                                                          -/
  map f := ⟨fun p ↦ p.comp f.unop, continuous_def.2 <| by rintro s ⟨u, rfl⟩; use f.unop u; rfl⟩
                                                                                           /-
                                                                                             🎉 no goals
                                                                                           -/

/-- The unit of the adjunction between locales and topological spaces, which associates with
a point `x` of the space `X` a point of the locale of opens of `X`. -/
@[simps]
def localePointOfSpacePoint (x : X) : PT (Opens X) where
  toFun := (x ∈ ·)
  map_inf' _ _ := rfl
  map_top' := rfl
                    /-
                      X : Type u_1
                      inst✝ : TopologicalSpace X
                      L : Locale
                      x : X
                      S : Set (TopologicalSpace.Opens X)
                      ⊢ Eq ({ toFun := fun x_1 => Membership.mem x_1 x, map_inf' := ⋯, map_top' := ⋯ …
                    -/
  map_sSup' S := by simp [Prop.exists_iff]
                    /-
                      🎉 no goals
                    -/


/-- The counit is a frame homomorphism. -/
def counitAppCont : FrameHom L (Opens <| PT L) where
  toFun u := ⟨openOfElementHom L u, u, rfl⟩
                     /-
                       X : Type u_1
                       inst✝ : TopologicalSpace X
                       L : Locale
                       a b : ↑(Opposite.unop L)
                       ⊢ Eq ((fun u => { carrier := (Locale.openOfElementHom ↑(Opposite.unop L)) u, i …
                     -/
  map_inf' a b := by simp
                     /-
                       🎉 no goals
                     -/
                 /-
                   X : Type u_1
                   inst✝ : TopologicalSpace X
                   L : Locale
                   ⊢ Eq ({ toFun := fun u => { carrier := (Locale.openOfElementHom ↑(Opposite.uno …
                 -/
  map_top' := by simp
                 /-
                   🎉 no goals
                 -/
                    /-
                      X : Type u_1
                      inst✝ : TopologicalSpace X
                      L : Locale
                      S : Set ↑(Opposite.unop L)
                      ⊢ Eq ({ toFun := fun u => { carrier := (Locale.openOfElementHom ↑(Opposite.uno …
                    -/
  map_sSup' S := by ext; simp
                         /-
                           🎉 no goals
                         -/


/-- The forgetful functor `topToLocale` is left adjoint to the functor `pt`. -/
def adjunctionTopToLocalePT : topToLocale ⊣ pt where
  unit := { app := fun X ↦ ⟨localePointOfSpacePoint X, continuous_def.2 <|
           /-
             X✝ : Type u_1
             inst✝ : TopologicalSpace X✝
             L : Locale
             X : TopCat
             ⊢ ∀ (s : Set ↑((topToLocale.comp Locale.pt).obj X)), IsOpen s → IsOpen (Set.pr …
           -/
        by rintro _ ⟨u, rfl⟩; simpa using u.2⟩ }
                              /-
                                🎉 no goals
                              -/
  counit := { app := fun L ↦ ⟨counitAppCont L⟩ }


