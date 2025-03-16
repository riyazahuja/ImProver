theorem HigherFacesVanish.inclusionOfMooreComplexMap (n : ℕ) :
    HigherFacesVanish (n + 1) ((inclusionOfMooreComplexMap X).f (n + 1)) := fun j _ => by
  /-
    A : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} A
    inst✝ : CategoryTheory.Abelian A
    X : CategoryTheory.SimplicialObject A
    n : Nat
    j : Fin (HAdd.hAdd n 1)
    x✝ : LE.le (HAdd.hAdd n 1) (HAdd.hAdd (↑j) (HAdd.hAdd n 1))
    ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.inclusionOfMooreC …
  -/
  dsimp [AlgebraicTopology.inclusionOfMooreComplexMap, NormalizedMooreComplex.objX]
  rw [← factorThru_arrow _ _ (finset_inf_arrow_factors Finset.univ _ j
    (by simp only [Finset.mem_univ])), assoc, kernelSubobject_arrow_comp, comp_zero]


theorem factors_normalizedMooreComplex_PInfty (n : ℕ) :
    Subobject.Factors (NormalizedMooreComplex.objX X n) (PInfty.f n) := by
  /-
    A : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} A
    inst✝ : CategoryTheory.Abelian A
    X : CategoryTheory.SimplicialObject A
    n : Nat
    ⊢ (AlgebraicTopology.NormalizedMooreComplex.objX X n).Factors (AlgebraicTopolo …
  -/
  rcases n with _|n
    /-
      case zero
      A : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} A
      inst✝ : CategoryTheory.Abelian A
      X : CategoryTheory.SimplicialObject A
      ⊢ (AlgebraicTopology.NormalizedMooreComplex.objX X 0).Factors (AlgebraicTopolo …
    -/
  · apply top_factors
    /-
      🎉 no goals
    -/
    /-
      case succ
      A : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} A
      inst✝ : CategoryTheory.Abelian A
      X : CategoryTheory.SimplicialObject A
      n : Nat
      ⊢ (AlgebraicTopology.NormalizedMooreComplex.objX X (HAdd.hAdd n 1)).Factors (A …
    -/
  · rw [PInfty_f, NormalizedMooreComplex.objX, finset_inf_factors]
    /-
      case succ
      A : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} A
      inst✝ : CategoryTheory.Abelian A
      X : CategoryTheory.SimplicialObject A
      n : Nat
      ⊢ ∀ (i : Fin (HAdd.hAdd n 1)), Membership.mem Finset.univ i → (CategoryTheory. …
    -/
    intro i _
    /-
      case succ
      A : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} A
      inst✝ : CategoryTheory.Abelian A
      X : CategoryTheory.SimplicialObject A
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      a✝ : Membership.mem Finset.univ i
      ⊢ (CategoryTheory.Limits.kernelSubobject (X.δ i.succ)).Factors ((AlgebraicTopo …
    -/
    apply kernelSubobject_factors
    /-
      case succ.w
      A : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} A
      inst✝ : CategoryTheory.Abelian A
      X : CategoryTheory.SimplicialObject A
      n : Nat
      i : Fin (HAdd.hAdd n 1)
      a✝ : Membership.mem Finset.univ i
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.DoldKan.P (HAdd.h …
    -/
    exact (HigherFacesVanish.of_P (n + 1) n) i le_add_self
    /-
      🎉 no goals
    -/


/-- `PInfty` factors through the normalized Moore complex -/
@[simps!]
def PInftyToNormalizedMooreComplex (X : SimplicialObject A) : K[X] ⟶ N[X] :=
  ChainComplex.ofHom _ _ _ _ _ _
    (fun n => factorThru _ _ (factors_normalizedMooreComplex_PInfty n)) fun n => by
    rw [← cancel_mono (NormalizedMooreComplex.objX X n).arrow, assoc, assoc, factorThru_arrow,
      ← inclusionOfMooreComplexMap_f, ← normalizedMooreComplex_objD,
      ← (inclusionOfMooreComplexMap X).comm (n + 1) n, inclusionOfMooreComplexMap_f,
      factorThru_arrow_assoc, ← alternatingFaceMapComplex_obj_d]
    /-
      A : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.9731, u_1} A
      inst✝ : CategoryTheory.Abelian A
      X✝ X : CategoryTheory.SimplicialObject A
      n : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInfty.f ( …
    -/
    exact PInfty.comm (n + 1) n
    /-
      🎉 no goals
    -/


@[reassoc (attr := simp)]
theorem PInftyToNormalizedMooreComplex_comp_inclusionOfMooreComplexMap (X : SimplicialObject A) :
                                                                                   /-
                                                                                     A : Type u_1
                                                                                     inst✝¹ : CategoryTheory.Category.{u_2, u_1} A
                                                                                     inst✝ : CategoryTheory.Abelian A
                                                                                     X : CategoryTheory.SimplicialObject A
                                                                                     ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.DoldKan.PInftyToNo …
                                                                                   -/
    PInftyToNormalizedMooreComplex X ≫ inclusionOfMooreComplexMap X = PInfty := by aesop_cat
                                                                                   /-
                                                                                     🎉 no goals
                                                                                   -/


@[reassoc (attr := simp)]
theorem PInftyToNormalizedMooreComplex_naturality {X Y : SimplicialObject A} (f : X ⟶ Y) :
    AlternatingFaceMapComplex.map f ≫ PInftyToNormalizedMooreComplex Y =
      PInftyToNormalizedMooreComplex X ≫ NormalizedMooreComplex.map f := by
  /-
    A : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} A
    inst✝ : CategoryTheory.Abelian A
    X Y : CategoryTheory.SimplicialObject A
    f : Quiver.Hom X Y
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.AlternatingFaceMap …
  -/
  aesop_cat
  /-
    🎉 no goals
  -/


@[reassoc (attr := simp)]
theorem PInfty_comp_PInftyToNormalizedMooreComplex (X : SimplicialObject A) :
                                                                                       /-
                                                                                         A : Type u_1
                                                                                         inst✝¹ : CategoryTheory.Category.{u_2, u_1} A
                                                                                         inst✝ : CategoryTheory.Abelian A
                                                                                         X : CategoryTheory.SimplicialObject A
                                                                                         ⊢ Eq (CategoryTheory.CategoryStruct.comp AlgebraicTopology.DoldKan.PInfty (Alg …
                                                                                       -/
    PInfty ≫ PInftyToNormalizedMooreComplex X = PInftyToNormalizedMooreComplex X := by aesop_cat
                                                                                       /-
                                                                                         🎉 no goals
                                                                                       -/


@[reassoc (attr := simp)]
theorem inclusionOfMooreComplexMap_comp_PInfty (X : SimplicialObject A) :
    inclusionOfMooreComplexMap X ≫ PInfty = inclusionOfMooreComplexMap X := by
  /-
    A : Type u_1
    inst✝¹ : CategoryTheory.Category.{u_2, u_1} A
    inst✝ : CategoryTheory.Abelian A
    X : CategoryTheory.SimplicialObject A
    ⊢ Eq (CategoryTheory.CategoryStruct.comp (AlgebraicTopology.inclusionOfMooreCo …
  -/
  ext (_|n)
    /-
      case h.zero
      A : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} A
      inst✝ : CategoryTheory.Abelian A
      X : CategoryTheory.SimplicialObject A
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (AlgebraicTopology.inclusionOfMooreC …
    -/
  · dsimp
    /-
      case h.zero
      A : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} A
      inst✝ : CategoryTheory.Abelian A
      X : CategoryTheory.SimplicialObject A
      ⊢ Eq (CategoryTheory.CategoryStruct.comp ((AlgebraicTopology.inclusionOfMooreC …
    -/
    simp only [comp_id]
    /-
      🎉 no goals
    -/
    /-
      case h.succ
      A : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} A
      inst✝ : CategoryTheory.Abelian A
      X : CategoryTheory.SimplicialObject A
      n : Nat
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp (AlgebraicTopology.inclusionOfMooreC …
    -/
  · exact (HigherFacesVanish.inclusionOfMooreComplexMap n).comp_P_eq_self
    /-
      🎉 no goals
    -/


instance : Mono (inclusionOfMooreComplexMap X) :=
  ⟨fun _ _ hf => by
    /-
      A : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} A
      inst✝ : CategoryTheory.Abelian A
      X : CategoryTheory.SimplicialObject A
      Z✝ : ChainComplex A Nat
      x✝¹ x✝ : Quiver.Hom Z✝ ((AlgebraicTopology.normalizedMooreComplex A).obj X)
      hf : Eq (CategoryTheory.CategoryStruct.comp x✝¹ (AlgebraicTopology.inclusionOf …
      ⊢ Eq x✝¹ x✝
    -/
    ext n
    /-
      case h
      A : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} A
      inst✝ : CategoryTheory.Abelian A
      X : CategoryTheory.SimplicialObject A
      Z✝ : ChainComplex A Nat
      x✝¹ x✝ : Quiver.Hom Z✝ ((AlgebraicTopology.normalizedMooreComplex A).obj X)
      hf : Eq (CategoryTheory.CategoryStruct.comp x✝¹ (AlgebraicTopology.inclusionOf …
      n : Nat
      ⊢ Eq (x✝¹.f n) (x✝.f n)
    -/
    dsimp
    /-
      case h
      A : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} A
      inst✝ : CategoryTheory.Abelian A
      X : CategoryTheory.SimplicialObject A
      Z✝ : ChainComplex A Nat
      x✝¹ x✝ : Quiver.Hom Z✝ ((AlgebraicTopology.normalizedMooreComplex A).obj X)
      hf : Eq (CategoryTheory.CategoryStruct.comp x✝¹ (AlgebraicTopology.inclusionOf …
      n : Nat
      ⊢ Eq (x✝¹.f n) (x✝.f n)
    -/
    ext
    /-
      case h.h
      A : Type u_1
      inst✝¹ : CategoryTheory.Category.{u_2, u_1} A
      inst✝ : CategoryTheory.Abelian A
      X : CategoryTheory.SimplicialObject A
      Z✝ : ChainComplex A Nat
      x✝¹ x✝ : Quiver.Hom Z✝ ((AlgebraicTopology.normalizedMooreComplex A).obj X)
      hf : Eq (CategoryTheory.CategoryStruct.comp x✝¹ (AlgebraicTopology.inclusionOf …
      n : Nat
      ⊢ Eq (CategoryTheory.CategoryStruct.comp (x✝¹.f n) (AlgebraicTopology.Normaliz …
    -/
    exact HomologicalComplex.congr_hom hf n⟩
    /-
      🎉 no goals
    -/


/-- `inclusionOfMooreComplexMap X` is a split mono. -/
def splitMonoInclusionOfMooreComplexMap (X : SimplicialObject A) :
    SplitMono (inclusionOfMooreComplexMap X) where
  retraction := PInftyToNormalizedMooreComplex X
  id := by
    simp only [← cancel_mono (inclusionOfMooreComplexMap X), assoc, id_comp,
      PInftyToNormalizedMooreComplex_comp_inclusionOfMooreComplexMap,
      inclusionOfMooreComplexMap_comp_PInfty]


/-- When the category `A` is abelian,
the functor `N₁ : SimplicialObject A ⥤ Karoubi (ChainComplex A ℕ)` defined
using `PInfty` identifies to the composition of the normalized Moore complex functor
and the inclusion in the Karoubi envelope. -/
def N₁_iso_normalizedMooreComplex_comp_toKaroubi : N₁ ≅ normalizedMooreComplex A ⋙ toKaroubi _ where
  hom :=
    { app := fun X =>
        { f := PInftyToNormalizedMooreComplex X
                     /-
                       A : Type u_1
                       inst✝¹ : CategoryTheory.Category.{?u.32668, u_1} A
                       inst✝ : CategoryTheory.Abelian A
                       X✝ X : CategoryTheory.SimplicialObject A
                       ⊢ Eq (AlgebraicTopology.DoldKan.PInftyToNormalizedMooreComplex X) (CategoryThe …
                     -/
          comm := by erw [comp_id, PInfty_comp_PInftyToNormalizedMooreComplex] }
                     /-
                       🎉 no goals
                     -/
      naturality := fun X Y f => by
        simp only [Functor.comp_map, normalizedMooreComplex_map,
          PInftyToNormalizedMooreComplex_naturality, Karoubi.hom_ext_iff, Karoubi.comp_f, N₁_map_f,
          PInfty_comp_PInftyToNormalizedMooreComplex_assoc, toKaroubi_map_f, assoc] }
  inv :=
    { app := fun X =>
        { f := inclusionOfMooreComplexMap X
                     /-
                       A : Type u_1
                       inst✝¹ : CategoryTheory.Category.{?u.32668, u_1} A
                       inst✝ : CategoryTheory.Abelian A
                       X✝ X : CategoryTheory.SimplicialObject A
                       ⊢ Eq (AlgebraicTopology.inclusionOfMooreComplexMap X) (CategoryTheory.Category …
                     -/
          comm := by erw [inclusionOfMooreComplexMap_comp_PInfty, id_comp] }
                     /-
                       🎉 no goals
                     -/
      naturality := fun X Y f => by
        /-
          A : Type u_1
          inst✝¹ : CategoryTheory.Category.{?u.32668, u_1} A
          inst✝ : CategoryTheory.Abelian A
          X✝ X Y : CategoryTheory.SimplicialObject A
          f : Quiver.Hom X Y
          ⊢ Eq (CategoryTheory.CategoryStruct.comp (((AlgebraicTopology.normalizedMooreC …
        -/
        ext
        simp only [Functor.comp_obj, normalizedMooreComplex_obj, toKaroubi_obj_X,
          NormalizedMooreComplex.obj_X, N₁_obj_X, AlternatingFaceMapComplex.obj_X, Functor.comp_map,
          normalizedMooreComplex_map, Karoubi.comp_f, toKaroubi_map_f, HomologicalComplex.comp_f,
          NormalizedMooreComplex.map_f, inclusionOfMooreComplexMap_f, NormalizedMooreComplex.objX,
          factorThru_arrow, N₁_map_f, inclusionOfMooreComplexMap_comp_PInfty_assoc,
          AlternatingFaceMapComplex.map_f]
         }
  hom_inv_id := by
    /-
      A : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.32668, u_1} A
      inst✝ : CategoryTheory.Abelian A
      X : CategoryTheory.SimplicialObject A
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { app := fun X => { f := AlgebraicTop …
    -/
    ext X : 3
    simp only [PInftyToNormalizedMooreComplex_comp_inclusionOfMooreComplexMap,
      NatTrans.comp_app, Karoubi.comp_f, N₁_obj_p, NatTrans.id_app, Karoubi.id_f]
  inv_hom_id := by
    /-
      A : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.32668, u_1} A
      inst✝ : CategoryTheory.Abelian A
      X : CategoryTheory.SimplicialObject A
      ⊢ Eq (CategoryTheory.CategoryStruct.comp { app := fun X => { f := AlgebraicTop …
    -/
    ext X : 3
    /-
      case w.h.h
      A : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.32668, u_1} A
      inst✝ : CategoryTheory.Abelian A
      X✝ X : CategoryTheory.SimplicialObject A
      ⊢ Eq ((CategoryTheory.CategoryStruct.comp { app := fun X => { f := AlgebraicTo …
    -/
    rw [← cancel_mono (inclusionOfMooreComplexMap X)]
    simp only [NatTrans.comp_app, Karoubi.comp_f, assoc, NatTrans.id_app, Karoubi.id_f,
      PInftyToNormalizedMooreComplex_comp_inclusionOfMooreComplexMap,
      inclusionOfMooreComplexMap_comp_PInfty]
    /-
      case w.h.h
      A : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.32668, u_1} A
      inst✝ : CategoryTheory.Abelian A
      X✝ X : CategoryTheory.SimplicialObject A
      ⊢ Eq (AlgebraicTopology.inclusionOfMooreComplexMap X) (CategoryTheory.Category …
    -/
    dsimp only [Functor.comp_obj, toKaroubi]
    /-
      case w.h.h
      A : Type u_1
      inst✝¹ : CategoryTheory.Category.{?u.32668, u_1} A
      inst✝ : CategoryTheory.Abelian A
      X✝ X : CategoryTheory.SimplicialObject A
      ⊢ Eq (AlgebraicTopology.inclusionOfMooreComplexMap X) (CategoryTheory.Category …
    -/
    rw [id_comp]
    /-
      🎉 no goals
    -/


