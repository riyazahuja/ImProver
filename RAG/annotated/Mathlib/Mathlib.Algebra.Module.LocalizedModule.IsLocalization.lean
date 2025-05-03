variable {S} in
theorem isLocalizedModule_iff_isLocalization {A Aₛ} [CommSemiring A] [Algebra R A] [CommSemiring Aₛ]
    [Algebra A Aₛ] [Algebra R Aₛ] [IsScalarTower R A Aₛ] :
    IsLocalizedModule S (IsScalarTower.toAlgHom R A Aₛ).toLinearMap ↔
      IsLocalization (Algebra.algebraMapSubmonoid A S) Aₛ := by
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    S : Submonoid R
    A : Type u_2
    Aₛ : Type u_3
    inst✝⁵ : CommSemiring A
    inst✝⁴ : Algebra R A
    inst✝³ : CommSemiring Aₛ
    inst✝² : Algebra A Aₛ
    inst✝¹ : Algebra R Aₛ
    inst✝ : IsScalarTower R A Aₛ
    ⊢ Iff (IsLocalizedModule S (IsScalarTower.toAlgHom R A Aₛ).toLinearMap) (IsLoc …
  -/
  rw [isLocalizedModule_iff, isLocalization_iff]
  /-
    R : Type u_1
    inst✝⁶ : CommSemiring R
    S : Submonoid R
    A : Type u_2
    Aₛ : Type u_3
    inst✝⁵ : CommSemiring A
    inst✝⁴ : Algebra R A
    inst✝³ : CommSemiring Aₛ
    inst✝² : Algebra A Aₛ
    inst✝¹ : Algebra R Aₛ
    inst✝ : IsScalarTower R A Aₛ
    ⊢ Iff (And (∀ (x : Subtype fun x => Membership.mem S x), IsUnit ((algebraMap R …
  -/
  refine and_congr ?_ (and_congr (forall_congr' fun _ ↦ ?_) (forall₂_congr fun _ _ ↦ ?_))
  · simp_rw [← (Algebra.lmul R Aₛ).commutes, Algebra.lmul_isUnit_iff, Subtype.forall,
      Algebra.algebraMapSubmonoid, ← SetLike.mem_coe, Submonoid.coe_map,
      Set.forall_mem_image, ← IsScalarTower.algebraMap_apply]
    /-
      case refine_2
      R : Type u_1
      inst✝⁶ : CommSemiring R
      S : Submonoid R
      A : Type u_2
      Aₛ : Type u_3
      inst✝⁵ : CommSemiring A
      inst✝⁴ : Algebra R A
      inst✝³ : CommSemiring Aₛ
      inst✝² : Algebra A Aₛ
      inst✝¹ : Algebra R Aₛ
      inst✝ : IsScalarTower R A Aₛ
      x✝ : Aₛ
      ⊢ Iff (Exists fun x => Eq (HSMul.hSMul x.2 x✝) ((IsScalarTower.toAlgHom R A Aₛ …
    -/
  · simp_rw [Prod.exists, Subtype.exists, Algebra.algebraMapSubmonoid]
    /-
      case refine_2
      R : Type u_1
      inst✝⁶ : CommSemiring R
      S : Submonoid R
      A : Type u_2
      Aₛ : Type u_3
      inst✝⁵ : CommSemiring A
      inst✝⁴ : Algebra R A
      inst✝³ : CommSemiring Aₛ
      inst✝² : Algebra A Aₛ
      inst✝¹ : Algebra R Aₛ
      inst✝ : IsScalarTower R A Aₛ
      x✝ : Aₛ
      ⊢ Iff (Exists fun a => Exists fun a_1 => Exists fun b => Eq (HSMul.hSMul ⟨a_1, …
    -/
    simp [← IsScalarTower.algebraMap_apply, Submonoid.mk_smul, Algebra.smul_def, mul_comm]
    /-
      🎉 no goals
    -/
    /-
      case refine_3
      R : Type u_1
      inst✝⁶ : CommSemiring R
      S : Submonoid R
      A : Type u_2
      Aₛ : Type u_3
      inst✝⁵ : CommSemiring A
      inst✝⁴ : Algebra R A
      inst✝³ : CommSemiring Aₛ
      inst✝² : Algebra A Aₛ
      inst✝¹ : Algebra R Aₛ
      inst✝ : IsScalarTower R A Aₛ
      x✝¹ x✝ : A
      ⊢ Iff (Eq ((IsScalarTower.toAlgHom R A Aₛ).toLinearMap x✝¹) ((IsScalarTower.to …
    -/
  · congr!; simp_rw [Subtype.exists, Algebra.algebraMapSubmonoid]; simp [Algebra.smul_def]
                                                                   /-
                                                                     🎉 no goals
                                                                   -/


instance {A Aₛ} [CommSemiring A] [Algebra R A][CommSemiring Aₛ] [Algebra A Aₛ] [Algebra R Aₛ]
    [IsScalarTower R A Aₛ] [h : IsLocalization (Algebra.algebraMapSubmonoid A S) Aₛ] :
    IsLocalizedModule S (IsScalarTower.toAlgHom R A Aₛ).toLinearMap :=
  isLocalizedModule_iff_isLocalization.mpr h


lemma isLocalizedModule_iff_isLocalization' (R') [CommSemiring R'] [Algebra R R'] :
    IsLocalizedModule S (Algebra.linearMap R R') ↔ IsLocalization S R' := by
  /-
    R : Type u_1
    inst✝² : CommSemiring R
    S : Submonoid R
    R' : Type u_2
    inst✝¹ : CommSemiring R'
    inst✝ : Algebra R R'
    ⊢ Iff (IsLocalizedModule S (Algebra.linearMap R R')) (IsLocalization S R')
  -/
  convert isLocalizedModule_iff_isLocalization (S := S) (A := R) (Aₛ := R')
  /-
    case h.e'_2.h.e'_3
    R : Type u_1
    inst✝² : CommSemiring R
    S : Submonoid R
    R' : Type u_2
    inst✝¹ : CommSemiring R'
    inst✝ : Algebra R R'
    ⊢ Eq S (Algebra.algebraMapSubmonoid R S)
  -/
  exact (Submonoid.map_id S).symm
  /-
    🎉 no goals
  -/


instance {A} [CommSemiring A] [Algebra R A] [IsLocalization S A] :
    IsLocalizedModule S (Algebra.linearMap R A) :=
  (isLocalizedModule_iff_isLocalization' S _).mpr inferInstance


