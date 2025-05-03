theorem IntermediateField.coe_isIntegral_iff {R : Type*} [CommRing R] [Algebra R K] [Algebra R L]
    [IsScalarTower R K L] {x : S} : IsIntegral R (x : L) ↔ IsIntegral R x :=
  isIntegral_algHom_iff (S.val.restrictScalars R) Subtype.val_injective


/-- Turn an algebraic subalgebra into an intermediate field, `Subalgebra.IsAlgebraic` version. -/
def Subalgebra.IsAlgebraic.toIntermediateField {S : Subalgebra K L} (hS : S.IsAlgebraic) :
    IntermediateField K L where
  toSubalgebra := S
  inv_mem' x hx := Algebra.adjoin_le_iff.mpr
    (Set.singleton_subset_iff.mpr hx) (hS x hx).isIntegral.inv_mem_adjoin


/-- Turn an algebraic subalgebra into an intermediate field, `Algebra.IsAlgebraic` version. -/
abbrev Algebra.IsAlgebraic.toIntermediateField (S : Subalgebra K L) [Algebra.IsAlgebraic K S] :
    IntermediateField K L := (S.isAlgebraic_iff.mpr ‹_›).toIntermediateField


instance isAlgebraic_tower_bot [Algebra.IsAlgebraic K L] : Algebra.IsAlgebraic K S :=
  Algebra.IsAlgebraic.of_injective S.val S.val.injective


instance isAlgebraic_tower_top [Algebra.IsAlgebraic K L] : Algebra.IsAlgebraic S L :=
  Algebra.IsAlgebraic.tower_top (K := K) S


instance finiteDimensional_left [FiniteDimensional K L] : FiniteDimensional K F := .left K F L

instance finiteDimensional_right [FiniteDimensional K L] : FiniteDimensional F L := .right K F L


@[simp]
theorem rank_eq_rank_subalgebra : Module.rank K F.toSubalgebra = Module.rank K F :=
  rfl


@[simp]
theorem finrank_eq_finrank_subalgebra : finrank K F.toSubalgebra = finrank K F :=
  rfl


/-- If `F ≤ E` are two intermediate fields of `L / K` such that `[E : K] ≤ [F : K]` are finite,
then `F = E`. -/
theorem eq_of_le_of_finrank_le [hfin : FiniteDimensional K E] (h_le : F ≤ E)
    (h_finrank : finrank K E ≤ finrank K F) : F = E :=
  haveI : Module.Finite K E.toSubalgebra := hfin
  toSubalgebra_injective <| Subalgebra.eq_of_le_of_finrank_le h_le h_finrank


/-- If `F ≤ E` are two intermediate fields of `L / K` such that `[F : K] = [E : K]` are finite,
then `F = E`. -/
theorem eq_of_le_of_finrank_eq [FiniteDimensional K E] (h_le : F ≤ E)
    (h_finrank : finrank K F = finrank K E) : F = E :=
  eq_of_le_of_finrank_le h_le h_finrank.ge

-- If `F ≤ E` are two intermediate fields of a finite extension `L / K` such that
-- `[L : F] ≤ [L : E]`, then `F = E`. Marked as private since it's a direct corollary of
-- `eq_of_le_of_finrank_le'` (the `FiniteDimensional K L` implies `FiniteDimensional F L`
-- automatically by typeclass resolution).

private theorem eq_of_le_of_finrank_le'' [FiniteDimensional K L] (h_le : F ≤ E)
    (h_finrank : finrank F L ≤ finrank E L) : F = E := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    F E : IntermediateField K L
    inst✝ : FiniteDimensional K L
    h_le : LE.le F E
    h_finrank : LE.le (Module.finrank (Subtype fun x => Membership.mem F x) L) (Mo …
    ⊢ Eq F E
  -/
  apply eq_of_le_of_finrank_le h_le
  /-
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    F E : IntermediateField K L
    inst✝ : FiniteDimensional K L
    h_le : LE.le F E
    h_finrank : LE.le (Module.finrank (Subtype fun x => Membership.mem F x) L) (Mo …
    ⊢ LE.le (Module.finrank K (Subtype fun x => Membership.mem E x)) (Module.finra …
  -/
  have h1 := finrank_mul_finrank K F L
  /-
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    F E : IntermediateField K L
    inst✝ : FiniteDimensional K L
    h_le : LE.le F E
    h_finrank : LE.le (Module.finrank (Subtype fun x => Membership.mem F x) L) (Mo …
    h1 : Eq (HMul.hMul (Module.finrank K (Subtype fun x => Membership.mem F x)) (M …
    ⊢ LE.le (Module.finrank K (Subtype fun x => Membership.mem E x)) (Module.finra …
  -/
  have h2 := finrank_mul_finrank K E L
  /-
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    F E : IntermediateField K L
    inst✝ : FiniteDimensional K L
    h_le : LE.le F E
    h_finrank : LE.le (Module.finrank (Subtype fun x => Membership.mem F x) L) (Mo …
    h1 : Eq (HMul.hMul (Module.finrank K (Subtype fun x => Membership.mem F x)) (M …
    h2 : Eq (HMul.hMul (Module.finrank K (Subtype fun x => Membership.mem E x)) (M …
    ⊢ LE.le (Module.finrank K (Subtype fun x => Membership.mem E x)) (Module.finra …
  -/
  have h3 : 0 < finrank E L := finrank_pos
  /-
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    F E : IntermediateField K L
    inst✝ : FiniteDimensional K L
    h_le : LE.le F E
    h_finrank : LE.le (Module.finrank (Subtype fun x => Membership.mem F x) L) (Mo …
    h1 : Eq (HMul.hMul (Module.finrank K (Subtype fun x => Membership.mem F x)) (M …
    h2 : Eq (HMul.hMul (Module.finrank K (Subtype fun x => Membership.mem E x)) (M …
    h3 : LT.lt 0 (Module.finrank (Subtype fun x => Membership.mem E x) L)
    ⊢ LE.le (Module.finrank K (Subtype fun x => Membership.mem E x)) (Module.finra …
  -/
  nlinarith
  /-
    🎉 no goals
  -/


/-- If `F ≤ E` are two intermediate fields of `L / K` such that `[L : F] ≤ [L : E]` are finite,
then `F = E`. -/
theorem eq_of_le_of_finrank_le' [FiniteDimensional F L] (h_le : F ≤ E)
    (h_finrank : finrank F L ≤ finrank E L) : F = E := by
  /-
    K : Type u_1
    L : Type u_2
    inst✝³ : Field K
    inst✝² : Field L
    inst✝¹ : Algebra K L
    F E : IntermediateField K L
    inst✝ : FiniteDimensional (Subtype fun x => Membership.mem F x) L
    h_le : LE.le F E
    h_finrank : LE.le (Module.finrank (Subtype fun x => Membership.mem F x) L) (Mo …
    ⊢ Eq F E
  -/
  refine le_antisymm h_le (fun l hl ↦ ?_)
  rwa [← mem_extendScalars (le_refl F), eq_of_le_of_finrank_le''
    ((extendScalars_le_extendScalars_iff (le_refl F) h_le).2 h_le) h_finrank, mem_extendScalars]


/-- If `F ≤ E` are two intermediate fields of `L / K` such that `[L : F] = [L : E]` are finite,
then `F = E`. -/
theorem eq_of_le_of_finrank_eq' [FiniteDimensional F L] (h_le : F ≤ E)
    (h_finrank : finrank F L = finrank E L) : F = E :=
  eq_of_le_of_finrank_le' h_le h_finrank.le


theorem isAlgebraic_iff {x : S} : IsAlgebraic K x ↔ IsAlgebraic K (x : L) :=
  (isAlgebraic_algebraMap_iff (algebraMap S L).injective).symm


theorem isIntegral_iff {x : S} : IsIntegral K x ↔ IsIntegral K (x : L) :=
  (isIntegral_algHom_iff S.val S.val.injective).symm


theorem minpoly_eq (x : S) : minpoly K x = minpoly K (x : L) :=
  (minpoly.algebraMap_eq (algebraMap S L).injective x).symm


/-- If `L/K` is algebraic, the `K`-subalgebras of `L` are all fields. -/
def subalgebraEquivIntermediateField [Algebra.IsAlgebraic K L] :
    Subalgebra K L ≃o IntermediateField K L where
  toFun S := S.toIntermediateField fun x hx => S.inv_mem_of_algebraic
    (Algebra.IsAlgebraic.isAlgebraic ((⟨x, hx⟩ : S) : L))
  invFun S := S.toSubalgebra
  left_inv _ := toSubalgebra_toIntermediateField _ _
  right_inv := toIntermediateField_toSubalgebra
  map_rel_iff' := Iff.rfl


@[simp]
theorem mem_subalgebraEquivIntermediateField [Algebra.IsAlgebraic K L] {S : Subalgebra K L}
    {x : L} : x ∈ subalgebraEquivIntermediateField S ↔ x ∈ S :=
  Iff.rfl


@[simp]
theorem mem_subalgebraEquivIntermediateField_symm [Algebra.IsAlgebraic K L]
    {S : IntermediateField K L} {x : L} :
    x ∈ subalgebraEquivIntermediateField.symm S ↔ x ∈ S :=
  Iff.rfl

