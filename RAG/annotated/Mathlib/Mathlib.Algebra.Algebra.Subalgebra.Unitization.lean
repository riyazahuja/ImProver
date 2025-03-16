/-- Turn a `Subalgebra` into a `NonUnitalSubalgebra` by forgetting that it contains `1`. -/
def Subalgebra.toNonUnitalSubalgebra (S : Subalgebra R A) : NonUnitalSubalgebra R A :=
  { S with
    smul_mem' := fun r _x hx => S.smul_mem hx r }


theorem Subalgebra.one_mem_toNonUnitalSubalgebra (S : Subalgebra R A) :
    (1 : A) ∈ S.toNonUnitalSubalgebra :=
  S.one_mem


/-- Turn a non-unital subalgebra containing `1` into a subalgebra. -/
def NonUnitalSubalgebra.toSubalgebra (S : NonUnitalSubalgebra R A) (h1 : (1 : A) ∈ S) :
    Subalgebra R A :=
  { S with
    one_mem' := h1
    algebraMap_mem' := fun r =>
      (Algebra.algebraMap_eq_smul_one (R := R) (A := A) r).symm ▸ SMulMemClass.smul_mem r h1 }


theorem Subalgebra.toNonUnitalSubalgebra_toSubalgebra (S : Subalgebra R A) :
                                                             /-
                                                               R : Type u_1
                                                               A : Type u_2
                                                               inst✝² : CommSemiring R
                                                               inst✝¹ : Semiring A
                                                               inst✝ : Algebra R A
                                                               S : Subalgebra R A
                                                               ⊢ Eq (S.toNonUnitalSubalgebra.toSubalgebra ⋯) S
                                                             -/
    S.toNonUnitalSubalgebra.toSubalgebra S.one_mem = S := by cases S; rfl
                                                                      /-
                                                                        🎉 no goals
                                                                      -/


theorem NonUnitalSubalgebra.toSubalgebra_toNonUnitalSubalgebra (S : NonUnitalSubalgebra R A)
    (h1 : (1 : A) ∈ S) : (NonUnitalSubalgebra.toSubalgebra S h1).toNonUnitalSubalgebra = S := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝² : CommSemiring R
    inst✝¹ : Semiring A
    inst✝ : Algebra R A
    S : NonUnitalSubalgebra R A
    h1 : Membership.mem S 1
    ⊢ Eq (S.toSubalgebra h1).toNonUnitalSubalgebra S
  -/
  cases S; rfl
           /-
             🎉 no goals
           -/


open Submodule in
lemma Algebra.adjoin_nonUnitalSubalgebra_eq_span (s : NonUnitalSubalgebra R A) :
    Subalgebra.toSubmodule (adjoin R (s : Set A)) = span R {1} ⊔ s.toSubmodule := by
  rw [adjoin_eq_span, Submonoid.closure_eq_one_union, span_union, ← NonUnitalAlgebra.adjoin_eq_span,
      NonUnitalAlgebra.adjoin_eq]


lemma NonUnitalAlgebra.adjoin_le_algebra_adjoin (s : Set A) :
    adjoin R s ≤ (Algebra.adjoin R s).toNonUnitalSubalgebra :=
  adjoin_le Algebra.subset_adjoin


lemma Algebra.adjoin_nonUnitalSubalgebra (s : Set A) :
    adjoin R (NonUnitalAlgebra.adjoin R s : Set A) = adjoin R s :=
  le_antisymm
    (adjoin_le <| NonUnitalAlgebra.adjoin_le_algebra_adjoin R s)
    (adjoin_le <| (NonUnitalAlgebra.subset_adjoin R).trans subset_adjoin)


theorem lift_range_le {f : A →ₙₐ[R] C} {S : Subalgebra R C} :
    (lift f).range ≤ S ↔ NonUnitalAlgHom.range f ≤ S.toNonUnitalSubalgebra := by
  /-
    R : Type u_1
    A : Type u_2
    C : Type u_3
    inst✝⁶ : CommSemiring R
    inst✝⁵ : NonUnitalSemiring A
    inst✝⁴ : Module R A
    inst✝³ : SMulCommClass R A A
    inst✝² : IsScalarTower R A A
    inst✝¹ : Semiring C
    inst✝ : Algebra R C
    f : NonUnitalAlgHom (MonoidHom.id R) A C
    S : Subalgebra R C
    ⊢ Iff (LE.le (Unitization.lift f).range S) (LE.le (NonUnitalAlgHom.range f) S. …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      R : Type u_1
      A : Type u_2
      C : Type u_3
      inst✝⁶ : CommSemiring R
      inst✝⁵ : NonUnitalSemiring A
      inst✝⁴ : Module R A
      inst✝³ : SMulCommClass R A A
      inst✝² : IsScalarTower R A A
      inst✝¹ : Semiring C
      inst✝ : Algebra R C
      f : NonUnitalAlgHom (MonoidHom.id R) A C
      S : Subalgebra R C
      h : LE.le (Unitization.lift f).range S
      ⊢ LE.le (NonUnitalAlgHom.range f) S.toNonUnitalSubalgebra
    -/
  · rintro - ⟨x, rfl⟩
    /-
      case refine_1.intro
      R : Type u_1
      A : Type u_2
      C : Type u_3
      inst✝⁶ : CommSemiring R
      inst✝⁵ : NonUnitalSemiring A
      inst✝⁴ : Module R A
      inst✝³ : SMulCommClass R A A
      inst✝² : IsScalarTower R A A
      inst✝¹ : Semiring C
      inst✝ : Algebra R C
      f : NonUnitalAlgHom (MonoidHom.id R) A C
      S : Subalgebra R C
      h : LE.le (Unitization.lift f).range S
      x : A
      ⊢ Membership.mem S.toNonUnitalSubalgebra (↑f x)
    -/
    exact @h (f x) ⟨x, by simp⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      A : Type u_2
      C : Type u_3
      inst✝⁶ : CommSemiring R
      inst✝⁵ : NonUnitalSemiring A
      inst✝⁴ : Module R A
      inst✝³ : SMulCommClass R A A
      inst✝² : IsScalarTower R A A
      inst✝¹ : Semiring C
      inst✝ : Algebra R C
      f : NonUnitalAlgHom (MonoidHom.id R) A C
      S : Subalgebra R C
      h : LE.le (NonUnitalAlgHom.range f) S.toNonUnitalSubalgebra
      ⊢ LE.le (Unitization.lift f).range S
    -/
  · rintro - ⟨x, rfl⟩
    induction x with
    | _ r a => simpa using add_mem (algebraMap_mem S r) (h ⟨a, rfl⟩)


theorem lift_range (f : A →ₙₐ[R] C) :
    (lift f).range = Algebra.adjoin R (NonUnitalAlgHom.range f : Set C) :=
                                 /-
                                   R : Type u_1
                                   A : Type u_2
                                   C : Type u_3
                                   inst✝⁶ : CommSemiring R
                                   inst✝⁵ : NonUnitalSemiring A
                                   inst✝⁴ : Module R A
                                   inst✝³ : SMulCommClass R A A
                                   inst✝² : IsScalarTower R A A
                                   inst✝¹ : Semiring C
                                   inst✝ : Algebra R C
                                   f : NonUnitalAlgHom (MonoidHom.id R) A C
                                   c : Subalgebra R C
                                   ⊢ Iff (LE.le (Unitization.lift f).range c) (LE.le (Algebra.adjoin R ↑(NonUnita …
                                 -/
  eq_of_forall_ge_iff fun c ↦ by rw [lift_range_le, Algebra.adjoin_le_iff]; rfl
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


/-- The natural `R`-algebra homomorphism from the unitization of a non-unital subalgebra into
the algebra containing it. -/
def unitization : Unitization R s →ₐ[R] A :=
  Unitization.lift (NonUnitalSubalgebraClass.subtype s)


@[simp]
theorem unitization_apply (x : Unitization R s) :
    unitization s x = algebraMap R A x.fst + x.snd :=
  rfl


theorem unitization_range : (unitization s).range = Algebra.adjoin R (s : Set A) := by
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝³ : CommSemiring R
    inst✝² : Semiring A
    inst✝¹ : Algebra R A
    inst✝ : SetLike S A
    hSA : NonUnitalSubsemiringClass S A
    hSRA : SMulMemClass S R A
    s : S
    ⊢ Eq (NonUnitalSubalgebra.unitization s).range (Algebra.adjoin R ↑s)
  -/
  rw [unitization, Unitization.lift_range]
  simp only [NonUnitalAlgHom.coe_range, NonUnitalSubalgebraClass.coeSubtype,
    Subtype.range_coe_subtype, SetLike.mem_coe]
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝³ : CommSemiring R
    inst✝² : Semiring A
    inst✝¹ : Algebra R A
    inst✝ : SetLike S A
    hSA : NonUnitalSubsemiringClass S A
    hSRA : SMulMemClass S R A
    s : S
    ⊢ Eq (Algebra.adjoin R (setOf fun x => Membership.mem s x)) (Algebra.adjoin R  …
  -/
  rfl
  /-
    🎉 no goals
  -/


/-- A sufficient condition for injectivity of `NonUnitalSubalgebra.unitization` when the scalars
are a commutative ring. When the scalars are a field, one should use the more natural
`NonUnitalStarSubalgebra.unitization_injective` whose hypothesis is easier to verify. -/
theorem _root_.AlgHomClass.unitization_injective' {F R S A : Type*} [CommRing R] [Ring A]
    [Algebra R A] [SetLike S A] [hSA : NonUnitalSubringClass S A] [hSRA : SMulMemClass S R A]
    (s : S) (h : ∀ r, r ≠ 0 → algebraMap R A r ∉ s)
    [FunLike F (Unitization R s) A] [AlgHomClass F R (Unitization R s) A]
    (f : F) (hf : ∀ x : s, f x = x) : Function.Injective f := by
  /-
    F : Type u_1
    R : Type u_2
    S : Type u_3
    A : Type u_4
    inst✝⁵ : CommRing R
    inst✝⁴ : Ring A
    inst✝³ : Algebra R A
    inst✝² : SetLike S A
    hSA : NonUnitalSubringClass S A
    hSRA : SMulMemClass S R A
    s : S
    h : ∀ (r : R), Ne r 0 → Not (Membership.mem s ((algebraMap R A) r))
    inst✝¹ : FunLike F (Unitization R (Subtype fun x => Membership.mem s x)) A
    inst✝ : AlgHomClass F R (Unitization R (Subtype fun x => Membership.mem s x)) A
    f : F
    hf : ∀ (x : Subtype fun x => Membership.mem s x), Eq (f ↑x) ↑x
    ⊢ Function.Injective ⇑f
  -/
  refine (injective_iff_map_eq_zero f).mpr fun x hx => ?_
  induction x with
  | inl_add_inr r a =>
    simp_rw [map_add, hf, ← Unitization.algebraMap_eq_inl, AlgHomClass.commutes] at hx
    rw [add_eq_zero_iff_eq_neg] at hx ⊢
    by_cases hr : r = 0
    · ext
      · simp [hr]
      · simpa [hr] using hx
    · exact (h r hr <| hx ▸ (neg_mem a.property)).elim


/-- This is a generic version which allows us to prove both
`NonUnitalSubalgebra.unitization_injective` and `NonUnitalStarSubalgebra.unitization_injective`. -/
theorem _root_.AlgHomClass.unitization_injective {F R S A : Type*} [Field R] [Ring A]
    [Algebra R A] [SetLike S A] [hSA : NonUnitalSubringClass S A] [hSRA : SMulMemClass S R A]
    (s : S) (h1 : 1 ∉ s) [FunLike F (Unitization R s) A] [AlgHomClass F R (Unitization R s) A]
    (f : F) (hf : ∀ x : s, f x = x) : Function.Injective f := by
  /-
    F : Type u_1
    R : Type u_2
    S : Type u_3
    A : Type u_4
    inst✝⁵ : Field R
    inst✝⁴ : Ring A
    inst✝³ : Algebra R A
    inst✝² : SetLike S A
    hSA : NonUnitalSubringClass S A
    hSRA : SMulMemClass S R A
    s : S
    h1 : Not (Membership.mem s 1)
    inst✝¹ : FunLike F (Unitization R (Subtype fun x => Membership.mem s x)) A
    inst✝ : AlgHomClass F R (Unitization R (Subtype fun x => Membership.mem s x)) A
    f : F
    hf : ∀ (x : Subtype fun x => Membership.mem s x), Eq (f ↑x) ↑x
    ⊢ Function.Injective ⇑f
  -/
  refine AlgHomClass.unitization_injective' s (fun r hr hr' ↦ ?_) f hf
  /-
    F : Type u_1
    R : Type u_2
    S : Type u_3
    A : Type u_4
    inst✝⁵ : Field R
    inst✝⁴ : Ring A
    inst✝³ : Algebra R A
    inst✝² : SetLike S A
    hSA : NonUnitalSubringClass S A
    hSRA : SMulMemClass S R A
    s : S
    h1 : Not (Membership.mem s 1)
    inst✝¹ : FunLike F (Unitization R (Subtype fun x => Membership.mem s x)) A
    inst✝ : AlgHomClass F R (Unitization R (Subtype fun x => Membership.mem s x)) A
    f : F
    hf : ∀ (x : Subtype fun x => Membership.mem s x), Eq (f ↑x) ↑x
    r : R
    hr : Ne r 0
    hr' : Membership.mem s ((algebraMap R A) r)
    ⊢ False
  -/
  rw [Algebra.algebraMap_eq_smul_one] at hr'
  /-
    F : Type u_1
    R : Type u_2
    S : Type u_3
    A : Type u_4
    inst✝⁵ : Field R
    inst✝⁴ : Ring A
    inst✝³ : Algebra R A
    inst✝² : SetLike S A
    hSA : NonUnitalSubringClass S A
    hSRA : SMulMemClass S R A
    s : S
    h1 : Not (Membership.mem s 1)
    inst✝¹ : FunLike F (Unitization R (Subtype fun x => Membership.mem s x)) A
    inst✝ : AlgHomClass F R (Unitization R (Subtype fun x => Membership.mem s x)) A
    f : F
    hf : ∀ (x : Subtype fun x => Membership.mem s x), Eq (f ↑x) ↑x
    r : R
    hr : Ne r 0
    hr' : Membership.mem s (HSMul.hSMul r 1)
    ⊢ False
  -/
  exact h1 <| inv_smul_smul₀ hr (1 : A) ▸ SMulMemClass.smul_mem r⁻¹ hr'
  /-
    🎉 no goals
  -/


theorem unitization_injective (h1 : (1 : A) ∉ s) : Function.Injective (unitization s) :=
                                                                    /-
                                                                      R : Type u_1
                                                                      S : Type u_2
                                                                      A : Type u_3
                                                                      inst✝³ : Field R
                                                                      inst✝² : Ring A
                                                                      inst✝¹ : Algebra R A
                                                                      inst✝ : SetLike S A
                                                                      hSA : NonUnitalSubringClass S A
                                                                      hSRA : SMulMemClass S R A
                                                                      s : S
                                                                      h1 : Not (Membership.mem s 1)
                                                                      x✝ : Subtype fun x => Membership.mem s x
                                                                      ⊢ Eq ((NonUnitalSubalgebra.unitization s) ↑x✝) ↑x✝
                                                                    -/
  AlgHomClass.unitization_injective s h1 (unitization s) fun _ ↦ by simp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- If a `NonUnitalSubalgebra` over a field does not contain `1`, then its unitization is
isomorphic to its `Algebra.adjoin`. -/
@[simps! apply_coe]
noncomputable def unitizationAlgEquiv (h1 : (1 : A) ∉ s) :
    Unitization R s ≃ₐ[R] Algebra.adjoin R (s : Set A) :=
  let algHom : Unitization R s →ₐ[R] Algebra.adjoin R (s : Set A) :=
    ((unitization s).codRestrict _
      fun x ↦ (unitization_range s).le <| AlgHom.mem_range_self _ x)
  AlgEquiv.ofBijective algHom <| by
    /-
      R : Type u_1
      S : Type u_2
      A : Type u_3
      inst✝³ : Field R
      inst✝² : Ring A
      inst✝¹ : Algebra R A
      inst✝ : SetLike S A
      hSA : NonUnitalSubringClass S A
      hSRA : SMulMemClass S R A
      s : S
      h1 : Not (Membership.mem s 1)
      algHom : AlgHom R (Unitization R (Subtype fun x => Membership.mem s x)) (Subty …
      ⊢ Function.Bijective ⇑algHom
    -/
    refine ⟨?_, fun x ↦ ?_⟩
    · have := AlgHomClass.unitization_injective s h1
        ((Subalgebra.val _).comp algHom) fun _ ↦ by simp [algHom]
      /-
        case refine_1
        R : Type u_1
        S : Type u_2
        A : Type u_3
        inst✝³ : Field R
        inst✝² : Ring A
        inst✝¹ : Algebra R A
        inst✝ : SetLike S A
        hSA : NonUnitalSubringClass S A
        hSRA : SMulMemClass S R A
        s : S
        h1 : Not (Membership.mem s 1)
        algHom : AlgHom R (Unitization R (Subtype fun x => Membership.mem s x)) (Subty …
        this : Function.Injective ⇑((Algebra.adjoin R ↑s).val.comp algHom)
        ⊢ Function.Injective ⇑algHom
      -/
      rw [AlgHom.coe_comp] at this
      /-
        case refine_1
        R : Type u_1
        S : Type u_2
        A : Type u_3
        inst✝³ : Field R
        inst✝² : Ring A
        inst✝¹ : Algebra R A
        inst✝ : SetLike S A
        hSA : NonUnitalSubringClass S A
        hSRA : SMulMemClass S R A
        s : S
        h1 : Not (Membership.mem s 1)
        algHom : AlgHom R (Unitization R (Subtype fun x => Membership.mem s x)) (Subty …
        this : Function.Injective (Function.comp ⇑(Algebra.adjoin R ↑s).val ⇑algHom)
        ⊢ Function.Injective ⇑algHom
      -/
      exact this.of_comp
      /-
        🎉 no goals
      -/
    · obtain (⟨a, ha⟩ : (x : A) ∈ (unitization s).range) :=
        (unitization_range s).ge x.property
      /-
        case refine_2.intro
        R : Type u_1
        S : Type u_2
        A : Type u_3
        inst✝³ : Field R
        inst✝² : Ring A
        inst✝¹ : Algebra R A
        inst✝ : SetLike S A
        hSA : NonUnitalSubringClass S A
        hSRA : SMulMemClass S R A
        s : S
        h1 : Not (Membership.mem s 1)
        algHom : AlgHom R (Unitization R (Subtype fun x => Membership.mem s x)) (Subty …
        x : Subtype fun x => Membership.mem (Algebra.adjoin R ↑s) x
        a : Unitization R (Subtype fun x => Membership.mem s x)
        ha : Eq ((NonUnitalSubalgebra.unitization s).toRingHom a) ↑x
        ⊢ Exists fun a => Eq (algHom a) x
      -/
      exact ⟨a, Subtype.ext ha⟩
      /-
        🎉 no goals
      -/


/-- Turn a `Subsemiring` into a `NonUnitalSubsemiring` by forgetting that it contains `1`. -/
def Subsemiring.toNonUnitalSubsemiring (S : Subsemiring R) : NonUnitalSubsemiring R :=
  { S with }


theorem Subsemiring.toNonUnitalSubsemiring_injective :
    Function.Injective (toNonUnitalSubsemiring : Subsemiring R → _) :=
  fun S₁ S₂ h => SetLike.ext'_iff.2 (
    show (S₁.toNonUnitalSubsemiring : Set R) = S₂ from SetLike.ext'_iff.1 h)


@[simp]
theorem Subsemiring.toNonUnitalSubsemiring_inj {S₁ S₂ : Subsemiring R} :
    S₁.toNonUnitalSubsemiring = S₂.toNonUnitalSubsemiring ↔ S₁ = S₂ :=
  toNonUnitalSubsemiring_injective.eq_iff


@[simp]
theorem Subsemiring.mem_toNonUnitalSubsemiring {S : Subsemiring R}
    {x : R} : x ∈ S.toNonUnitalSubsemiring ↔ x ∈ S := Iff.rfl


@[simp]
theorem Subsemiring.coe_toNonUnitalSubsemiring (S : Subsemiring R) :
  (S.toNonUnitalSubsemiring : Set R) = S := rfl


theorem Subsemiring.one_mem_toNonUnitalSubsemiring (S : Subsemiring R) :
    (1 : R) ∈ S.toNonUnitalSubsemiring :=
  S.one_mem


@[simp]
theorem Submonoid.subsemiringClosure_toNonUnitalSubsemiring {M : Submonoid R} :
    M.subsemiringClosure.toNonUnitalSubsemiring = .closure M := by
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    M : Submonoid R
    ⊢ Eq M.subsemiringClosure.toNonUnitalSubsemiring (NonUnitalSubsemiring.closure …
  -/
  refine Eq.symm (NonUnitalSubsemiring.closure_eq_of_le ?_ (fun _ hx => ?_))
    /-
      case refine_1
      R : Type u_1
      inst✝ : NonAssocSemiring R
      M : Submonoid R
      ⊢ HasSubset.Subset ↑M ↑M.subsemiringClosure.toNonUnitalSubsemiring
    -/
  · simp [Submonoid.subsemiringClosure_coe]
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      inst✝ : NonAssocSemiring R
      M : Submonoid R
      x✝ : R
      hx : Membership.mem M.subsemiringClosure.toNonUnitalSubsemiring x✝
      ⊢ Membership.mem (NonUnitalSubsemiring.closure ↑M) x✝
    -/
  · simp [Submonoid.subsemiringClosure_mem] at hx
    /-
      case refine_2
      R : Type u_1
      inst✝ : NonAssocSemiring R
      M : Submonoid R
      x✝ : R
      hx : Membership.mem (AddSubmonoid.closure ↑M) x✝
      ⊢ Membership.mem (NonUnitalSubsemiring.closure ↑M) x✝
    -/
                                                          /-
                                                            🎉 no goals
                                                          -/
                                                          /-
                                                            🎉 no goals
                                                          -/
    induction hx using AddSubmonoid.closure_induction <;> aesop
                                                          /-
                                                            🎉 no goals
                                                          -/


/-- Turn a non-unital subsemiring containing `1` into a subsemiring. -/
def NonUnitalSubsemiring.toSubsemiring (S : NonUnitalSubsemiring R) (h1 : (1 : R) ∈ S) :
    Subsemiring R :=
  { S with
    one_mem' := h1 }


theorem Subsemiring.toNonUnitalSubsemiring_toSubsemiring (S : Subsemiring R) :
                                                               /-
                                                                 R : Type u_1
                                                                 inst✝ : NonAssocSemiring R
                                                                 S : Subsemiring R
                                                                 ⊢ Eq (S.toNonUnitalSubsemiring.toSubsemiring ⋯) S
                                                               -/
    S.toNonUnitalSubsemiring.toSubsemiring S.one_mem = S := by cases S; rfl
                                                                        /-
                                                                          🎉 no goals
                                                                        -/


theorem NonUnitalSubsemiring.toSubsemiring_toNonUnitalSubsemiring (S : NonUnitalSubsemiring R)
    (h1 : (1 : R) ∈ S) : (NonUnitalSubsemiring.toSubsemiring S h1).toNonUnitalSubsemiring = S := by
  /-
    R : Type u_1
    inst✝ : NonAssocSemiring R
    S : NonUnitalSubsemiring R
    h1 : Membership.mem S 1
    ⊢ Eq (S.toSubsemiring h1).toNonUnitalSubsemiring S
  -/
  cases S; rfl
           /-
             🎉 no goals
           -/


/-- The natural `ℕ`-algebra homomorphism from the unitization of a non-unital subsemiring to
its `Subsemiring.closure`. -/
def unitization : Unitization ℕ s →ₐ[ℕ] R :=
  NonUnitalSubalgebra.unitization (hSRA := AddSubmonoidClass.nsmulMemClass) s


@[simp]
theorem unitization_apply (x : Unitization ℕ s) : unitization s x = x.fst + x.snd :=
  rfl


theorem unitization_range :
    (unitization s).range = subalgebraOfSubsemiring (.closure s) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : Semiring R
    inst✝ : SetLike S R
    hSR : NonUnitalSubsemiringClass S R
    s : S
    ⊢ Eq (NonUnitalSubsemiring.unitization s).range (subalgebraOfSubsemiring (Subs …
  -/
  have := AddSubmonoidClass.nsmulMemClass (S := S)
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : Semiring R
    inst✝ : SetLike S R
    hSR : NonUnitalSubsemiringClass S R
    s : S
    this : SMulMemClass S Nat R
    ⊢ Eq (NonUnitalSubsemiring.unitization s).range (subalgebraOfSubsemiring (Subs …
  -/
  rw [unitization, NonUnitalSubalgebra.unitization_range (hSRA := this), Algebra.adjoin_nat]
  /-
    🎉 no goals
  -/


/-- Turn a `Subring` into a `NonUnitalSubring` by forgetting that it contains `1`. -/
def Subring.toNonUnitalSubring (S : Subring R) : NonUnitalSubring R :=
  { S with }


theorem Subring.one_mem_toNonUnitalSubring (S : Subring R) : (1 : R) ∈ S.toNonUnitalSubring :=
  S.one_mem


/-- Turn a non-unital subring containing `1` into a subring. -/
def NonUnitalSubring.toSubring (S : NonUnitalSubring R) (h1 : (1 : R) ∈ S) : Subring R :=
  { S with
    one_mem' := h1 }


theorem Subring.toNonUnitalSubring_toSubring (S : Subring R) :
                                                       /-
                                                         R : Type u_1
                                                         inst✝ : Ring R
                                                         S : Subring R
                                                         ⊢ Eq (S.toNonUnitalSubring.toSubring ⋯) S
                                                       -/
    S.toNonUnitalSubring.toSubring S.one_mem = S := by cases S; rfl
                                                                /-
                                                                  🎉 no goals
                                                                -/


theorem NonUnitalSubring.toSubring_toNonUnitalSubring (S : NonUnitalSubring R) (h1 : (1 : R) ∈ S) :
                                                                   /-
                                                                     R : Type u_1
                                                                     inst✝ : Ring R
                                                                     S : NonUnitalSubring R
                                                                     h1 : Membership.mem S 1
                                                                     ⊢ Eq (S.toSubring h1).toNonUnitalSubring S
                                                                   -/
    (NonUnitalSubring.toSubring S h1).toNonUnitalSubring = S := by cases S; rfl
                                                                            /-
                                                                              🎉 no goals
                                                                            -/


/-- The natural `ℤ`-algebra homomorphism from the unitization of a non-unital subring to
its `Subring.closure`. -/
def unitization : Unitization ℤ s →ₐ[ℤ] R :=
  NonUnitalSubalgebra.unitization (hSRA := AddSubgroupClass.zsmulMemClass) s


@[simp]
theorem unitization_apply (x : Unitization ℤ s) : unitization s x = x.fst + x.snd :=
  rfl


theorem unitization_range :
    (unitization s).range = subalgebraOfSubring (.closure s) := by
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : Ring R
    inst✝ : SetLike S R
    hSR : NonUnitalSubringClass S R
    s : S
    ⊢ Eq (NonUnitalSubring.unitization s).range (subalgebraOfSubring (Subring.clos …
  -/
  have := AddSubgroupClass.zsmulMemClass (S := S)
  /-
    R : Type u_1
    S : Type u_2
    inst✝¹ : Ring R
    inst✝ : SetLike S R
    hSR : NonUnitalSubringClass S R
    s : S
    this : SMulMemClass S Int R
    ⊢ Eq (NonUnitalSubring.unitization s).range (subalgebraOfSubring (Subring.clos …
  -/
  rw [unitization, NonUnitalSubalgebra.unitization_range (hSRA := this), Algebra.adjoin_int]
  /-
    🎉 no goals
  -/


/-- Turn a `StarSubalgebra` into a `NonUnitalStarSubalgebra` by forgetting that it contains `1`. -/
def StarSubalgebra.toNonUnitalStarSubalgebra (S : StarSubalgebra R A) :
    NonUnitalStarSubalgebra R A :=
  { S with
    carrier := S.carrier
    smul_mem' := fun r _x hx => S.smul_mem hx r }


theorem StarSubalgebra.one_mem_toNonUnitalStarSubalgebra (S : StarSubalgebra R A) :
    (1 : A) ∈ S.toNonUnitalStarSubalgebra :=
  S.one_mem'


/-- Turn a non-unital star subalgebra containing `1` into a `StarSubalgebra`. -/
def NonUnitalStarSubalgebra.toStarSubalgebra (S : NonUnitalStarSubalgebra R A) (h1 : (1 : A) ∈ S) :
    StarSubalgebra R A :=
  { S with
    carrier := S.carrier
    one_mem' := h1
    algebraMap_mem' := fun r =>
      (Algebra.algebraMap_eq_smul_one (R := R) (A := A) r).symm ▸ SMulMemClass.smul_mem r h1 }


theorem StarSubalgebra.toNonUnitalStarSubalgebra_toStarSubalgebra (S : StarSubalgebra R A) :
                                                                      /-
                                                                        R : Type u_1
                                                                        A : Type u_2
                                                                        inst✝⁵ : CommSemiring R
                                                                        inst✝⁴ : StarRing R
                                                                        inst✝³ : Semiring A
                                                                        inst✝² : StarRing A
                                                                        inst✝¹ : Algebra R A
                                                                        inst✝ : StarModule R A
                                                                        S : StarSubalgebra R A
                                                                        ⊢ Eq (S.toNonUnitalStarSubalgebra.toStarSubalgebra ⋯) S
                                                                      -/
    S.toNonUnitalStarSubalgebra.toStarSubalgebra S.one_mem' = S := by cases S; rfl
                                                                               /-
                                                                                 🎉 no goals
                                                                               -/


theorem NonUnitalStarSubalgebra.toStarSubalgebra_toNonUnitalStarSubalgebra
    (S : NonUnitalStarSubalgebra R A) (h1 : (1 : A) ∈ S) :
    (S.toStarSubalgebra h1).toNonUnitalStarSubalgebra = S := by
  /-
    R : Type u_1
    A : Type u_2
    inst✝⁵ : CommSemiring R
    inst✝⁴ : StarRing R
    inst✝³ : Semiring A
    inst✝² : StarRing A
    inst✝¹ : Algebra R A
    inst✝ : StarModule R A
    S : NonUnitalStarSubalgebra R A
    h1 : Membership.mem S 1
    ⊢ Eq (S.toStarSubalgebra h1).toNonUnitalStarSubalgebra S
  -/
  cases S; rfl
           /-
             🎉 no goals
           -/


open Submodule in
lemma StarAlgebra.adjoin_nonUnitalStarSubalgebra_eq_span (s : NonUnitalStarSubalgebra R A) :
    Subalgebra.toSubmodule (adjoin R (s : Set A)).toSubalgebra = span R {1} ⊔ s.toSubmodule := by
  rw [adjoin_eq_span, Submonoid.closure_eq_one_union, span_union,
    ← NonUnitalStarAlgebra.adjoin_eq_span, NonUnitalStarAlgebra.adjoin_eq]


lemma NonUnitalStarAlgebra.adjoin_le_starAlgebra_adjoin (s : Set A) :
    adjoin R s ≤ (StarAlgebra.adjoin R s).toNonUnitalStarSubalgebra :=
  adjoin_le <| StarAlgebra.subset_adjoin R s


lemma StarAlgebra.adjoin_nonUnitalStarSubalgebra (s : Set A) :
    adjoin R (NonUnitalStarAlgebra.adjoin R s : Set A) = adjoin R s :=
  le_antisymm
    (adjoin_le <| NonUnitalStarAlgebra.adjoin_le_starAlgebra_adjoin R s)
    (adjoin_le <| (NonUnitalStarAlgebra.subset_adjoin R s).trans <| subset_adjoin R _)


theorem starLift_range_le
    {f : A →⋆ₙₐ[R] C} {S : StarSubalgebra R C} :
    (starLift f).range ≤ S ↔ NonUnitalStarAlgHom.range f ≤ S.toNonUnitalStarSubalgebra := by
  /-
    R : Type u_1
    A : Type u_2
    C : Type u_3
    inst✝¹¹ : CommSemiring R
    inst✝¹⁰ : NonUnitalSemiring A
    inst✝⁹ : StarRing R
    inst✝⁸ : StarRing A
    inst✝⁷ : Module R A
    inst✝⁶ : SMulCommClass R A A
    inst✝⁵ : IsScalarTower R A A
    inst✝⁴ : StarModule R A
    inst✝³ : Semiring C
    inst✝² : StarRing C
    inst✝¹ : Algebra R C
    inst✝ : StarModule R C
    f : NonUnitalStarAlgHom R A C
    S : StarSubalgebra R C
    ⊢ Iff (LE.le (Unitization.starLift f).range S) (LE.le (NonUnitalStarAlgHom.ran …
  -/
  refine ⟨fun h ↦ ?_, fun h ↦ ?_⟩
    /-
      case refine_1
      R : Type u_1
      A : Type u_2
      C : Type u_3
      inst✝¹¹ : CommSemiring R
      inst✝¹⁰ : NonUnitalSemiring A
      inst✝⁹ : StarRing R
      inst✝⁸ : StarRing A
      inst✝⁷ : Module R A
      inst✝⁶ : SMulCommClass R A A
      inst✝⁵ : IsScalarTower R A A
      inst✝⁴ : StarModule R A
      inst✝³ : Semiring C
      inst✝² : StarRing C
      inst✝¹ : Algebra R C
      inst✝ : StarModule R C
      f : NonUnitalStarAlgHom R A C
      S : StarSubalgebra R C
      h : LE.le (Unitization.starLift f).range S
      ⊢ LE.le (NonUnitalStarAlgHom.range f) S.toNonUnitalStarSubalgebra
    -/
  · rintro - ⟨x, rfl⟩
    /-
      case refine_1.intro
      R : Type u_1
      A : Type u_2
      C : Type u_3
      inst✝¹¹ : CommSemiring R
      inst✝¹⁰ : NonUnitalSemiring A
      inst✝⁹ : StarRing R
      inst✝⁸ : StarRing A
      inst✝⁷ : Module R A
      inst✝⁶ : SMulCommClass R A A
      inst✝⁵ : IsScalarTower R A A
      inst✝⁴ : StarModule R A
      inst✝³ : Semiring C
      inst✝² : StarRing C
      inst✝¹ : Algebra R C
      inst✝ : StarModule R C
      f : NonUnitalStarAlgHom R A C
      S : StarSubalgebra R C
      h : LE.le (Unitization.starLift f).range S
      x : A
      ⊢ Membership.mem S.toNonUnitalStarSubalgebra (↑(NonUnitalAlgHomClass.toNonUnit …
    -/
    exact @h (f x) ⟨x, by simp⟩
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      R : Type u_1
      A : Type u_2
      C : Type u_3
      inst✝¹¹ : CommSemiring R
      inst✝¹⁰ : NonUnitalSemiring A
      inst✝⁹ : StarRing R
      inst✝⁸ : StarRing A
      inst✝⁷ : Module R A
      inst✝⁶ : SMulCommClass R A A
      inst✝⁵ : IsScalarTower R A A
      inst✝⁴ : StarModule R A
      inst✝³ : Semiring C
      inst✝² : StarRing C
      inst✝¹ : Algebra R C
      inst✝ : StarModule R C
      f : NonUnitalStarAlgHom R A C
      S : StarSubalgebra R C
      h : LE.le (NonUnitalStarAlgHom.range f) S.toNonUnitalStarSubalgebra
      ⊢ LE.le (Unitization.starLift f).range S
    -/
  · rintro - ⟨x, rfl⟩
    induction x with
    | _ r a => simpa using add_mem (algebraMap_mem S r) (h ⟨a, rfl⟩)


theorem starLift_range (f : A →⋆ₙₐ[R] C) :
    (starLift f).range = StarAlgebra.adjoin R (NonUnitalStarAlgHom.range f : Set C) :=
  eq_of_forall_ge_iff fun c ↦ by
    /-
      R : Type u_1
      A : Type u_2
      C : Type u_3
      inst✝¹¹ : CommSemiring R
      inst✝¹⁰ : NonUnitalSemiring A
      inst✝⁹ : StarRing R
      inst✝⁸ : StarRing A
      inst✝⁷ : Module R A
      inst✝⁶ : SMulCommClass R A A
      inst✝⁵ : IsScalarTower R A A
      inst✝⁴ : StarModule R A
      inst✝³ : Semiring C
      inst✝² : StarRing C
      inst✝¹ : Algebra R C
      inst✝ : StarModule R C
      f : NonUnitalStarAlgHom R A C
      c : StarSubalgebra R C
      ⊢ Iff (LE.le (Unitization.starLift f).range c) (LE.le (StarAlgebra.adjoin R ↑( …
    -/
    rw [starLift_range_le, StarAlgebra.adjoin_le_iff]
    /-
      R : Type u_1
      A : Type u_2
      C : Type u_3
      inst✝¹¹ : CommSemiring R
      inst✝¹⁰ : NonUnitalSemiring A
      inst✝⁹ : StarRing R
      inst✝⁸ : StarRing A
      inst✝⁷ : Module R A
      inst✝⁶ : SMulCommClass R A A
      inst✝⁵ : IsScalarTower R A A
      inst✝⁴ : StarModule R A
      inst✝³ : Semiring C
      inst✝² : StarRing C
      inst✝¹ : Algebra R C
      inst✝ : StarModule R C
      f : NonUnitalStarAlgHom R A C
      c : StarSubalgebra R C
      ⊢ Iff (LE.le (NonUnitalStarAlgHom.range f) c.toNonUnitalStarSubalgebra) (HasSu …
    -/
    rfl
    /-
      🎉 no goals
    -/


/-- The natural star `R`-algebra homomorphism from the unitization of a non-unital star subalgebra
to its `StarAlgebra.adjoin`. -/
def unitization : Unitization R s →⋆ₐ[R] A :=
  Unitization.starLift <| NonUnitalStarSubalgebraClass.subtype s


@[simp]
theorem unitization_apply (x : Unitization R s) : unitization s x = algebraMap R A x.fst + x.snd :=
  rfl


theorem unitization_range : (unitization s).range = StarAlgebra.adjoin R s := by
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁷ : CommSemiring R
    inst✝⁶ : StarRing R
    inst✝⁵ : Semiring A
    inst✝⁴ : StarRing A
    inst✝³ : Algebra R A
    inst✝² : StarModule R A
    inst✝¹ : SetLike S A
    hSA : NonUnitalSubsemiringClass S A
    hSRA : SMulMemClass S R A
    inst✝ : StarMemClass S A
    s : S
    ⊢ Eq (NonUnitalStarSubalgebra.unitization s).range (StarAlgebra.adjoin R ↑s)
  -/
  rw [unitization, Unitization.starLift_range]
  simp only [NonUnitalStarAlgHom.coe_range, NonUnitalStarSubalgebraClass.coeSubtype,
    Subtype.range_coe_subtype]
  /-
    R : Type u_1
    S : Type u_2
    A : Type u_3
    inst✝⁷ : CommSemiring R
    inst✝⁶ : StarRing R
    inst✝⁵ : Semiring A
    inst✝⁴ : StarRing A
    inst✝³ : Algebra R A
    inst✝² : StarModule R A
    inst✝¹ : SetLike S A
    hSA : NonUnitalSubsemiringClass S A
    hSRA : SMulMemClass S R A
    inst✝ : StarMemClass S A
    s : S
    ⊢ Eq (StarAlgebra.adjoin R (setOf fun x => Membership.mem s x)) (StarAlgebra.a …
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem unitization_injective (h1 : (1 : A) ∉ s) : Function.Injective (unitization s) :=
                                                                    /-
                                                                      R : Type u_1
                                                                      S : Type u_2
                                                                      A : Type u_3
                                                                      inst✝⁷ : Field R
                                                                      inst✝⁶ : StarRing R
                                                                      inst✝⁵ : Ring A
                                                                      inst✝⁴ : StarRing A
                                                                      inst✝³ : Algebra R A
                                                                      inst✝² : StarModule R A
                                                                      inst✝¹ : SetLike S A
                                                                      hSA : NonUnitalSubringClass S A
                                                                      hSRA : SMulMemClass S R A
                                                                      inst✝ : StarMemClass S A
                                                                      s : S
                                                                      h1 : Not (Membership.mem s 1)
                                                                      x✝ : Subtype fun x => Membership.mem s x
                                                                      ⊢ Eq ((NonUnitalStarSubalgebra.unitization s) ↑x✝) ↑x✝
                                                                    -/
  AlgHomClass.unitization_injective s h1 (unitization s) fun _ ↦ by simp
                                                                    /-
                                                                      🎉 no goals
                                                                    -/


/-- If a `NonUnitalStarSubalgebra` over a field does not contain `1`, then its unitization is
isomorphic to its `StarAlgebra.adjoin`. -/
@[simps! apply_coe]
noncomputable def unitizationStarAlgEquiv (h1 : (1 : A) ∉ s) :
    Unitization R s ≃⋆ₐ[R] StarAlgebra.adjoin R (s : Set A) :=
  let starAlgHom : Unitization R s →⋆ₐ[R] StarAlgebra.adjoin R (s : Set A) :=
    ((unitization s).codRestrict _
      fun x ↦ (unitization_range s).le <| Set.mem_range_self x)
  StarAlgEquiv.ofBijective starAlgHom <| by
    /-
      R : Type u_1
      S : Type u_2
      A : Type u_3
      inst✝⁷ : Field R
      inst✝⁶ : StarRing R
      inst✝⁵ : Ring A
      inst✝⁴ : StarRing A
      inst✝³ : Algebra R A
      inst✝² : StarModule R A
      inst✝¹ : SetLike S A
      hSA : NonUnitalSubringClass S A
      hSRA : SMulMemClass S R A
      inst✝ : StarMemClass S A
      s : S
      h1 : Not (Membership.mem s 1)
      starAlgHom : StarAlgHom R (Unitization R (Subtype fun x => Membership.mem s x) …
      ⊢ Function.Bijective ⇑starAlgHom
    -/
    refine ⟨?_, fun x ↦ ?_⟩
    · have := AlgHomClass.unitization_injective s h1 ((StarSubalgebra.subtype _).comp starAlgHom)
        fun _ ↦ by simp [starAlgHom]
      /-
        case refine_1
        R : Type u_1
        S : Type u_2
        A : Type u_3
        inst✝⁷ : Field R
        inst✝⁶ : StarRing R
        inst✝⁵ : Ring A
        inst✝⁴ : StarRing A
        inst✝³ : Algebra R A
        inst✝² : StarModule R A
        inst✝¹ : SetLike S A
        hSA : NonUnitalSubringClass S A
        hSRA : SMulMemClass S R A
        inst✝ : StarMemClass S A
        s : S
        h1 : Not (Membership.mem s 1)
        starAlgHom : StarAlgHom R (Unitization R (Subtype fun x => Membership.mem s x) …
        this : Function.Injective ⇑((StarAlgebra.adjoin R ↑s).subtype.comp starAlgHom)
        ⊢ Function.Injective ⇑starAlgHom
      -/
      rw [StarAlgHom.coe_comp] at this
      /-
        case refine_1
        R : Type u_1
        S : Type u_2
        A : Type u_3
        inst✝⁷ : Field R
        inst✝⁶ : StarRing R
        inst✝⁵ : Ring A
        inst✝⁴ : StarRing A
        inst✝³ : Algebra R A
        inst✝² : StarModule R A
        inst✝¹ : SetLike S A
        hSA : NonUnitalSubringClass S A
        hSRA : SMulMemClass S R A
        inst✝ : StarMemClass S A
        s : S
        h1 : Not (Membership.mem s 1)
        starAlgHom : StarAlgHom R (Unitization R (Subtype fun x => Membership.mem s x) …
        this : Function.Injective (Function.comp ⇑(StarAlgebra.adjoin R ↑s).subtype ⇑s …
        ⊢ Function.Injective ⇑starAlgHom
      -/
      exact this.of_comp
      /-
        🎉 no goals
      -/
    · obtain (⟨a, ha⟩ : (x : A) ∈ (unitization s).range) :=
        (unitization_range s).ge x.property
      /-
        case refine_2.intro
        R : Type u_1
        S : Type u_2
        A : Type u_3
        inst✝⁷ : Field R
        inst✝⁶ : StarRing R
        inst✝⁵ : Ring A
        inst✝⁴ : StarRing A
        inst✝³ : Algebra R A
        inst✝² : StarModule R A
        inst✝¹ : SetLike S A
        hSA : NonUnitalSubringClass S A
        hSRA : SMulMemClass S R A
        inst✝ : StarMemClass S A
        s : S
        h1 : Not (Membership.mem s 1)
        starAlgHom : StarAlgHom R (Unitization R (Subtype fun x => Membership.mem s x) …
        x : Subtype fun x => Membership.mem (StarAlgebra.adjoin R ↑s) x
        a : Unitization R (Subtype fun x => Membership.mem s x)
        ha : Eq ((NonUnitalStarSubalgebra.unitization s).toRingHom a) ↑x
        ⊢ Exists fun a => Eq (starAlgHom a) x
      -/
      exact ⟨a, Subtype.ext ha⟩
      /-
        🎉 no goals
      -/


