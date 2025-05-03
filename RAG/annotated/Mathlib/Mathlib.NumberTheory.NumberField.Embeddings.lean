/-- There are finitely many embeddings of a number field. -/
noncomputable instance : Fintype (K →+* A) :=
  Fintype.ofEquiv (K →ₐ[ℚ] A) RingHom.equivRatAlgHom.symm


/-- The number of embeddings of a number field is equal to its finrank. -/
theorem card : Fintype.card (K →+* A) = finrank ℚ K := by
  /-
    K : Type u_1
    inst✝⁴ : Field K
    inst✝³ : NumberField K
    A : Type u_2
    inst✝² : Field A
    inst✝¹ : CharZero A
    inst✝ : IsAlgClosed A
    ⊢ Eq (Fintype.card (RingHom K A)) (Module.finrank Rat K)
  -/
  rw [Fintype.ofEquiv_card RingHom.equivRatAlgHom.symm, AlgHom.card]
  /-
    🎉 no goals
  -/


instance : Nonempty (K →+* A) := by
  /-
    K : Type u_1
    inst✝⁴ : Field K
    inst✝³ : NumberField K
    A : Type u_2
    inst✝² : Field A
    inst✝¹ : CharZero A
    inst✝ : IsAlgClosed A
    ⊢ Nonempty (RingHom K A)
  -/
  rw [← Fintype.card_pos_iff, NumberField.Embeddings.card K A]
  /-
    K : Type u_1
    inst✝⁴ : Field K
    inst✝³ : NumberField K
    A : Type u_2
    inst✝² : Field A
    inst✝¹ : CharZero A
    inst✝ : IsAlgClosed A
    ⊢ LT.lt 0 (Module.finrank Rat K)
  -/
  exact Module.finrank_pos
  /-
    🎉 no goals
  -/


/-- Let `A` be an algebraically closed field and let `x ∈ K`, with `K` a number field.
The images of `x` by the embeddings of `K` in `A` are exactly the roots in `A` of
the minimal polynomial of `x` over `ℚ`. -/
theorem range_eval_eq_rootSet_minpoly :
    (range fun φ : K →+* A => φ x) = (minpoly ℚ x).rootSet A := by
  /-
    K : Type u_1
    A : Type u_2
    inst✝⁴ : Field K
    inst✝³ : NumberField K
    inst✝² : Field A
    inst✝¹ : Algebra Rat A
    inst✝ : IsAlgClosed A
    x : K
    ⊢ Eq (Set.range fun φ => φ x) ((minpoly Rat x).rootSet A)
  -/
  convert (NumberField.isAlgebraic K).range_eval_eq_rootSet_minpoly A x using 1
  /-
    case h.e'_2
    K : Type u_1
    A : Type u_2
    inst✝⁴ : Field K
    inst✝³ : NumberField K
    inst✝² : Field A
    inst✝¹ : Algebra Rat A
    inst✝ : IsAlgClosed A
    x : K
    ⊢ Eq (Set.range fun φ => φ x) (Set.range fun ψ => ψ x)
  -/
  ext a
  /-
    case h.e'_2.h
    K : Type u_1
    A : Type u_2
    inst✝⁴ : Field K
    inst✝³ : NumberField K
    inst✝² : Field A
    inst✝¹ : Algebra Rat A
    inst✝ : IsAlgClosed A
    x : K
    a : A
    ⊢ Iff (Membership.mem (Set.range fun φ => φ x) a) (Membership.mem (Set.range f …
  -/
  exact ⟨fun ⟨φ, hφ⟩ => ⟨φ.toRatAlgHom, hφ⟩, fun ⟨φ, hφ⟩ => ⟨φ.toRingHom, hφ⟩⟩
  /-
    🎉 no goals
  -/


theorem coeff_bdd_of_norm_le {B : ℝ} {x : K} (h : ∀ φ : K →+* A, ‖φ x‖ ≤ B) (i : ℕ) :
    ‖(minpoly ℚ x).coeff i‖ ≤ max B 1 ^ finrank ℚ K * (finrank ℚ K).choose (finrank ℚ K / 2) := by
  /-
    K : Type u_1
    inst✝⁴ : Field K
    inst✝³ : NumberField K
    A : Type u_2
    inst✝² : NormedField A
    inst✝¹ : IsAlgClosed A
    inst✝ : NormedAlgebra Rat A
    B : Real
    x : K
    h : ∀ (φ : RingHom K A), LE.le (Norm.norm (φ x)) B
    i : Nat
    ⊢ LE.le (Norm.norm ((minpoly Rat x).coeff i)) (HMul.hMul (HPow.hPow (Max.max B …
  -/
  have hx := Algebra.IsSeparable.isIntegral ℚ x
  /-
    K : Type u_1
    inst✝⁴ : Field K
    inst✝³ : NumberField K
    A : Type u_2
    inst✝² : NormedField A
    inst✝¹ : IsAlgClosed A
    inst✝ : NormedAlgebra Rat A
    B : Real
    x : K
    h : ∀ (φ : RingHom K A), LE.le (Norm.norm (φ x)) B
    i : Nat
    hx : IsIntegral Rat x
    ⊢ LE.le (Norm.norm ((minpoly Rat x).coeff i)) (HMul.hMul (HPow.hPow (Max.max B …
  -/
  rw [← norm_algebraMap' A, ← coeff_map (algebraMap ℚ A)]
  refine coeff_bdd_of_roots_le _ (minpoly.monic hx)
      (IsAlgClosed.splits_codomain _) (minpoly.natDegree_le x) (fun z hz => ?_) i
  classical
  rw [← Multiset.mem_toFinset] at hz
  obtain ⟨φ, rfl⟩ := (range_eval_eq_rootSet_minpoly K A x).symm.subset hz
  exact h φ


/-- Let `B` be a real number. The set of algebraic integers in `K` whose conjugates are all
smaller in norm than `B` is finite. -/
theorem finite_of_norm_le (B : ℝ) : {x : K | IsIntegral ℤ x ∧ ∀ φ : K →+* A, ‖φ x‖ ≤ B}.Finite := by
  /-
    K : Type u_1
    inst✝⁴ : Field K
    inst✝³ : NumberField K
    A : Type u_2
    inst✝² : NormedField A
    inst✝¹ : IsAlgClosed A
    inst✝ : NormedAlgebra Rat A
    B : Real
    ⊢ (setOf fun x => And (IsIntegral Int x) (∀ (φ : RingHom K A), LE.le (Norm.nor …
  -/
  let C := Nat.ceil (max B 1 ^ finrank ℚ K * (finrank ℚ K).choose (finrank ℚ K / 2))
  /-
    K : Type u_1
    inst✝⁴ : Field K
    inst✝³ : NumberField K
    A : Type u_2
    inst✝² : NormedField A
    inst✝¹ : IsAlgClosed A
    inst✝ : NormedAlgebra Rat A
    B : Real
    C : Nat := Nat.ceil (HMul.hMul (HPow.hPow (Max.max B 1) (Module.finrank Rat K) …
    ⊢ (setOf fun x => And (IsIntegral Int x) (∀ (φ : RingHom K A), LE.le (Norm.nor …
  -/
  have := bUnion_roots_finite (algebraMap ℤ K) (finrank ℚ K) (finite_Icc (-C : ℤ) C)
  /-
    K : Type u_1
    inst✝⁴ : Field K
    inst✝³ : NumberField K
    A : Type u_2
    inst✝² : NormedField A
    inst✝¹ : IsAlgClosed A
    inst✝ : NormedAlgebra Rat A
    B : Real
    C : Nat := Nat.ceil (HMul.hMul (HPow.hPow (Max.max B 1) (Module.finrank Rat K) …
    this : (Set.iUnion fun f => Set.iUnion fun x => ↑(Polynomial.map (algebraMap I …
    ⊢ (setOf fun x => And (IsIntegral Int x) (∀ (φ : RingHom K A), LE.le (Norm.nor …
  -/
  refine this.subset fun x hx => ?_; simp_rw [mem_iUnion]
  /-
    K : Type u_1
    inst✝⁴ : Field K
    inst✝³ : NumberField K
    A : Type u_2
    inst✝² : NormedField A
    inst✝¹ : IsAlgClosed A
    inst✝ : NormedAlgebra Rat A
    B : Real
    C : Nat := Nat.ceil (HMul.hMul (HPow.hPow (Max.max B 1) (Module.finrank Rat K) …
    this : (Set.iUnion fun f => Set.iUnion fun x => ↑(Polynomial.map (algebraMap I …
    x : K
    hx : Membership.mem (setOf fun x => And (IsIntegral Int x) (∀ (φ : RingHom K A …
    ⊢ Exists fun i => Exists fun i_1 => Membership.mem (↑(Polynomial.map (algebraM …
  -/
  have h_map_ℚ_minpoly := minpoly.isIntegrallyClosed_eq_field_fractions' ℚ hx.1
  /-
    K : Type u_1
    inst✝⁴ : Field K
    inst✝³ : NumberField K
    A : Type u_2
    inst✝² : NormedField A
    inst✝¹ : IsAlgClosed A
    inst✝ : NormedAlgebra Rat A
    B : Real
    C : Nat := Nat.ceil (HMul.hMul (HPow.hPow (Max.max B 1) (Module.finrank Rat K) …
    this : (Set.iUnion fun f => Set.iUnion fun x => ↑(Polynomial.map (algebraMap I …
    x : K
    hx : Membership.mem (setOf fun x => And (IsIntegral Int x) (∀ (φ : RingHom K A …
    h_map_ℚ_minpoly : Eq (minpoly Rat x) (Polynomial.map (algebraMap Int Rat) (min …
    ⊢ Exists fun i => Exists fun i_1 => Membership.mem (↑(Polynomial.map (algebraM …
  -/
  refine ⟨_, ⟨?_, fun i => ?_⟩, mem_rootSet.2 ⟨minpoly.ne_zero hx.1, minpoly.aeval ℤ x⟩⟩
    /-
      case refine_1
      K : Type u_1
      inst✝⁴ : Field K
      inst✝³ : NumberField K
      A : Type u_2
      inst✝² : NormedField A
      inst✝¹ : IsAlgClosed A
      inst✝ : NormedAlgebra Rat A
      B : Real
      C : Nat := Nat.ceil (HMul.hMul (HPow.hPow (Max.max B 1) (Module.finrank Rat K) …
      this : (Set.iUnion fun f => Set.iUnion fun x => ↑(Polynomial.map (algebraMap I …
      x : K
      hx : Membership.mem (setOf fun x => And (IsIntegral Int x) (∀ (φ : RingHom K A …
      h_map_ℚ_minpoly : Eq (minpoly Rat x) (Polynomial.map (algebraMap Int Rat) (min …
      ⊢ LE.le (minpoly Int x).natDegree (Module.finrank Rat K)
    -/
  · rw [← (minpoly.monic hx.1).natDegree_map (algebraMap ℤ ℚ), ← h_map_ℚ_minpoly]
    /-
      case refine_1
      K : Type u_1
      inst✝⁴ : Field K
      inst✝³ : NumberField K
      A : Type u_2
      inst✝² : NormedField A
      inst✝¹ : IsAlgClosed A
      inst✝ : NormedAlgebra Rat A
      B : Real
      C : Nat := Nat.ceil (HMul.hMul (HPow.hPow (Max.max B 1) (Module.finrank Rat K) …
      this : (Set.iUnion fun f => Set.iUnion fun x => ↑(Polynomial.map (algebraMap I …
      x : K
      hx : Membership.mem (setOf fun x => And (IsIntegral Int x) (∀ (φ : RingHom K A …
      h_map_ℚ_minpoly : Eq (minpoly Rat x) (Polynomial.map (algebraMap Int Rat) (min …
      ⊢ LE.le (minpoly Rat x).natDegree (Module.finrank Rat K)
    -/
    exact minpoly.natDegree_le x
    /-
      🎉 no goals
    -/
  /-
    case refine_2
    K : Type u_1
    inst✝⁴ : Field K
    inst✝³ : NumberField K
    A : Type u_2
    inst✝² : NormedField A
    inst✝¹ : IsAlgClosed A
    inst✝ : NormedAlgebra Rat A
    B : Real
    C : Nat := Nat.ceil (HMul.hMul (HPow.hPow (Max.max B 1) (Module.finrank Rat K) …
    this : (Set.iUnion fun f => Set.iUnion fun x => ↑(Polynomial.map (algebraMap I …
    x : K
    hx : Membership.mem (setOf fun x => And (IsIntegral Int x) (∀ (φ : RingHom K A …
    h_map_ℚ_minpoly : Eq (minpoly Rat x) (Polynomial.map (algebraMap Int Rat) (min …
    i : Nat
    ⊢ Membership.mem (Set.Icc (Neg.neg ↑C) ↑C) ((minpoly Int x).coeff i)
  -/
  rw [mem_Icc, ← abs_le, ← @Int.cast_le ℝ]
  /-
    case refine_2
    K : Type u_1
    inst✝⁴ : Field K
    inst✝³ : NumberField K
    A : Type u_2
    inst✝² : NormedField A
    inst✝¹ : IsAlgClosed A
    inst✝ : NormedAlgebra Rat A
    B : Real
    C : Nat := Nat.ceil (HMul.hMul (HPow.hPow (Max.max B 1) (Module.finrank Rat K) …
    this : (Set.iUnion fun f => Set.iUnion fun x => ↑(Polynomial.map (algebraMap I …
    x : K
    hx : Membership.mem (setOf fun x => And (IsIntegral Int x) (∀ (φ : RingHom K A …
    h_map_ℚ_minpoly : Eq (minpoly Rat x) (Polynomial.map (algebraMap Int Rat) (min …
    i : Nat
    ⊢ LE.le ↑(abs ((minpoly Int x).coeff i)) ↑↑C
  -/
  refine (Eq.trans_le ?_ <| coeff_bdd_of_norm_le hx.2 i).trans (Nat.le_ceil _)
  /-
    case refine_2
    K : Type u_1
    inst✝⁴ : Field K
    inst✝³ : NumberField K
    A : Type u_2
    inst✝² : NormedField A
    inst✝¹ : IsAlgClosed A
    inst✝ : NormedAlgebra Rat A
    B : Real
    C : Nat := Nat.ceil (HMul.hMul (HPow.hPow (Max.max B 1) (Module.finrank Rat K) …
    this : (Set.iUnion fun f => Set.iUnion fun x => ↑(Polynomial.map (algebraMap I …
    x : K
    hx : Membership.mem (setOf fun x => And (IsIntegral Int x) (∀ (φ : RingHom K A …
    h_map_ℚ_minpoly : Eq (minpoly Rat x) (Polynomial.map (algebraMap Int Rat) (min …
    i : Nat
    ⊢ Eq (↑(abs ((minpoly Int x).coeff i))) (Norm.norm ((minpoly Rat x).coeff i))
  -/
  rw [h_map_ℚ_minpoly, coeff_map, eq_intCast, Int.norm_cast_rat, Int.norm_eq_abs, Int.cast_abs]
  /-
    🎉 no goals
  -/


/-- An algebraic integer whose conjugates are all of norm one is a root of unity. -/
theorem pow_eq_one_of_norm_eq_one {x : K} (hxi : IsIntegral ℤ x) (hx : ∀ φ : K →+* A, ‖φ x‖ = 1) :
    ∃ (n : ℕ) (_ : 0 < n), x ^ n = 1 := by
  obtain ⟨a, -, b, -, habne, h⟩ :=
    @Set.Infinite.exists_ne_map_eq_of_mapsTo _ _ _ _ (x ^ · : ℕ → K) Set.infinite_univ
      (by exact fun a _ => ⟨hxi.pow a, fun φ => by simp [hx φ]⟩) (finite_of_norm_le K A (1 : ℝ))
  /-
    case intro.intro.intro.intro.intro
    K : Type u_1
    inst✝⁴ : Field K
    inst✝³ : NumberField K
    A : Type u_2
    inst✝² : NormedField A
    inst✝¹ : IsAlgClosed A
    inst✝ : NormedAlgebra Rat A
    x : K
    hxi : IsIntegral Int x
    hx : ∀ (φ : RingHom K A), Eq (Norm.norm (φ x)) 1
    a b : Nat
    habne : Ne a b
    h : Eq (HPow.hPow x a) (HPow.hPow x b)
    ⊢ Exists fun n => Exists fun x_1 => Eq (HPow.hPow x n) 1
  -/
  wlog hlt : b < a
    /-
      case intro.intro.intro.intro.intro.inr
      K : Type u_1
      inst✝⁴ : Field K
      inst✝³ : NumberField K
      A : Type u_2
      inst✝² : NormedField A
      inst✝¹ : IsAlgClosed A
      inst✝ : NormedAlgebra Rat A
      x : K
      hxi : IsIntegral Int x
      hx : ∀ (φ : RingHom K A), Eq (Norm.norm (φ x)) 1
      a b : Nat
      habne : Ne a b
      h : Eq (HPow.hPow x a) (HPow.hPow x b)
      this : ∀ (K : Type u_1) [inst : Field K] [inst_1 : NumberField K] (A : Type u_ …
      hlt : Not (LT.lt b a)
      ⊢ Exists fun n => Exists fun x_1 => Eq (HPow.hPow x n) 1
    -/
  · exact this K A hxi hx b a habne.symm h.symm (habne.lt_or_lt.resolve_right hlt)
    /-
      🎉 no goals
    -/
  /-
    K✝ : Type u_1
    inst✝⁶ : Field K✝
    A✝ : Type u_2
    inst✝⁵ : NormedField A✝
    K : Type u_1
    inst✝⁴ : Field K
    inst✝³ : NumberField K
    A : Type u_2
    inst✝² : NormedField A
    inst✝¹ : IsAlgClosed A
    inst✝ : NormedAlgebra Rat A
    x : K
    hxi : IsIntegral Int x
    hx : ∀ (φ : RingHom K A), Eq (Norm.norm (φ x)) 1
    a b : Nat
    habne : Ne a b
    h : Eq (HPow.hPow x a) (HPow.hPow x b)
    hlt : LT.lt b a
    ⊢ Exists fun n => Exists fun x_1 => Eq (HPow.hPow x n) 1
  -/
  refine ⟨a - b, tsub_pos_of_lt hlt, ?_⟩
  /-
    K✝ : Type u_1
    inst✝⁶ : Field K✝
    A✝ : Type u_2
    inst✝⁵ : NormedField A✝
    K : Type u_1
    inst✝⁴ : Field K
    inst✝³ : NumberField K
    A : Type u_2
    inst✝² : NormedField A
    inst✝¹ : IsAlgClosed A
    inst✝ : NormedAlgebra Rat A
    x : K
    hxi : IsIntegral Int x
    hx : ∀ (φ : RingHom K A), Eq (Norm.norm (φ x)) 1
    a b : Nat
    habne : Ne a b
    h : Eq (HPow.hPow x a) (HPow.hPow x b)
    hlt : LT.lt b a
    ⊢ Eq (HPow.hPow x (HSub.hSub a b)) 1
  -/
  rw [← Nat.sub_add_cancel hlt.le, pow_add, mul_left_eq_self₀] at h
  /-
    K✝ : Type u_1
    inst✝⁶ : Field K✝
    A✝ : Type u_2
    inst✝⁵ : NormedField A✝
    K : Type u_1
    inst✝⁴ : Field K
    inst✝³ : NumberField K
    A : Type u_2
    inst✝² : NormedField A
    inst✝¹ : IsAlgClosed A
    inst✝ : NormedAlgebra Rat A
    x : K
    hxi : IsIntegral Int x
    hx : ∀ (φ : RingHom K A), Eq (Norm.norm (φ x)) 1
    a b : Nat
    habne : Ne a b
    h : Or (Eq (HPow.hPow x (HSub.hSub a b)) 1) (Eq (HPow.hPow x b) 0)
    hlt : LT.lt b a
    ⊢ Eq (HPow.hPow x (HSub.hSub a b)) 1
  -/
  refine h.resolve_right fun hp => ?_
  /-
    K✝ : Type u_1
    inst✝⁶ : Field K✝
    A✝ : Type u_2
    inst✝⁵ : NormedField A✝
    K : Type u_1
    inst✝⁴ : Field K
    inst✝³ : NumberField K
    A : Type u_2
    inst✝² : NormedField A
    inst✝¹ : IsAlgClosed A
    inst✝ : NormedAlgebra Rat A
    x : K
    hxi : IsIntegral Int x
    hx : ∀ (φ : RingHom K A), Eq (Norm.norm (φ x)) 1
    a b : Nat
    habne : Ne a b
    h : Or (Eq (HPow.hPow x (HSub.hSub a b)) 1) (Eq (HPow.hPow x b) 0)
    hlt : LT.lt b a
    hp : Eq (HPow.hPow x b) 0
    ⊢ False
  -/
  specialize hx (IsAlgClosed.lift (R := ℚ)).toRingHom
  /-
    K✝ : Type u_1
    inst✝⁶ : Field K✝
    A✝ : Type u_2
    inst✝⁵ : NormedField A✝
    K : Type u_1
    inst✝⁴ : Field K
    inst✝³ : NumberField K
    A : Type u_2
    inst✝² : NormedField A
    inst✝¹ : IsAlgClosed A
    inst✝ : NormedAlgebra Rat A
    x : K
    hxi : IsIntegral Int x
    a b : Nat
    habne : Ne a b
    h : Or (Eq (HPow.hPow x (HSub.hSub a b)) 1) (Eq (HPow.hPow x b) 0)
    hlt : LT.lt b a
    hp : Eq (HPow.hPow x b) 0
    hx : Eq (Norm.norm (IsAlgClosed.lift.toRingHom x)) 1
    ⊢ False
  -/
  rw [pow_eq_zero hp, map_zero, norm_zero] at hx; norm_num at hx
                                                  /-
                                                    🎉 no goals
                                                  -/


/-- An embedding into a normed division ring defines a place of `K` -/
def NumberField.place : AbsoluteValue K ℝ :=
  (IsAbsoluteValue.toAbsoluteValue (norm : A → ℝ)).comp φ.injective


@[simp]
theorem NumberField.place_apply (x : K) : (NumberField.place φ) x = norm (φ x) := rfl


/-- The conjugate of a complex embedding as a complex embedding. -/
abbrev conjugate (φ : K →+* ℂ) : K →+* ℂ := star φ


@[simp]
theorem conjugate_coe_eq (φ : K →+* ℂ) (x : K) : (conjugate φ) x = conj (φ x) := rfl


theorem place_conjugate (φ : K →+* ℂ) : place (conjugate φ) = place φ := by
  /-
    K : Type u_1
    inst✝ : Field K
    φ : RingHom K Complex
    ⊢ Eq (NumberField.place (NumberField.ComplexEmbedding.conjugate φ)) (NumberFie …
  -/
  ext; simp only [place_apply, norm_eq_abs, abs_conj, conjugate_coe_eq]
       /-
         🎉 no goals
       -/


/-- An embedding into `ℂ` is real if it is fixed by complex conjugation. -/
abbrev IsReal (φ : K →+* ℂ) : Prop := IsSelfAdjoint φ


theorem isReal_iff {φ : K →+* ℂ} : IsReal φ ↔ conjugate φ = φ := isSelfAdjoint_iff


theorem isReal_conjugate_iff {φ : K →+* ℂ} : IsReal (conjugate φ) ↔ IsReal φ :=
  IsSelfAdjoint.star_iff


/-- A real embedding as a ring homomorphism from `K` to `ℝ` . -/
def IsReal.embedding {φ : K →+* ℂ} (hφ : IsReal φ) : K →+* ℝ where
  toFun x := (φ x).re
                 /-
                   K : Type u_1
                   inst✝¹ : Field K
                   k : Type u_2
                   inst✝ : Field k
                   φ : RingHom K Complex
                   hφ : NumberField.ComplexEmbedding.IsReal φ
                   ⊢ Eq ((fun x => (φ x).re) 1) 1
                 -/
  map_one' := by simp only [map_one, one_re]
                 /-
                   🎉 no goals
                 -/
  map_mul' := by
    simp only [Complex.conj_eq_iff_im.mp (RingHom.congr_fun hφ _), map_mul, mul_re,
      mul_zero, tsub_zero, eq_self_iff_true, forall_const]
                  /-
                    K : Type u_1
                    inst✝¹ : Field K
                    k : Type u_2
                    inst✝ : Field k
                    φ : RingHom K Complex
                    hφ : NumberField.ComplexEmbedding.IsReal φ
                    ⊢ Eq ((↑{ toFun := fun x => (φ x).re, map_one' := ⋯, map_mul' := ⋯ }).toFun 0) 0
                  -/
  map_zero' := by simp only [map_zero, zero_re]
                  /-
                    🎉 no goals
                  -/
                 /-
                   K : Type u_1
                   inst✝¹ : Field K
                   k : Type u_2
                   inst✝ : Field k
                   φ : RingHom K Complex
                   hφ : NumberField.ComplexEmbedding.IsReal φ
                   ⊢ ∀ (x y : K), Eq ((↑{ toFun := fun x => (φ x).re, map_one' := ⋯, map_mul' :=  …
                 -/
  map_add' := by simp only [map_add, add_re, eq_self_iff_true, forall_const]
                 /-
                   🎉 no goals
                 -/


@[simp]
theorem IsReal.coe_embedding_apply {φ : K →+* ℂ} (hφ : IsReal φ) (x : K) :
    (hφ.embedding x : ℂ) = φ x := by
  /-
    K : Type u_1
    inst✝ : Field K
    φ : RingHom K Complex
    hφ : NumberField.ComplexEmbedding.IsReal φ
    x : K
    ⊢ Eq (↑(hφ.embedding x)) (φ x)
  -/
  apply Complex.ext
    /-
      case a
      K : Type u_1
      inst✝ : Field K
      φ : RingHom K Complex
      hφ : NumberField.ComplexEmbedding.IsReal φ
      x : K
      ⊢ Eq (↑(hφ.embedding x)).re (φ x).re
    -/
  · rfl
    /-
      🎉 no goals
    -/
    /-
      case a
      K : Type u_1
      inst✝ : Field K
      φ : RingHom K Complex
      hφ : NumberField.ComplexEmbedding.IsReal φ
      x : K
      ⊢ Eq (↑(hφ.embedding x)).im (φ x).im
    -/
  · rw [ofReal_im, eq_comm, ← Complex.conj_eq_iff_im]
    /-
      case a
      K : Type u_1
      inst✝ : Field K
      φ : RingHom K Complex
      hφ : NumberField.ComplexEmbedding.IsReal φ
      x : K
      ⊢ Eq ((starRingEnd Complex) (φ x)) (φ x)
    -/
    exact RingHom.congr_fun hφ x
    /-
      🎉 no goals
    -/


lemma IsReal.comp (f : k →+* K) {φ : K →+* ℂ} (hφ : IsReal φ) :
                            /-
                              K : Type u_1
                              inst✝¹ : Field K
                              k : Type u_2
                              inst✝ : Field k
                              f : RingHom k K
                              φ : RingHom K Complex
                              hφ : NumberField.ComplexEmbedding.IsReal φ
                              ⊢ NumberField.ComplexEmbedding.IsReal (φ.comp f)
                            -/
    IsReal (φ.comp f) := by ext1 x; simpa using RingHom.congr_fun hφ (f x)
                                    /-
                                      🎉 no goals
                                    -/


lemma isReal_comp_iff {f : k ≃+* K} {φ : K →+* ℂ} :
    IsReal (φ.comp (f : k →+* K)) ↔ IsReal φ :=
              /-
                K : Type u_1
                inst✝¹ : Field K
                k : Type u_2
                inst✝ : Field k
                f : RingEquiv k K
                φ : RingHom K Complex
                H : NumberField.ComplexEmbedding.IsReal (φ.comp ↑f)
                ⊢ NumberField.ComplexEmbedding.IsReal φ
              -/
  ⟨fun H ↦ by convert H.comp f.symm.toRingHom; ext1; simp, IsReal.comp _⟩
                                                     /-
                                                       🎉 no goals
                                                     -/


lemma exists_comp_symm_eq_of_comp_eq [Algebra k K] [IsGalois k K] (φ ψ : K →+* ℂ)
    (h : φ.comp (algebraMap k K) = ψ.comp (algebraMap k K)) :
    ∃ σ : K ≃ₐ[k] K, φ.comp σ.symm = ψ := by
  /-
    K : Type u_1
    inst✝³ : Field K
    k : Type u_2
    inst✝² : Field k
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    φ ψ : RingHom K Complex
    h : Eq (φ.comp (algebraMap k K)) (ψ.comp (algebraMap k K))
    ⊢ Exists fun σ => Eq (φ.comp ↑σ.symm) ψ
  -/
  letI := (φ.comp (algebraMap k K)).toAlgebra
  /-
    K : Type u_1
    inst✝³ : Field K
    k : Type u_2
    inst✝² : Field k
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    φ ψ : RingHom K Complex
    h : Eq (φ.comp (algebraMap k K)) (ψ.comp (algebraMap k K))
    this : Algebra k Complex := (φ.comp (algebraMap k K)).toAlgebra
    ⊢ Exists fun σ => Eq (φ.comp ↑σ.symm) ψ
  -/
  letI := φ.toAlgebra
  /-
    K : Type u_1
    inst✝³ : Field K
    k : Type u_2
    inst✝² : Field k
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    φ ψ : RingHom K Complex
    h : Eq (φ.comp (algebraMap k K)) (ψ.comp (algebraMap k K))
    this✝ : Algebra k Complex := (φ.comp (algebraMap k K)).toAlgebra
    this : Algebra K Complex := φ.toAlgebra
    ⊢ Exists fun σ => Eq (φ.comp ↑σ.symm) ψ
  -/
  have : IsScalarTower k K ℂ := IsScalarTower.of_algebraMap_eq' rfl
  /-
    K : Type u_1
    inst✝³ : Field K
    k : Type u_2
    inst✝² : Field k
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    φ ψ : RingHom K Complex
    h : Eq (φ.comp (algebraMap k K)) (ψ.comp (algebraMap k K))
    this✝¹ : Algebra k Complex := (φ.comp (algebraMap k K)).toAlgebra
    this✝ : Algebra K Complex := φ.toAlgebra
    this : IsScalarTower k K Complex
    ⊢ Exists fun σ => Eq (φ.comp ↑σ.symm) ψ
  -/
  let ψ' : K →ₐ[k] ℂ := { ψ with commutes' := fun r ↦ (RingHom.congr_fun h r).symm }
  /-
    K : Type u_1
    inst✝³ : Field K
    k : Type u_2
    inst✝² : Field k
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    φ ψ : RingHom K Complex
    h : Eq (φ.comp (algebraMap k K)) (ψ.comp (algebraMap k K))
    this✝¹ : Algebra k Complex := (φ.comp (algebraMap k K)).toAlgebra
    this✝ : Algebra K Complex := φ.toAlgebra
    this : IsScalarTower k K Complex
    ψ' : AlgHom k K Complex := { toRingHom := ψ, commutes' := ⋯ }
    ⊢ Exists fun σ => Eq (φ.comp ↑σ.symm) ψ
  -/
  use (AlgHom.restrictNormal' ψ' K).symm
  /-
    case h
    K : Type u_1
    inst✝³ : Field K
    k : Type u_2
    inst✝² : Field k
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    φ ψ : RingHom K Complex
    h : Eq (φ.comp (algebraMap k K)) (ψ.comp (algebraMap k K))
    this✝¹ : Algebra k Complex := (φ.comp (algebraMap k K)).toAlgebra
    this✝ : Algebra K Complex := φ.toAlgebra
    this : IsScalarTower k K Complex
    ψ' : AlgHom k K Complex := { toRingHom := ψ, commutes' := ⋯ }
    ⊢ Eq (φ.comp ↑(ψ'.restrictNormal' K).symm.symm) ψ
  -/
  ext1 x
  /-
    case h.a
    K : Type u_1
    inst✝³ : Field K
    k : Type u_2
    inst✝² : Field k
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    φ ψ : RingHom K Complex
    h : Eq (φ.comp (algebraMap k K)) (ψ.comp (algebraMap k K))
    this✝¹ : Algebra k Complex := (φ.comp (algebraMap k K)).toAlgebra
    this✝ : Algebra K Complex := φ.toAlgebra
    this : IsScalarTower k K Complex
    ψ' : AlgHom k K Complex := { toRingHom := ψ, commutes' := ⋯ }
    x : K
    ⊢ Eq ((φ.comp ↑(ψ'.restrictNormal' K).symm.symm) x) (ψ x)
  -/
  exact AlgHom.restrictNormal_commutes ψ' K x
  /-
    🎉 no goals
  -/


/--
`IsConj φ σ` states that `σ : K ≃ₐ[k] K` is the conjugation under the embedding `φ : K →+* ℂ`.
-/
def IsConj : Prop := conjugate φ = φ.comp σ


lemma IsConj.eq (h : IsConj φ σ) (x) : φ (σ x) = star (φ x) := RingHom.congr_fun h.symm x


lemma IsConj.ext {σ₁ σ₂ : K ≃ₐ[k] K} (h₁ : IsConj φ σ₁) (h₂ : IsConj φ σ₂) : σ₁ = σ₂ :=
  AlgEquiv.ext fun x ↦ φ.injective ((h₁.eq x).trans (h₂.eq x).symm)


lemma IsConj.ext_iff {σ₁ σ₂ : K ≃ₐ[k] K} (h₁ : IsConj φ σ₁) : σ₁ = σ₂ ↔ IsConj φ σ₂ :=
  ⟨fun e ↦ e ▸ h₁, h₁.ext⟩


lemma IsConj.isReal_comp (h : IsConj φ σ) : IsReal (φ.comp (algebraMap k K)) := by
  /-
    K : Type u_1
    inst✝² : Field K
    k : Type u_2
    inst✝¹ : Field k
    inst✝ : Algebra k K
    φ : RingHom K Complex
    σ : AlgEquiv k K K
    h : NumberField.ComplexEmbedding.IsConj φ σ
    ⊢ NumberField.ComplexEmbedding.IsReal (φ.comp (algebraMap k K))
  -/
  ext1 x
  simp only [conjugate_coe_eq, RingHom.coe_comp, Function.comp_apply, ← h.eq,
    starRingEnd_apply, AlgEquiv.commutes]


lemma isConj_one_iff : IsConj φ (1 : K ≃ₐ[k] K) ↔ IsReal φ := Iff.rfl


alias ⟨_, IsReal.isConjGal_one⟩ := ComplexEmbedding.isConj_one_iff


lemma IsConj.symm (hσ : IsConj φ σ) :
                                              /-
                                                K : Type u_1
                                                inst✝² : Field K
                                                k : Type u_2
                                                inst✝¹ : Field k
                                                inst✝ : Algebra k K
                                                φ : RingHom K Complex
                                                σ : AlgEquiv k K K
                                                hσ : NumberField.ComplexEmbedding.IsConj φ σ
                                                x : K
                                                ⊢ Eq ((NumberField.ComplexEmbedding.conjugate φ) x) ((φ.comp ↑σ.symm) x)
                                              -/
    IsConj φ σ.symm := RingHom.ext fun x ↦ by simpa using congr_arg star (hσ.eq (σ.symm x))
                                              /-
                                                🎉 no goals
                                              -/


lemma isConj_symm : IsConj φ σ.symm ↔ IsConj φ σ :=
  ⟨IsConj.symm, IsConj.symm⟩


/-- An infinite place of a number field `K` is a place associated to a complex embedding. -/
def NumberField.InfinitePlace := { w : AbsoluteValue K ℝ // ∃ φ : K →+* ℂ, place φ = w }


instance [NumberField K] : Nonempty (NumberField.InfinitePlace K) := Set.instNonemptyRange _


/-- Return the infinite place defined by a complex embedding `φ`. -/
noncomputable def NumberField.InfinitePlace.mk (φ : K →+* ℂ) : NumberField.InfinitePlace K :=
  ⟨place φ, ⟨φ, rfl⟩⟩


instance {K : Type*} [Field K] : FunLike (InfinitePlace K) K ℝ where
  coe w x := w.1 x
  coe_injective' _ _ h := Subtype.eq (AbsoluteValue.ext fun x => congr_fun h x)


lemma coe_apply {K : Type*} [Field K] (v : InfinitePlace K) (x : K) :
  v x = v.1 x := rfl


@[ext]
lemma ext {K : Type*} [Field K] (v₁ v₂ : InfinitePlace K) (h : ∀ k, v₁ k = v₂ k) : v₁ = v₂ :=
  Subtype.ext <| AbsoluteValue.ext h


instance : MonoidWithZeroHomClass (InfinitePlace K) K ℝ where
  map_mul w _ _ := w.1.map_mul _ _
  map_one w := w.1.map_one
  map_zero w := w.1.map_zero


instance : NonnegHomClass (InfinitePlace K) K ℝ where
  apply_nonneg w _ := w.1.nonneg _


@[simp]
theorem apply (φ : K →+* ℂ) (x : K) : (mk φ) x = Complex.abs (φ x) := rfl


/-- For an infinite place `w`, return an embedding `φ` such that `w = infinite_place φ` . -/
noncomputable def embedding (w : InfinitePlace K) : K →+* ℂ := w.2.choose


@[simp]
theorem mk_embedding (w : InfinitePlace K) : mk (embedding w) = w := Subtype.ext w.2.choose_spec


@[simp]
theorem mk_conjugate_eq (φ : K →+* ℂ) : mk (ComplexEmbedding.conjugate φ) = mk φ := by
  /-
    K : Type u_2
    inst✝ : Field K
    φ : RingHom K Complex
    ⊢ Eq (NumberField.InfinitePlace.mk (NumberField.ComplexEmbedding.conjugate φ)) …
  -/
  refine DFunLike.ext _ _ (fun x => ?_)
  /-
    K : Type u_2
    inst✝ : Field K
    φ : RingHom K Complex
    x : K
    ⊢ Eq ((NumberField.InfinitePlace.mk (NumberField.ComplexEmbedding.conjugate φ) …
  -/
  rw [apply, apply, ComplexEmbedding.conjugate_coe_eq, Complex.abs_conj]
  /-
    🎉 no goals
  -/


theorem norm_embedding_eq (w : InfinitePlace K) (x : K) :
    ‖(embedding w) x‖ = w x := by
  /-
    K : Type u_2
    inst✝ : Field K
    w : NumberField.InfinitePlace K
    x : K
    ⊢ Eq (Norm.norm (w.embedding x)) (w x)
  -/
  nth_rewrite 2 [← mk_embedding w]
  /-
    K : Type u_2
    inst✝ : Field K
    w : NumberField.InfinitePlace K
    x : K
    ⊢ Eq (Norm.norm (w.embedding x)) ((NumberField.InfinitePlace.mk w.embedding) x)
  -/
  rfl
  /-
    🎉 no goals
  -/


theorem eq_iff_eq (x : K) (r : ℝ) : (∀ w : InfinitePlace K, w x = r) ↔ ∀ φ : K →+* ℂ, ‖φ x‖ = r :=
                             /-
                               K : Type u_2
                               inst✝ : Field K
                               x : K
                               r : Real
                               ⊢ (∀ (φ : RingHom K Complex), Eq (Norm.norm (φ x)) r) → ∀ (w : NumberField.Inf …
                             -/
  ⟨fun hw φ => hw (mk φ), by rintro hφ ⟨w, ⟨φ, rfl⟩⟩; exact hφ φ⟩
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem le_iff_le (x : K) (r : ℝ) : (∀ w : InfinitePlace K, w x ≤ r) ↔ ∀ φ : K →+* ℂ, ‖φ x‖ ≤ r :=
                             /-
                               K : Type u_2
                               inst✝ : Field K
                               x : K
                               r : Real
                               ⊢ (∀ (φ : RingHom K Complex), LE.le (Norm.norm (φ x)) r) → ∀ (w : NumberField. …
                             -/
  ⟨fun hw φ => hw (mk φ), by rintro hφ ⟨w, ⟨φ, rfl⟩⟩; exact hφ φ⟩
                                                      /-
                                                        🎉 no goals
                                                      -/


theorem pos_iff {w : InfinitePlace K} {x : K} : 0 < w x ↔ x ≠ 0 := AbsoluteValue.pos_iff w.1


@[simp]
theorem mk_eq_iff {φ ψ : K →+* ℂ} : mk φ = mk ψ ↔ φ = ψ ∨ ComplexEmbedding.conjugate φ = ψ := by
  /-
    K : Type u_2
    inst✝ : Field K
    φ ψ : RingHom K Complex
    ⊢ Iff (Eq (NumberField.InfinitePlace.mk φ) (NumberField.InfinitePlace.mk ψ)) ( …
  -/
  constructor
  · -- We prove that the map ψ ∘ φ⁻¹ between φ(K) and ℂ is uniform continuous, thus it is either the
    -- inclusion or the complex conjugation using `Complex.uniformContinuous_ringHom_eq_id_or_conj`
    /-
      case mp
      K : Type u_2
      inst✝ : Field K
      φ ψ : RingHom K Complex
      ⊢ Eq (NumberField.InfinitePlace.mk φ) (NumberField.InfinitePlace.mk ψ) → Or (E …
    -/
    intro h₀
    /-
      case mp
      K : Type u_2
      inst✝ : Field K
      φ ψ : RingHom K Complex
      h₀ : Eq (NumberField.InfinitePlace.mk φ) (NumberField.InfinitePlace.mk ψ)
      ⊢ Or (Eq φ ψ) (Eq (NumberField.ComplexEmbedding.conjugate φ) ψ)
    -/
    obtain ⟨j, hiφ⟩ := (φ.injective).hasLeftInverse
    /-
      case mp.intro
      K : Type u_2
      inst✝ : Field K
      φ ψ : RingHom K Complex
      h₀ : Eq (NumberField.InfinitePlace.mk φ) (NumberField.InfinitePlace.mk ψ)
      j : Complex → K
      hiφ : Function.LeftInverse j ⇑φ
      ⊢ Or (Eq φ ψ) (Eq (NumberField.ComplexEmbedding.conjugate φ) ψ)
    -/
    let ι := RingEquiv.ofLeftInverse hiφ
    have hlip : LipschitzWith 1 (RingHom.comp ψ ι.symm.toRingHom) := by
      change LipschitzWith 1 (ψ ∘ ι.symm)
      apply LipschitzWith.of_dist_le_mul
      intro x y
      rw [NNReal.coe_one, one_mul, NormedField.dist_eq, Function.comp_apply, Function.comp_apply,
        ← map_sub, ← map_sub]
      apply le_of_eq
      suffices ‖φ (ι.symm (x - y))‖ = ‖ψ (ι.symm (x - y))‖ by
        rw [← this, ← RingEquiv.ofLeftInverse_apply hiφ _, RingEquiv.apply_symm_apply ι _]
        rfl
      exact congrFun (congrArg (↑) h₀) _
    cases
      Complex.uniformContinuous_ringHom_eq_id_or_conj φ.fieldRange hlip.uniformContinuous with
    | inl h =>
        left; ext1 x
        conv_rhs => rw [← hiφ x]
        exact (congrFun h (ι x)).symm
    | inr h =>
        right; ext1 x
        conv_rhs => rw [← hiφ x]
        exact (congrFun h (ι x)).symm
    /-
      case mpr
      K : Type u_2
      inst✝ : Field K
      φ ψ : RingHom K Complex
      ⊢ Or (Eq φ ψ) (Eq (NumberField.ComplexEmbedding.conjugate φ) ψ) → Eq (NumberFi …
    -/
  · rintro (⟨h⟩ | ⟨h⟩)
      /-
        case mpr.inl
        K : Type u_2
        inst✝ : Field K
        φ ψ : RingHom K Complex
        h : Eq φ ψ
        ⊢ Eq (NumberField.InfinitePlace.mk φ) (NumberField.InfinitePlace.mk ψ)
      -/
    · exact congr_arg mk h
      /-
        🎉 no goals
      -/
      /-
        case mpr.inr
        K : Type u_2
        inst✝ : Field K
        φ ψ : RingHom K Complex
        h : Eq (NumberField.ComplexEmbedding.conjugate φ) ψ
        ⊢ Eq (NumberField.InfinitePlace.mk φ) (NumberField.InfinitePlace.mk ψ)
      -/
    · rw [← mk_conjugate_eq]
      /-
        case mpr.inr
        K : Type u_2
        inst✝ : Field K
        φ ψ : RingHom K Complex
        h : Eq (NumberField.ComplexEmbedding.conjugate φ) ψ
        ⊢ Eq (NumberField.InfinitePlace.mk (NumberField.ComplexEmbedding.conjugate φ)) …
      -/
      exact congr_arg mk h
      /-
        🎉 no goals
      -/


/-- An infinite place is real if it is defined by a real embedding. -/
def IsReal (w : InfinitePlace K) : Prop := ∃ φ : K →+* ℂ, ComplexEmbedding.IsReal φ ∧ mk φ = w


/-- An infinite place is complex if it is defined by a complex (ie. not real) embedding. -/
def IsComplex (w : InfinitePlace K) : Prop := ∃ φ : K →+* ℂ, ¬ComplexEmbedding.IsReal φ ∧ mk φ = w


theorem embedding_mk_eq (φ : K →+* ℂ) :
    embedding (mk φ) = φ ∨ embedding (mk φ) = ComplexEmbedding.conjugate φ := by
  /-
    K : Type u_2
    inst✝ : Field K
    φ : RingHom K Complex
    ⊢ Or (Eq (NumberField.InfinitePlace.mk φ).embedding φ) (Eq (NumberField.Infini …
  -/
  rw [@eq_comm _ _ φ, @eq_comm _ _ (ComplexEmbedding.conjugate φ), ← mk_eq_iff, mk_embedding]
  /-
    🎉 no goals
  -/


@[simp]
theorem embedding_mk_eq_of_isReal {φ : K →+* ℂ} (h : ComplexEmbedding.IsReal φ) :
    embedding (mk φ) = φ := by
  /-
    K : Type u_2
    inst✝ : Field K
    φ : RingHom K Complex
    h : NumberField.ComplexEmbedding.IsReal φ
    ⊢ Eq (NumberField.InfinitePlace.mk φ).embedding φ
  -/
  have := embedding_mk_eq φ
  /-
    K : Type u_2
    inst✝ : Field K
    φ : RingHom K Complex
    h : NumberField.ComplexEmbedding.IsReal φ
    this : Or (Eq (NumberField.InfinitePlace.mk φ).embedding φ) (Eq (NumberField.I …
    ⊢ Eq (NumberField.InfinitePlace.mk φ).embedding φ
  -/
  rwa [ComplexEmbedding.isReal_iff.mp h, or_self] at this
  /-
    🎉 no goals
  -/


theorem isReal_iff {w : InfinitePlace K} :
    IsReal w ↔ ComplexEmbedding.IsReal (embedding w) := by
  /-
    K : Type u_2
    inst✝ : Field K
    w : NumberField.InfinitePlace K
    ⊢ Iff w.IsReal (NumberField.ComplexEmbedding.IsReal w.embedding)
  -/
  refine ⟨?_, fun h => ⟨embedding w, h, mk_embedding w⟩⟩
  /-
    K : Type u_2
    inst✝ : Field K
    w : NumberField.InfinitePlace K
    ⊢ w.IsReal → NumberField.ComplexEmbedding.IsReal w.embedding
  -/
  rintro ⟨φ, ⟨hφ, rfl⟩⟩
  /-
    case intro.intro
    K : Type u_2
    inst✝ : Field K
    φ : RingHom K Complex
    hφ : NumberField.ComplexEmbedding.IsReal φ
    ⊢ NumberField.ComplexEmbedding.IsReal (NumberField.InfinitePlace.mk φ).embedding
  -/
  rwa [embedding_mk_eq_of_isReal hφ]
  /-
    🎉 no goals
  -/


theorem isComplex_iff {w : InfinitePlace K} :
    IsComplex w ↔ ¬ComplexEmbedding.IsReal (embedding w) := by
  /-
    K : Type u_2
    inst✝ : Field K
    w : NumberField.InfinitePlace K
    ⊢ Iff w.IsComplex (Not (NumberField.ComplexEmbedding.IsReal w.embedding))
  -/
  refine ⟨?_, fun h => ⟨embedding w, h, mk_embedding w⟩⟩
  /-
    K : Type u_2
    inst✝ : Field K
    w : NumberField.InfinitePlace K
    ⊢ w.IsComplex → Not (NumberField.ComplexEmbedding.IsReal w.embedding)
  -/
  rintro ⟨φ, ⟨hφ, rfl⟩⟩
  /-
    case intro.intro
    K : Type u_2
    inst✝ : Field K
    φ : RingHom K Complex
    hφ : Not (NumberField.ComplexEmbedding.IsReal φ)
    ⊢ Not (NumberField.ComplexEmbedding.IsReal (NumberField.InfinitePlace.mk φ).em …
  -/
  contrapose! hφ
  cases mk_eq_iff.mp (mk_embedding (mk φ)) with
  | inl h => rwa [h] at hφ
  | inr h => rwa [← ComplexEmbedding.isReal_conjugate_iff, h] at hφ


@[simp]
theorem conjugate_embedding_eq_of_isReal {w : InfinitePlace K} (h : IsReal w) :
    ComplexEmbedding.conjugate (embedding w) = embedding w :=
  ComplexEmbedding.isReal_iff.mpr (isReal_iff.mp h)


@[simp]
theorem not_isReal_iff_isComplex {w : InfinitePlace K} : ¬IsReal w ↔ IsComplex w := by
  /-
    K : Type u_2
    inst✝ : Field K
    w : NumberField.InfinitePlace K
    ⊢ Iff (Not w.IsReal) w.IsComplex
  -/
  rw [isComplex_iff, isReal_iff]
  /-
    🎉 no goals
  -/


@[simp]
theorem not_isComplex_iff_isReal {w : InfinitePlace K} : ¬IsComplex w ↔ IsReal w := by
  /-
    K : Type u_2
    inst✝ : Field K
    w : NumberField.InfinitePlace K
    ⊢ Iff (Not w.IsComplex) w.IsReal
  -/
  rw [isComplex_iff, isReal_iff, not_not]
  /-
    🎉 no goals
  -/


theorem isReal_or_isComplex (w : InfinitePlace K) : IsReal w ∨ IsComplex w := by
  /-
    K : Type u_2
    inst✝ : Field K
    w : NumberField.InfinitePlace K
    ⊢ Or w.IsReal w.IsComplex
  -/
  rw [← not_isReal_iff_isComplex]; exact em _
                                   /-
                                     🎉 no goals
                                   -/


theorem ne_of_isReal_isComplex {w w' : InfinitePlace K} (h : IsReal w) (h' : IsComplex w') :
    w ≠ w' := fun h_eq ↦ not_isReal_iff_isComplex.mpr h' (h_eq ▸ h)


variable (K) in
theorem disjoint_isReal_isComplex :
    Disjoint {(w : InfinitePlace K) | IsReal w} {(w : InfinitePlace K) | IsComplex w} :=
  Set.disjoint_iff.2 <| fun _ hw ↦ not_isReal_iff_isComplex.2 hw.2 hw.1


/-- The real embedding associated to a real infinite place. -/
noncomputable def embedding_of_isReal {w : InfinitePlace K} (hw : IsReal w) : K →+* ℝ :=
  ComplexEmbedding.IsReal.embedding (isReal_iff.mp hw)


@[simp]
theorem embedding_of_isReal_apply {w : InfinitePlace K} (hw : IsReal w) (x : K) :
    ((embedding_of_isReal hw) x : ℂ) = (embedding w) x :=
  ComplexEmbedding.IsReal.coe_embedding_apply (isReal_iff.mp hw) x


theorem norm_embedding_of_isReal {w : InfinitePlace K} (hw : IsReal w) (x : K) :
    ‖embedding_of_isReal hw x‖ = w x := by
  /-
    K : Type u_2
    inst✝ : Field K
    w : NumberField.InfinitePlace K
    hw : w.IsReal
    x : K
    ⊢ Eq (Norm.norm ((NumberField.InfinitePlace.embedding_of_isReal hw) x)) (w x)
  -/
  rw [← norm_embedding_eq, ← embedding_of_isReal_apply hw, Complex.norm_real]
  /-
    🎉 no goals
  -/


@[simp]
theorem isReal_of_mk_isReal {φ : K →+* ℂ} (h : IsReal (mk φ)) :
    ComplexEmbedding.IsReal φ := by
  /-
    K : Type u_2
    inst✝ : Field K
    φ : RingHom K Complex
    h : (NumberField.InfinitePlace.mk φ).IsReal
    ⊢ NumberField.ComplexEmbedding.IsReal φ
  -/
  contrapose! h
  /-
    K : Type u_2
    inst✝ : Field K
    φ : RingHom K Complex
    h : Not (NumberField.ComplexEmbedding.IsReal φ)
    ⊢ Not (NumberField.InfinitePlace.mk φ).IsReal
  -/
  rw [not_isReal_iff_isComplex]
  /-
    K : Type u_2
    inst✝ : Field K
    φ : RingHom K Complex
    h : Not (NumberField.ComplexEmbedding.IsReal φ)
    ⊢ (NumberField.InfinitePlace.mk φ).IsComplex
  -/
  exact ⟨φ, h, rfl⟩
  /-
    🎉 no goals
  -/


lemma isReal_mk_iff {φ : K →+* ℂ} :
    IsReal (mk φ) ↔ ComplexEmbedding.IsReal φ :=
  ⟨isReal_of_mk_isReal, fun H ↦ ⟨_, H, rfl⟩⟩


lemma isComplex_mk_iff {φ : K →+* ℂ} :
    IsComplex (mk φ) ↔ ¬ ComplexEmbedding.IsReal φ :=
  not_isReal_iff_isComplex.symm.trans isReal_mk_iff.not


@[simp]
theorem not_isReal_of_mk_isComplex {φ : K →+* ℂ} (h : IsComplex (mk φ)) :
                                      /-
                                        K : Type u_2
                                        inst✝ : Field K
                                        φ : RingHom K Complex
                                        h : (NumberField.InfinitePlace.mk φ).IsComplex
                                        ⊢ Not (NumberField.ComplexEmbedding.IsReal φ)
                                      -/
    ¬ ComplexEmbedding.IsReal φ := by rwa [← isComplex_mk_iff]
                                      /-
                                        🎉 no goals
                                      -/


/-- The multiplicity of an infinite place, that is the number of distinct complex embeddings that
define it, see `card_filter_mk_eq`. -/
noncomputable def mult (w : InfinitePlace K) : ℕ := if (IsReal w) then 1 else 2


theorem mult_pos {w : InfinitePlace K} : 0 < mult w := by
  /-
    K : Type u_2
    inst✝ : Field K
    w : NumberField.InfinitePlace K
    ⊢ LT.lt 0 w.mult
  -/
  rw [mult]
  /-
    K : Type u_2
    inst✝ : Field K
    w : NumberField.InfinitePlace K
    ⊢ LT.lt 0 (ite w.IsReal 1 2)
  -/
                /-
                  🎉 no goals
                -/
  split_ifs <;> norm_num
                /-
                  🎉 no goals
                -/


@[simp]
theorem mult_ne_zero {w : InfinitePlace K} : mult w ≠ 0 := ne_of_gt mult_pos


theorem one_le_mult {w : InfinitePlace K} : (1 : ℝ) ≤ mult w := by
  /-
    K : Type u_2
    inst✝ : Field K
    w : NumberField.InfinitePlace K
    ⊢ LE.le 1 ↑w.mult
  -/
  rw [← Nat.cast_one, Nat.cast_le]
  /-
    K : Type u_2
    inst✝ : Field K
    w : NumberField.InfinitePlace K
    ⊢ LE.le 1 w.mult
  -/
  exact mult_pos
  /-
    🎉 no goals
  -/


theorem card_filter_mk_eq [NumberField K] (w : InfinitePlace K) : #{φ | mk φ = w} = mult w := by
  conv_lhs =>
    congr; congr; ext
    rw [← mk_embedding w, mk_eq_iff, ComplexEmbedding.conjugate, star_involutive.eq_iff]
  simp_rw [Finset.filter_or, Finset.filter_eq' _ (embedding w),
    Finset.filter_eq' _ (ComplexEmbedding.conjugate (embedding w)),
    Finset.mem_univ, ite_true, mult]
  /-
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : NumberField K
    w : NumberField.InfinitePlace K
    ⊢ Eq (Union.union (Singleton.singleton w.embedding) (Singleton.singleton (Numb …
  -/
  split_ifs with hw
  · rw [ComplexEmbedding.isReal_iff.mp (isReal_iff.mp hw), Finset.union_idempotent,
      Finset.card_singleton]
    /-
      case neg
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : NumberField K
      w : NumberField.InfinitePlace K
      hw : Not w.IsReal
      ⊢ Eq (Union.union (Singleton.singleton w.embedding) (Singleton.singleton (Numb …
    -/
  · refine Finset.card_pair ?_
    /-
      case neg
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : NumberField K
      w : NumberField.InfinitePlace K
      hw : Not w.IsReal
      ⊢ Ne w.embedding (NumberField.ComplexEmbedding.conjugate w.embedding)
    -/
    rwa [Ne, eq_comm, ← ComplexEmbedding.isReal_iff, ← isReal_iff]
    /-
      🎉 no goals
    -/


noncomputable instance NumberField.InfinitePlace.fintype [NumberField K] :
    Fintype (InfinitePlace K) := Set.fintypeRange _


theorem sum_mult_eq [NumberField K] :
    ∑ w : InfinitePlace K, mult w = Module.finrank ℚ K := by
  rw [← Embeddings.card K ℂ, Fintype.card, Finset.card_eq_sum_ones, ← Finset.univ.sum_fiberwise
    (fun φ => InfinitePlace.mk φ)]
  exact Finset.sum_congr rfl
    (fun _ _ => by rw [Finset.sum_const, smul_eq_mul, mul_one, card_filter_mk_eq])


/-- The map from real embeddings to real infinite places as an equiv -/
noncomputable def mkReal :
    { φ : K →+* ℂ // ComplexEmbedding.IsReal φ } ≃ { w : InfinitePlace K // IsReal w } := by
  /-
    k : Type u_1
    inst✝² : Field k
    K : Type u_2
    inst✝¹ : Field K
    F : Type u_3
    inst✝ : Field F
    ⊢ Equiv (Subtype fun φ => NumberField.ComplexEmbedding.IsReal φ) (Subtype fun  …
  -/
  refine (Equiv.ofBijective (fun φ => ⟨mk φ, ?_⟩) ⟨fun φ ψ h => ?_, fun w => ?_⟩)
    /-
      case refine_1
      k : Type u_1
      inst✝² : Field k
      K : Type u_2
      inst✝¹ : Field K
      F : Type u_3
      inst✝ : Field F
      φ : Subtype fun φ => NumberField.ComplexEmbedding.IsReal φ
      ⊢ (NumberField.InfinitePlace.mk ↑φ).IsReal
    -/
  · exact ⟨φ, φ.prop, rfl⟩
    /-
      🎉 no goals
    -/
  · rwa [Subtype.mk.injEq, mk_eq_iff, ComplexEmbedding.isReal_iff.mp φ.prop, or_self,
      ← Subtype.ext_iff] at h
    /-
      case refine_3
      k : Type u_1
      inst✝² : Field k
      K : Type u_2
      inst✝¹ : Field K
      F : Type u_3
      inst✝ : Field F
      w : Subtype fun w => w.IsReal
      ⊢ Exists fun a => Eq ((fun φ => ⟨NumberField.InfinitePlace.mk ↑φ, ⋯⟩) a) w
    -/
  · exact ⟨⟨embedding w, isReal_iff.mp w.prop⟩, by simp⟩
    /-
      🎉 no goals
    -/


/-- The map from nonreal embeddings to complex infinite places -/
noncomputable def mkComplex :
    { φ : K →+* ℂ // ¬ComplexEmbedding.IsReal φ } → { w : InfinitePlace K // IsComplex w } :=
  Subtype.map mk fun φ hφ => ⟨φ, hφ, rfl⟩


@[simp]
theorem mkReal_coe (φ : { φ : K →+* ℂ // ComplexEmbedding.IsReal φ }) :
    (mkReal φ : InfinitePlace K) = mk (φ : K →+* ℂ) := rfl


@[simp]
theorem mkComplex_coe (φ : { φ : K →+* ℂ // ¬ComplexEmbedding.IsReal φ }) :
    (mkComplex φ : InfinitePlace K) = mk (φ : K →+* ℂ) := rfl


/-- The infinite part of the product formula : for `x ∈ K`, we have `Π_w ‖x‖_w = |norm(x)|` where
`‖·‖_w` is the normalized absolute value for `w`. -/
theorem prod_eq_abs_norm (x : K) :
    ∏ w : InfinitePlace K, w x ^ mult w = abs (Algebra.norm ℚ x) := by
  /-
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : K
    ⊢ Eq (Finset.univ.prod fun w => HPow.hPow (w x) w.mult) ↑(abs ((Algebra.norm R …
  -/
  convert (congr_arg Complex.abs (@Algebra.norm_eq_prod_embeddings ℚ _ _ _ _ ℂ _ _ _ _ _ x)).symm
  · rw [map_prod, ← Fintype.prod_equiv RingHom.equivRatAlgHom (fun f => Complex.abs (f x))
      (fun φ => Complex.abs (φ x)) fun _ => by simp [RingHom.equivRatAlgHom_apply]; rfl]
    /-
      case h.e'_2
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : K
      ⊢ Eq (Finset.univ.prod fun w => HPow.hPow (w x) w.mult) (Finset.univ.prod fun  …
    -/
    rw [← Finset.prod_fiberwise Finset.univ mk (fun φ => Complex.abs (φ x))]
    have (w : InfinitePlace K) (φ) (hφ : φ ∈ ({φ | mk φ = w} : Finset _)) :
        Complex.abs (φ x) = w x := by rw [← (Finset.mem_filter.mp hφ).2, apply]
    /-
      case h.e'_2
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : K
      this : ∀ (w : NumberField.InfinitePlace K) (φ : RingHom K Complex), Membership …
      ⊢ Eq (Finset.univ.prod fun w => HPow.hPow (w x) w.mult) (Finset.univ.prod fun  …
    -/
    simp_rw [Finset.prod_congr rfl (this _), Finset.prod_const, card_filter_mk_eq]
    /-
      🎉 no goals
    -/
    /-
      case h.e'_3
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : K
      ⊢ Eq (↑(abs ((Algebra.norm Rat) x))) (Complex.abs ((algebraMap Rat Complex) (( …
    -/
  · rw [eq_ratCast, Rat.cast_abs, ← Complex.abs_ofReal, Complex.ofReal_ratCast]
    /-
      🎉 no goals
    -/


theorem one_le_of_lt_one {w : InfinitePlace K} {a : (𝓞 K)} (ha : a ≠ 0)
    (h : ∀ ⦃z⦄, z ≠ w → z a < 1) : 1 ≤ w a := by
  suffices (1 : ℝ) ≤ |Algebra.norm ℚ (a : K)| by
    contrapose! this
    rw [← InfinitePlace.prod_eq_abs_norm, ← Finset.prod_const_one]
    refine Finset.prod_lt_prod_of_nonempty (fun _ _ ↦ ?_) (fun z _ ↦ ?_) Finset.univ_nonempty
    · exact pow_pos (pos_iff.mpr ((Subalgebra.coe_eq_zero _).not.mpr ha)) _
    · refine pow_lt_one₀ (apply_nonneg _ _) ?_ (by rw [mult]; split_ifs <;> norm_num)
      by_cases hz : z = w
      · rwa [hz]
      · exact h hz
  /-
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : NumberField K
    w : NumberField.InfinitePlace K
    a : NumberField.RingOfIntegers K
    ha : Ne a 0
    h : ∀ ⦃z : NumberField.InfinitePlace K⦄, Ne z w → LT.lt (z ↑a) 1
    ⊢ LE.le 1 ↑(abs ((Algebra.norm Rat) ↑a))
  -/
  rw [← Algebra.coe_norm_int, ← Int.cast_one, ← Int.cast_abs, Rat.cast_intCast, Int.cast_le]
  /-
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : NumberField K
    w : NumberField.InfinitePlace K
    a : NumberField.RingOfIntegers K
    ha : Ne a 0
    h : ∀ ⦃z : NumberField.InfinitePlace K⦄, Ne z w → LT.lt (z ↑a) 1
    ⊢ LE.le 1 (abs ((Algebra.norm Int) a))
  -/
  exact Int.one_le_abs (Algebra.norm_ne_zero_iff.mpr ha)
  /-
    🎉 no goals
  -/


open scoped IntermediateField in
theorem _root_.NumberField.is_primitive_element_of_infinitePlace_lt {x : 𝓞 K}
    {w : InfinitePlace K} (h₁ : x ≠ 0) (h₂ : ∀ ⦃w'⦄, w' ≠ w → w' x < 1)
    (h₃ : IsReal w ∨ |(w.embedding x).re| < 1) : ℚ⟮(x : K)⟯ = ⊤ := by
  /-
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.RingOfIntegers K
    w : NumberField.InfinitePlace K
    h₁ : Ne x 0
    h₂ : ∀ ⦃w' : NumberField.InfinitePlace K⦄, Ne w' w → LT.lt (w' ↑x) 1
    h₃ : Or w.IsReal (LT.lt (abs (w.embedding ↑x).re) 1)
    ⊢ Eq (IntermediateField.adjoin Rat (Singleton.singleton ↑x)) Top.top
  -/
  rw [Field.primitive_element_iff_algHom_eq_of_eval ℚ ℂ ?_ _ w.embedding.toRatAlgHom]
    /-
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : NumberField.RingOfIntegers K
      w : NumberField.InfinitePlace K
      h₁ : Ne x 0
      h₂ : ∀ ⦃w' : NumberField.InfinitePlace K⦄, Ne w' w → LT.lt (w' ↑x) 1
      h₃ : Or w.IsReal (LT.lt (abs (w.embedding ↑x).re) 1)
      ⊢ ∀ (ψ : AlgHom Rat K Complex), Eq (w.embedding.toRatAlgHom ↑x) (ψ ↑x) → Eq w. …
    -/
  · intro ψ hψ
    /-
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : NumberField.RingOfIntegers K
      w : NumberField.InfinitePlace K
      h₁ : Ne x 0
      h₂ : ∀ ⦃w' : NumberField.InfinitePlace K⦄, Ne w' w → LT.lt (w' ↑x) 1
      h₃ : Or w.IsReal (LT.lt (abs (w.embedding ↑x).re) 1)
      ψ : AlgHom Rat K Complex
      hψ : Eq (w.embedding.toRatAlgHom ↑x) (ψ ↑x)
      ⊢ Eq w.embedding.toRatAlgHom ψ
    -/
    have h : 1 ≤ w x := one_le_of_lt_one h₁ h₂
    have main : w = InfinitePlace.mk ψ.toRingHom := by
      erw [← norm_embedding_eq, hψ] at h
      contrapose! h
      exact h₂ h.symm
    /-
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : NumberField.RingOfIntegers K
      w : NumberField.InfinitePlace K
      h₁ : Ne x 0
      h₂ : ∀ ⦃w' : NumberField.InfinitePlace K⦄, Ne w' w → LT.lt (w' ↑x) 1
      h₃ : Or w.IsReal (LT.lt (abs (w.embedding ↑x).re) 1)
      ψ : AlgHom Rat K Complex
      hψ : Eq (w.embedding.toRatAlgHom ↑x) (ψ ↑x)
      h : LE.le 1 (w ↑x)
      main : Eq w (NumberField.InfinitePlace.mk ψ.toRingHom)
      ⊢ Eq w.embedding.toRatAlgHom ψ
    -/
    rw [(mk_embedding w).symm, mk_eq_iff] at main
    cases h₃ with
    | inl hw =>
      rw [conjugate_embedding_eq_of_isReal hw, or_self] at main
      exact congr_arg RingHom.toRatAlgHom main
    | inr hw =>
      refine congr_arg RingHom.toRatAlgHom (main.resolve_right fun h' ↦ hw.not_le ?_)
      have : (embedding w x).im = 0 := by
        erw [← Complex.conj_eq_iff_im, RingHom.congr_fun h' x]
        exact hψ.symm
      rwa [← norm_embedding_eq, ← Complex.re_add_im (embedding w x), this, Complex.ofReal_zero,
        zero_mul, add_zero, Complex.norm_eq_abs, Complex.abs_ofReal] at h
    /-
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : NumberField K
      x : NumberField.RingOfIntegers K
      w : NumberField.InfinitePlace K
      h₁ : Ne x 0
      h₂ : ∀ ⦃w' : NumberField.InfinitePlace K⦄, Ne w' w → LT.lt (w' ↑x) 1
      h₃ : Or w.IsReal (LT.lt (abs (w.embedding ↑x).re) 1)
      ⊢ ∀ (x : K), Polynomial.Splits (algebraMap Rat Complex) (minpoly Rat x)
    -/
  · exact fun x ↦ IsAlgClosed.splits_codomain (minpoly ℚ x)
    /-
      🎉 no goals
    -/


theorem _root_.NumberField.adjoin_eq_top_of_infinitePlace_lt {x : 𝓞 K} {w : InfinitePlace K}
    (h₁ : x ≠ 0) (h₂ : ∀ ⦃w'⦄, w' ≠ w → w' x < 1) (h₃ : IsReal w ∨ |(w.embedding x).re| < 1) :
    Algebra.adjoin ℚ {(x : K)} = ⊤ := by
  /-
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : NumberField K
    x : NumberField.RingOfIntegers K
    w : NumberField.InfinitePlace K
    h₁ : Ne x 0
    h₂ : ∀ ⦃w' : NumberField.InfinitePlace K⦄, Ne w' w → LT.lt (w' ↑x) 1
    h₃ : Or w.IsReal (LT.lt (abs (w.embedding ↑x).re) 1)
    ⊢ Eq (Algebra.adjoin Rat (Singleton.singleton ↑x)) Top.top
  -/
  rw [← IntermediateField.adjoin_simple_toSubalgebra_of_integral (IsIntegral.of_finite ℚ _)]
  exact congr_arg IntermediateField.toSubalgebra <|
    NumberField.is_primitive_element_of_infinitePlace_lt h₁ h₂ h₃


/-- The number of infinite real places of the number field `K`. -/
noncomputable abbrev nrRealPlaces := card { w : InfinitePlace K // IsReal w }


@[deprecated (since := "2024-10-24")] alias NrRealPlaces := nrRealPlaces


/-- The number of infinite complex places of the number field `K`. -/
noncomputable abbrev nrComplexPlaces := card { w : InfinitePlace K // IsComplex w }


@[deprecated (since := "2024-10-24")] alias NrComplexPlaces := nrComplexPlaces


theorem card_real_embeddings :
    card { φ : K →+* ℂ // ComplexEmbedding.IsReal φ } = nrRealPlaces K := Fintype.card_congr mkReal


theorem card_eq_nrRealPlaces_add_nrComplexPlaces :
    Fintype.card (InfinitePlace K) = nrRealPlaces K + nrComplexPlaces K := by
  convert Fintype.card_subtype_or_disjoint (IsReal (K := K)) (IsComplex (K := K))
    (disjoint_isReal_isComplex K) using 1
  /-
    case h.e'_2
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ Eq (Fintype.card (NumberField.InfinitePlace K)) (Fintype.card (Subtype fun x …
  -/
  exact (Fintype.card_of_subtype _ (fun w ↦ ⟨fun _ ↦ isReal_or_isComplex w, fun _ ↦ by simp⟩)).symm
  /-
    🎉 no goals
  -/


theorem card_complex_embeddings :
    card { φ : K →+* ℂ // ¬ComplexEmbedding.IsReal φ } = 2 * nrComplexPlaces K := by
  suffices ∀ w : { w : InfinitePlace K // IsComplex w },
     #{φ : {φ //¬ ComplexEmbedding.IsReal φ} | mkComplex φ = w} = 2 by
    rw [Fintype.card, Finset.card_eq_sum_ones, ← Finset.sum_fiberwise _ (fun φ => mkComplex φ)]
    simp_rw [Finset.sum_const, this, smul_eq_mul, mul_one, Fintype.card, Finset.card_eq_sum_ones,
      Finset.mul_sum, Finset.sum_const, smul_eq_mul, mul_one]
  /-
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ ∀ (w : Subtype fun w => w.IsComplex), Eq (Finset.filter (fun φ => Eq (Number …
  -/
  rintro ⟨w, hw⟩
  /-
    case mk
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : NumberField K
    w : NumberField.InfinitePlace K
    hw : w.IsComplex
    ⊢ Eq (Finset.filter (fun φ => Eq (NumberField.InfinitePlace.mkComplex φ) ⟨w, h …
  -/
  convert card_filter_mk_eq w
    /-
      case h.e'_2
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : NumberField K
      w : NumberField.InfinitePlace K
      hw : w.IsComplex
      ⊢ Eq (Finset.filter (fun φ => Eq (NumberField.InfinitePlace.mkComplex φ) ⟨w, h …
    -/
  · rw [← Fintype.card_subtype, ← Fintype.card_subtype]
    /-
      case h.e'_2
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : NumberField K
      w : NumberField.InfinitePlace K
      hw : w.IsComplex
      ⊢ Eq (Fintype.card (Subtype fun x => Eq (NumberField.InfinitePlace.mkComplex x …
    -/
    refine Fintype.card_congr (Equiv.ofBijective ?_ ⟨fun _ _ h => ?_, fun ⟨φ, hφ⟩ => ?_⟩)
      /-
        case h.e'_2.refine_1
        K : Type u_2
        inst✝¹ : Field K
        inst✝ : NumberField K
        w : NumberField.InfinitePlace K
        hw : w.IsComplex
        ⊢ (Subtype fun x => Eq (NumberField.InfinitePlace.mkComplex x) ⟨w, hw⟩) → Subt …
      -/
    · exact fun ⟨φ, hφ⟩ => ⟨φ.val, by rwa [Subtype.ext_iff] at hφ⟩
      /-
        🎉 no goals
      -/
      /-
        case h.e'_2.refine_2
        K : Type u_2
        inst✝¹ : Field K
        inst✝ : NumberField K
        w : NumberField.InfinitePlace K
        hw : w.IsComplex
        x✝¹ x✝ : Subtype fun x => Eq (NumberField.InfinitePlace.mkComplex x) ⟨w, hw⟩
        h : Eq (NumberField.InfinitePlace.card_complex_embeddings.match_2 K w hw (fun  …
        ⊢ Eq x✝¹ x✝
      -/
    · rwa [Subtype.mk_eq_mk, ← Subtype.ext_iff, ← Subtype.ext_iff] at h
      /-
        🎉 no goals
      -/
      /-
        case h.e'_2.refine_3
        K : Type u_2
        inst✝¹ : Field K
        inst✝ : NumberField K
        w : NumberField.InfinitePlace K
        hw : w.IsComplex
        x✝ : Subtype fun x => Eq (NumberField.InfinitePlace.mk x) w
        φ : RingHom K Complex
        hφ : Eq (NumberField.InfinitePlace.mk φ) w
        ⊢ Exists fun a => Eq (NumberField.InfinitePlace.card_complex_embeddings.match_ …
      -/
    · refine ⟨⟨⟨φ, not_isReal_of_mk_isComplex (hφ.symm ▸ hw)⟩, ?_⟩, rfl⟩
      /-
        case h.e'_2.refine_3
        K : Type u_2
        inst✝¹ : Field K
        inst✝ : NumberField K
        w : NumberField.InfinitePlace K
        hw : w.IsComplex
        x✝ : Subtype fun x => Eq (NumberField.InfinitePlace.mk x) w
        φ : RingHom K Complex
        hφ : Eq (NumberField.InfinitePlace.mk φ) w
        ⊢ Eq (NumberField.InfinitePlace.mkComplex ⟨φ, ⋯⟩) ⟨w, hw⟩
      -/
      rwa [Subtype.ext_iff, mkComplex_coe]
      /-
        🎉 no goals
      -/
    /-
      case h.e'_3
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : NumberField K
      w : NumberField.InfinitePlace K
      hw : w.IsComplex
      ⊢ Eq 2 w.mult
    -/
  · simp_rw [mult, not_isReal_iff_isComplex.mpr hw, ite_false]
    /-
      🎉 no goals
    -/


theorem card_add_two_mul_card_eq_rank :
    nrRealPlaces K + 2 * nrComplexPlaces K = finrank ℚ K := by
  rw [← card_real_embeddings, ← card_complex_embeddings, Fintype.card_subtype_compl,
    ← Embeddings.card K ℂ, Nat.add_sub_of_le]
  /-
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : NumberField K
    ⊢ LE.le (Fintype.card (Subtype fun φ => NumberField.ComplexEmbedding.IsReal φ) …
  -/
  exact Fintype.card_subtype_le _
  /-
    🎉 no goals
  -/


theorem nrComplexPlaces_eq_zero_of_finrank_eq_one (h : finrank ℚ K = 1) :
                                /-
                                  K : Type u_2
                                  inst✝¹ : Field K
                                  inst✝ : NumberField K
                                  h : Eq (Module.finrank Rat K) 1
                                  ⊢ Eq (NumberField.InfinitePlace.nrComplexPlaces K) 0
                                -/
    nrComplexPlaces K = 0 := by linarith [card_add_two_mul_card_eq_rank K]
                                /-
                                  🎉 no goals
                                -/


theorem nrRealPlaces_eq_one_of_finrank_eq_one (h : finrank ℚ K = 1) :
    nrRealPlaces K = 1 := by
  /-
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : NumberField K
    h : Eq (Module.finrank Rat K) 1
    ⊢ Eq (NumberField.InfinitePlace.nrRealPlaces K) 1
  -/
  have := card_add_two_mul_card_eq_rank K
  /-
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : NumberField K
    h : Eq (Module.finrank Rat K) 1
    this : Eq (HAdd.hAdd (NumberField.InfinitePlace.nrRealPlaces K) (HMul.hMul 2 ( …
    ⊢ Eq (NumberField.InfinitePlace.nrRealPlaces K) 1
  -/
  rwa [nrComplexPlaces_eq_zero_of_finrank_eq_one h, h, mul_zero, add_zero] at this
  /-
    🎉 no goals
  -/


/-- The restriction of an infinite place along an embedding. -/
def comap (w : InfinitePlace K) (f : k →+* K) : InfinitePlace k :=
  ⟨w.1.comp f.injective, w.embedding.comp f,
       /-
         k : Type u_1
         inst✝³ : Field k
         K : Type u_2
         inst✝² : Field K
         F : Type u_3
         inst✝¹ : Field F
         inst✝ : NumberField K
         w : NumberField.InfinitePlace K
         f : RingHom k K
         ⊢ Eq (NumberField.place (w.embedding.comp f)) ((↑w).comp ⋯)
       -/
    by { ext x; show _ = w.1 (f x); rw [← w.2.choose_spec]; rfl }⟩
       /-
         🎉 no goals
       -/


@[simp]
lemma comap_mk (φ : K →+* ℂ) (f : k →+* K) : (mk φ).comap f = mk (φ.comp f) := rfl


lemma comap_id (w : InfinitePlace K) : w.comap (RingHom.id K) = w := rfl


lemma comap_comp (w : InfinitePlace K) (f : F →+* K) (g : k →+* F) :
    w.comap (f.comp g) = (w.comap f).comap g := rfl


lemma IsReal.comap (f : k →+* K) {w : InfinitePlace K} (hφ : IsReal w) :
    IsReal (w.comap f) := by
  /-
    k : Type u_1
    inst✝¹ : Field k
    K : Type u_2
    inst✝ : Field K
    f : RingHom k K
    w : NumberField.InfinitePlace K
    hφ : w.IsReal
    ⊢ (w.comap f).IsReal
  -/
  rw [← mk_embedding w, comap_mk, isReal_mk_iff]
  /-
    k : Type u_1
    inst✝¹ : Field k
    K : Type u_2
    inst✝ : Field K
    f : RingHom k K
    w : NumberField.InfinitePlace K
    hφ : w.IsReal
    ⊢ NumberField.ComplexEmbedding.IsReal (w.embedding.comp f)
  -/
  rw [← mk_embedding w, isReal_mk_iff] at hφ
  /-
    k : Type u_1
    inst✝¹ : Field k
    K : Type u_2
    inst✝ : Field K
    f : RingHom k K
    w : NumberField.InfinitePlace K
    hφ : NumberField.ComplexEmbedding.IsReal w.embedding
    ⊢ NumberField.ComplexEmbedding.IsReal (w.embedding.comp f)
  -/
  exact hφ.comp f
  /-
    🎉 no goals
  -/


lemma isReal_comap_iff (f : k ≃+* K) {w : InfinitePlace K} :
    IsReal (w.comap (f : k →+* K)) ↔ IsReal w := by
  /-
    k : Type u_1
    inst✝¹ : Field k
    K : Type u_2
    inst✝ : Field K
    f : RingEquiv k K
    w : NumberField.InfinitePlace K
    ⊢ Iff (w.comap ↑f).IsReal w.IsReal
  -/
  rw [← mk_embedding w, comap_mk, isReal_mk_iff, isReal_mk_iff, ComplexEmbedding.isReal_comp_iff]
  /-
    🎉 no goals
  -/


lemma comap_surjective [Algebra k K] [Algebra.IsAlgebraic k K] :
    Function.Surjective (comap · (algebraMap k K)) := fun w ↦
  letI := w.embedding.toAlgebra
  ⟨mk (IsAlgClosed.lift (M := ℂ) (R := k)).toRingHom,
       /-
         k : Type u_1
         inst✝³ : Field k
         K : Type u_2
         inst✝² : Field K
         inst✝¹ : Algebra k K
         inst✝ : Algebra.IsAlgebraic k K
         w : NumberField.InfinitePlace k
         this : Algebra k Complex := w.embedding.toAlgebra
         ⊢ Eq ((fun x => x.comap (algebraMap k K)) (NumberField.InfinitePlace.mk IsAlgC …
       -/
    by simp [this, comap_mk, RingHom.algebraMap_toAlgebra]⟩
       /-
         🎉 no goals
       -/


lemma mult_comap_le (f : k →+* K) (w : InfinitePlace K) : mult (w.comap f) ≤ mult w := by
  /-
    k : Type u_1
    inst✝¹ : Field k
    K : Type u_2
    inst✝ : Field K
    f : RingHom k K
    w : NumberField.InfinitePlace K
    ⊢ LE.le (w.comap f).mult w.mult
  -/
  rw [mult, mult]
  /-
    k : Type u_1
    inst✝¹ : Field k
    K : Type u_2
    inst✝ : Field K
    f : RingHom k K
    w : NumberField.InfinitePlace K
    ⊢ LE.le (ite (w.comap f).IsReal 1 2) (ite w.IsReal 1 2)
  -/
  split_ifs with h₁ h₂ h₂
  /-
    case pos
    k : Type u_1
    inst✝¹ : Field k
    K : Type u_2
    inst✝ : Field K
    f : RingHom k K
    w : NumberField.InfinitePlace K
    h₁ : (w.comap f).IsReal
    h₂ : w.IsReal
    ⊢ LE.le 1 1
  -/
  pick_goal 3
    /-
      case pos
      k : Type u_1
      inst✝¹ : Field k
      K : Type u_2
      inst✝ : Field K
      f : RingHom k K
      w : NumberField.InfinitePlace K
      h₁ : Not (w.comap f).IsReal
      h₂ : w.IsReal
      ⊢ LE.le 2 1
    -/
  · exact (h₁ (h₂.comap _)).elim
    /-
      🎉 no goals
    -/
  /-
    case pos
    k : Type u_1
    inst✝¹ : Field k
    K : Type u_2
    inst✝ : Field K
    f : RingHom k K
    w : NumberField.InfinitePlace K
    h₁ : (w.comap f).IsReal
    h₂ : w.IsReal
    ⊢ LE.le 1 1
  -/
  all_goals decide
  /-
    🎉 no goals
  -/


lemma card_mono [NumberField k] [NumberField K] :
    card (InfinitePlace k) ≤ card (InfinitePlace K) :=
  have := Module.Finite.of_restrictScalars_finite ℚ k K
  Fintype.card_le_of_surjective _ comap_surjective


/-- The action of the galois group on infinite places. -/
@[simps! smul_coe_apply]
instance : MulAction (K ≃ₐ[k] K) (InfinitePlace K) where
  smul := fun σ w ↦ w.comap σ.symm
  one_smul := fun _ ↦ rfl
  mul_smul := fun _ _ _ ↦ rfl


lemma smul_eq_comap : σ • w = w.comap σ.symm := rfl


@[simp] lemma smul_apply (x) : (σ • w) x = w (σ.symm x) := rfl


@[simp] lemma smul_mk (φ : K →+* ℂ) : σ • mk φ = mk (φ.comp σ.symm) := rfl


lemma comap_smul {f : F →+* K} : (σ • w).comap f = w.comap (RingHom.comp σ.symm f) := rfl


lemma isReal_smul_iff : IsReal (σ • w) ↔ IsReal w := isReal_comap_iff (f := σ.symm.toRingEquiv)


lemma isComplex_smul_iff : IsComplex (σ • w) ↔ IsComplex w := by
  /-
    k : Type u_1
    inst✝² : Field k
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : Algebra k K
    σ : AlgEquiv k K K
    w : NumberField.InfinitePlace K
    ⊢ Iff (HSMul.hSMul σ w).IsComplex w.IsComplex
  -/
  rw [← not_isReal_iff_isComplex, ← not_isReal_iff_isComplex, isReal_smul_iff]
  /-
    🎉 no goals
  -/


lemma ComplexEmbedding.exists_comp_symm_eq_of_comp_eq [IsGalois k K] (φ ψ : K →+* ℂ)
    (h : φ.comp (algebraMap k K) = ψ.comp (algebraMap k K)) :
    ∃ σ : K ≃ₐ[k] K, φ.comp σ.symm = ψ := by
  /-
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    φ ψ : RingHom K Complex
    h : Eq (φ.comp (algebraMap k K)) (ψ.comp (algebraMap k K))
    ⊢ Exists fun σ => Eq (φ.comp ↑σ.symm) ψ
  -/
  letI := (φ.comp (algebraMap k K)).toAlgebra
  /-
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    φ ψ : RingHom K Complex
    h : Eq (φ.comp (algebraMap k K)) (ψ.comp (algebraMap k K))
    this : Algebra k Complex := (φ.comp (algebraMap k K)).toAlgebra
    ⊢ Exists fun σ => Eq (φ.comp ↑σ.symm) ψ
  -/
  letI := φ.toAlgebra
  /-
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    φ ψ : RingHom K Complex
    h : Eq (φ.comp (algebraMap k K)) (ψ.comp (algebraMap k K))
    this✝ : Algebra k Complex := (φ.comp (algebraMap k K)).toAlgebra
    this : Algebra K Complex := φ.toAlgebra
    ⊢ Exists fun σ => Eq (φ.comp ↑σ.symm) ψ
  -/
  have : IsScalarTower k K ℂ := IsScalarTower.of_algebraMap_eq' rfl
  /-
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    φ ψ : RingHom K Complex
    h : Eq (φ.comp (algebraMap k K)) (ψ.comp (algebraMap k K))
    this✝¹ : Algebra k Complex := (φ.comp (algebraMap k K)).toAlgebra
    this✝ : Algebra K Complex := φ.toAlgebra
    this : IsScalarTower k K Complex
    ⊢ Exists fun σ => Eq (φ.comp ↑σ.symm) ψ
  -/
  let ψ' : K →ₐ[k] ℂ := { ψ with commutes' := fun r ↦ (RingHom.congr_fun h r).symm }
  /-
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    φ ψ : RingHom K Complex
    h : Eq (φ.comp (algebraMap k K)) (ψ.comp (algebraMap k K))
    this✝¹ : Algebra k Complex := (φ.comp (algebraMap k K)).toAlgebra
    this✝ : Algebra K Complex := φ.toAlgebra
    this : IsScalarTower k K Complex
    ψ' : AlgHom k K Complex := { toRingHom := ψ, commutes' := ⋯ }
    ⊢ Exists fun σ => Eq (φ.comp ↑σ.symm) ψ
  -/
  use (AlgHom.restrictNormal' ψ' K).symm
  /-
    case h
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    φ ψ : RingHom K Complex
    h : Eq (φ.comp (algebraMap k K)) (ψ.comp (algebraMap k K))
    this✝¹ : Algebra k Complex := (φ.comp (algebraMap k K)).toAlgebra
    this✝ : Algebra K Complex := φ.toAlgebra
    this : IsScalarTower k K Complex
    ψ' : AlgHom k K Complex := { toRingHom := ψ, commutes' := ⋯ }
    ⊢ Eq (φ.comp ↑(ψ'.restrictNormal' K).symm.symm) ψ
  -/
  ext1 x
  /-
    case h.a
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    φ ψ : RingHom K Complex
    h : Eq (φ.comp (algebraMap k K)) (ψ.comp (algebraMap k K))
    this✝¹ : Algebra k Complex := (φ.comp (algebraMap k K)).toAlgebra
    this✝ : Algebra K Complex := φ.toAlgebra
    this : IsScalarTower k K Complex
    ψ' : AlgHom k K Complex := { toRingHom := ψ, commutes' := ⋯ }
    x : K
    ⊢ Eq ((φ.comp ↑(ψ'.restrictNormal' K).symm.symm) x) (ψ x)
  -/
  exact AlgHom.restrictNormal_commutes ψ' K x
  /-
    🎉 no goals
  -/


lemma exists_smul_eq_of_comap_eq [IsGalois k K] {w w' : InfinitePlace K}
    (h : w.comap (algebraMap k K) = w'.comap (algebraMap k K)) : ∃ σ : K ≃ₐ[k] K, σ • w = w' := by
  /-
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    w w' : NumberField.InfinitePlace K
    h : Eq (w.comap (algebraMap k K)) (w'.comap (algebraMap k K))
    ⊢ Exists fun σ => Eq (HSMul.hSMul σ w) w'
  -/
  rw [← mk_embedding w, ← mk_embedding w', comap_mk, comap_mk, mk_eq_iff] at h
  cases h with
  | inl h =>
    obtain ⟨σ, hσ⟩ := ComplexEmbedding.exists_comp_symm_eq_of_comp_eq w.embedding w'.embedding h
    use σ
    rw [← mk_embedding w, ← mk_embedding w', smul_mk, hσ]
  | inr h =>
    obtain ⟨σ, hσ⟩ := ComplexEmbedding.exists_comp_symm_eq_of_comp_eq
      ((starRingEnd ℂ).comp (embedding w)) w'.embedding h
    use σ
    rw [← mk_embedding w, ← mk_embedding w', smul_mk, mk_eq_iff]
    exact Or.inr hσ


lemma mem_orbit_iff [IsGalois k K] {w w' : InfinitePlace K} :
    w' ∈ MulAction.orbit (K ≃ₐ[k] K) w ↔ w.comap (algebraMap k K) = w'.comap (algebraMap k K) := by
  /-
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    w w' : NumberField.InfinitePlace K
    ⊢ Iff (Membership.mem (MulAction.orbit (AlgEquiv k K K) w) w') (Eq (w.comap (a …
  -/
  refine ⟨?_, exists_smul_eq_of_comap_eq⟩
  /-
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    w w' : NumberField.InfinitePlace K
    ⊢ Membership.mem (MulAction.orbit (AlgEquiv k K K) w) w' → Eq (w.comap (algebr …
  -/
  rintro ⟨σ, rfl : σ • w = w'⟩
  /-
    case intro
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    w : NumberField.InfinitePlace K
    σ : AlgEquiv k K K
    ⊢ Eq (w.comap (algebraMap k K)) ((HSMul.hSMul σ w).comap (algebraMap k K))
  -/
  rw [← mk_embedding w, comap_mk, smul_mk, comap_mk]
  /-
    case intro
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    w : NumberField.InfinitePlace K
    σ : AlgEquiv k K K
    ⊢ Eq (NumberField.InfinitePlace.mk (w.embedding.comp (algebraMap k K))) (Numbe …
  -/
  congr 1; ext1; simp
                 /-
                   🎉 no goals
                 -/


/-- The orbits of infinite places under the action of the galois group are indexed by
the infinite places of the base field. -/
noncomputable
def orbitRelEquiv [IsGalois k K] :
    Quotient (MulAction.orbitRel (K ≃ₐ[k] K) (InfinitePlace K)) ≃ InfinitePlace k := by
  refine Equiv.ofBijective (Quotient.lift (comap · (algebraMap k K))
    fun _ _ e ↦ (mem_orbit_iff.mp e).symm) ⟨?_, ?_⟩
    /-
      case refine_1
      k : Type u_1
      inst✝⁴ : Field k
      K : Type u_2
      inst✝³ : Field K
      F : Type u_3
      inst✝² : Field F
      inst✝¹ : Algebra k K
      σ : AlgEquiv k K K
      w : NumberField.InfinitePlace K
      inst✝ : IsGalois k K
      ⊢ Function.Injective (Quotient.lift (fun x => x.comap (algebraMap k K)) ⋯)
    -/
  · rintro ⟨w⟩ ⟨w'⟩ e
    /-
      case refine_1.mk.mk
      k : Type u_1
      inst✝⁴ : Field k
      K : Type u_2
      inst✝³ : Field K
      F : Type u_3
      inst✝² : Field F
      inst✝¹ : Algebra k K
      σ : AlgEquiv k K K
      w✝ : NumberField.InfinitePlace K
      inst✝ : IsGalois k K
      a₁✝ : Quotient (MulAction.orbitRel (AlgEquiv k K K) (NumberField.InfinitePlace …
      w : NumberField.InfinitePlace K
      a₂✝ : Quotient (MulAction.orbitRel (AlgEquiv k K K) (NumberField.InfinitePlace …
      w' : NumberField.InfinitePlace K
      e : Eq (Quotient.lift (fun x => x.comap (algebraMap k K)) ⋯ (Quot.mk (⇑(MulAct …
      ⊢ Eq (Quot.mk (⇑(MulAction.orbitRel (AlgEquiv k K K) (NumberField.InfinitePlac …
    -/
    exact Quotient.sound (mem_orbit_iff.mpr e.symm)
    /-
      🎉 no goals
    -/
    /-
      case refine_2
      k : Type u_1
      inst✝⁴ : Field k
      K : Type u_2
      inst✝³ : Field K
      F : Type u_3
      inst✝² : Field F
      inst✝¹ : Algebra k K
      σ : AlgEquiv k K K
      w : NumberField.InfinitePlace K
      inst✝ : IsGalois k K
      ⊢ Function.Surjective (Quotient.lift (fun x => x.comap (algebraMap k K)) ⋯)
    -/
  · intro w
    /-
      case refine_2
      k : Type u_1
      inst✝⁴ : Field k
      K : Type u_2
      inst✝³ : Field K
      F : Type u_3
      inst✝² : Field F
      inst✝¹ : Algebra k K
      σ : AlgEquiv k K K
      w✝ : NumberField.InfinitePlace K
      inst✝ : IsGalois k K
      w : NumberField.InfinitePlace k
      ⊢ Exists fun a => Eq (Quotient.lift (fun x => x.comap (algebraMap k K)) ⋯ a) w
    -/
    obtain ⟨w', hw⟩ := comap_surjective (K := K) w
    /-
      case refine_2.intro
      k : Type u_1
      inst✝⁴ : Field k
      K : Type u_2
      inst✝³ : Field K
      F : Type u_3
      inst✝² : Field F
      inst✝¹ : Algebra k K
      σ : AlgEquiv k K K
      w✝ : NumberField.InfinitePlace K
      inst✝ : IsGalois k K
      w : NumberField.InfinitePlace k
      w' : NumberField.InfinitePlace K
      hw : Eq ((fun x => x.comap (algebraMap k K)) w') w
      ⊢ Exists fun a => Eq (Quotient.lift (fun x => x.comap (algebraMap k K)) ⋯ a) w
    -/
    exact ⟨⟦w'⟧, hw⟩
    /-
      🎉 no goals
    -/


lemma orbitRelEquiv_apply_mk'' [IsGalois k K] (w : InfinitePlace K) :
    orbitRelEquiv (Quotient.mk'' w) = comap w (algebraMap k K) := rfl


/--
An infinite place is unramified in a field extension if the restriction has the same multiplicity.
-/
def IsUnramified : Prop := mult (w.comap (algebraMap k K)) = mult w


lemma isUnramified_self : IsUnramified K w := rfl


lemma IsUnramified.eq (h : IsUnramified k w) : mult (w.comap (algebraMap k K)) = mult w := h


lemma isUnramified_iff_mult_le :
    IsUnramified k w ↔ mult w ≤ mult (w.comap (algebraMap k K)) := by
  /-
    k : Type u_1
    inst✝² : Field k
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : Algebra k K
    w : NumberField.InfinitePlace K
    ⊢ Iff (NumberField.InfinitePlace.IsUnramified k w) (LE.le w.mult (w.comap (alg …
  -/
  rw [IsUnramified, le_antisymm_iff, and_iff_right]
  /-
    k : Type u_1
    inst✝² : Field k
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : Algebra k K
    w : NumberField.InfinitePlace K
    ⊢ LE.le (w.comap (algebraMap k K)).mult w.mult
  -/
  exact mult_comap_le _ _
  /-
    🎉 no goals
  -/


lemma IsUnramified.comap_algHom {w : InfinitePlace F} (h : IsUnramified k w) (f : K →ₐ[k] F) :
    IsUnramified k (w.comap (f : K →+* F)) := by
  /-
    k : Type u_1
    inst✝⁴ : Field k
    K : Type u_2
    inst✝³ : Field K
    F : Type u_3
    inst✝² : Field F
    inst✝¹ : Algebra k K
    inst✝ : Algebra k F
    w : NumberField.InfinitePlace F
    h : NumberField.InfinitePlace.IsUnramified k w
    f : AlgHom k K F
    ⊢ NumberField.InfinitePlace.IsUnramified k (w.comap ↑f)
  -/
  rw [InfinitePlace.isUnramified_iff_mult_le, ← InfinitePlace.comap_comp, f.comp_algebraMap, h.eq]
  /-
    k : Type u_1
    inst✝⁴ : Field k
    K : Type u_2
    inst✝³ : Field K
    F : Type u_3
    inst✝² : Field F
    inst✝¹ : Algebra k K
    inst✝ : Algebra k F
    w : NumberField.InfinitePlace F
    h : NumberField.InfinitePlace.IsUnramified k w
    f : AlgHom k K F
    ⊢ LE.le (w.comap ↑f).mult w.mult
  -/
  exact InfinitePlace.mult_comap_le _ _
  /-
    🎉 no goals
  -/


lemma IsUnramified.of_restrictScalars {w : InfinitePlace F} (h : IsUnramified k w) :
    IsUnramified K w := by
  rw [InfinitePlace.isUnramified_iff_mult_le, ← h.eq, IsScalarTower.algebraMap_eq k K F,
    InfinitePlace.comap_comp]
  /-
    k : Type u_1
    inst✝⁶ : Field k
    K : Type u_2
    inst✝⁵ : Field K
    F : Type u_3
    inst✝⁴ : Field F
    inst✝³ : Algebra k K
    inst✝² : Algebra k F
    inst✝¹ : Algebra K F
    inst✝ : IsScalarTower k K F
    w : NumberField.InfinitePlace F
    h : NumberField.InfinitePlace.IsUnramified k w
    ⊢ LE.le ((w.comap (algebraMap K F)).comap (algebraMap k K)).mult (w.comap (alg …
  -/
  exact InfinitePlace.mult_comap_le _ _
  /-
    🎉 no goals
  -/


lemma IsUnramified.comap {w : InfinitePlace F} (h : IsUnramified k w) :
    IsUnramified k (w.comap (algebraMap K F)) :=
  h.comap_algHom (IsScalarTower.toAlgHom k K F)


lemma not_isUnramified_iff :
    ¬ IsUnramified k w ↔ IsComplex w ∧ IsReal (w.comap (algebraMap k K)) := by
  /-
    k : Type u_1
    inst✝² : Field k
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : Algebra k K
    w : NumberField.InfinitePlace K
    ⊢ Iff (Not (NumberField.InfinitePlace.IsUnramified k w)) (And w.IsComplex (w.c …
  -/
  rw [IsUnramified, mult, mult, ← not_isReal_iff_isComplex]
  /-
    k : Type u_1
    inst✝² : Field k
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : Algebra k K
    w : NumberField.InfinitePlace K
    ⊢ Iff (Not (Eq (ite (w.comap (algebraMap k K)).IsReal 1 2) (ite w.IsReal 1 2)) …
  -/
  split_ifs with h₁ h₂ h₂ <;>
    simp only [not_true_eq_false, false_iff, and_self, forall_true_left, IsEmpty.forall_iff,
      not_and, OfNat.one_ne_ofNat, not_false_eq_true, true_iff, OfNat.ofNat_ne_one, h₁, h₂]
  /-
    case pos
    k : Type u_1
    inst✝² : Field k
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : Algebra k K
    w : NumberField.InfinitePlace K
    h₁ : Not (w.comap (algebraMap k K)).IsReal
    h₂ : w.IsReal
    ⊢ False
  -/
  exact h₁ (h₂.comap _)
  /-
    🎉 no goals
  -/


lemma isUnramified_iff :
    IsUnramified k w ↔ IsReal w ∨ IsComplex (w.comap (algebraMap k K)) := by
  rw [← not_iff_not, not_isUnramified_iff, not_or,
    not_isReal_iff_isComplex, not_isComplex_iff_isReal]


lemma IsReal.isUnramified (h : IsReal w) : IsUnramified k w := isUnramified_iff.mpr (Or.inl h)


lemma _root_.NumberField.ComplexEmbedding.IsConj.isUnramified_mk_iff
    {φ : K →+* ℂ} (h : ComplexEmbedding.IsConj φ σ) :
    IsUnramified k (mk φ) ↔ σ = 1 := by
  rw [h.ext_iff, ComplexEmbedding.isConj_one_iff, ← not_iff_not, not_isUnramified_iff,
    ← not_isReal_iff_isComplex, comap_mk, isReal_mk_iff, isReal_mk_iff, eq_true h.isReal_comp,
    and_true]


lemma isUnramified_mk_iff_forall_isConj [IsGalois k K] {φ : K →+* ℂ} :
    IsUnramified k (mk φ) ↔ ∀ σ : K ≃ₐ[k] K, ComplexEmbedding.IsConj φ σ → σ = 1 := by
  refine ⟨fun H σ hσ ↦ hσ.isUnramified_mk_iff.mp H,
    fun H ↦ ?_⟩
  /-
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    φ : RingHom K Complex
    H : ∀ (σ : AlgEquiv k K K), NumberField.ComplexEmbedding.IsConj φ σ → Eq σ 1
    ⊢ NumberField.InfinitePlace.IsUnramified k (NumberField.InfinitePlace.mk φ)
  -/
  by_contra hφ
  /-
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    φ : RingHom K Complex
    H : ∀ (σ : AlgEquiv k K K), NumberField.ComplexEmbedding.IsConj φ σ → Eq σ 1
    hφ : Not (NumberField.InfinitePlace.IsUnramified k (NumberField.InfinitePlace. …
    ⊢ False
  -/
  rw [not_isUnramified_iff] at hφ
  rw [comap_mk, isReal_mk_iff, ← not_isReal_iff_isComplex, isReal_mk_iff,
    ← ComplexEmbedding.isConj_one_iff (k := k)] at hφ
  /-
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    φ : RingHom K Complex
    H : ∀ (σ : AlgEquiv k K K), NumberField.ComplexEmbedding.IsConj φ σ → Eq σ 1
    hφ : And (Not (NumberField.ComplexEmbedding.IsConj φ 1)) (NumberField.ComplexE …
    ⊢ False
  -/
  letI := (φ.comp (algebraMap k K)).toAlgebra
  /-
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    φ : RingHom K Complex
    H : ∀ (σ : AlgEquiv k K K), NumberField.ComplexEmbedding.IsConj φ σ → Eq σ 1
    hφ : And (Not (NumberField.ComplexEmbedding.IsConj φ 1)) (NumberField.ComplexE …
    this : Algebra k Complex := (φ.comp (algebraMap k K)).toAlgebra
    ⊢ False
  -/
  letI := φ.toAlgebra
  /-
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    φ : RingHom K Complex
    H : ∀ (σ : AlgEquiv k K K), NumberField.ComplexEmbedding.IsConj φ σ → Eq σ 1
    hφ : And (Not (NumberField.ComplexEmbedding.IsConj φ 1)) (NumberField.ComplexE …
    this✝ : Algebra k Complex := (φ.comp (algebraMap k K)).toAlgebra
    this : Algebra K Complex := φ.toAlgebra
    ⊢ False
  -/
  have : IsScalarTower k K ℂ := IsScalarTower.of_algebraMap_eq' rfl
  /-
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    φ : RingHom K Complex
    H : ∀ (σ : AlgEquiv k K K), NumberField.ComplexEmbedding.IsConj φ σ → Eq σ 1
    hφ : And (Not (NumberField.ComplexEmbedding.IsConj φ 1)) (NumberField.ComplexE …
    this✝¹ : Algebra k Complex := (φ.comp (algebraMap k K)).toAlgebra
    this✝ : Algebra K Complex := φ.toAlgebra
    this : IsScalarTower k K Complex
    ⊢ False
  -/
  let φ' : K →ₐ[k] ℂ := { star φ with commutes' := fun r ↦ by simpa using RingHom.congr_fun hφ.2 r }
  have : ComplexEmbedding.IsConj φ (AlgHom.restrictNormal' φ' K) :=
    (RingHom.ext <| AlgHom.restrictNormal_commutes φ' K).symm
  /-
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    φ : RingHom K Complex
    H : ∀ (σ : AlgEquiv k K K), NumberField.ComplexEmbedding.IsConj φ σ → Eq σ 1
    hφ : And (Not (NumberField.ComplexEmbedding.IsConj φ 1)) (NumberField.ComplexE …
    this✝² : Algebra k Complex := (φ.comp (algebraMap k K)).toAlgebra
    this✝¹ : Algebra K Complex := φ.toAlgebra
    this✝ : IsScalarTower k K Complex
    φ' : AlgHom k K Complex :=
      let __src := Star.star φ;
      { toRingHom := __src, commutes' := ⋯ }
    this : NumberField.ComplexEmbedding.IsConj φ (φ'.restrictNormal' K)
    ⊢ False
  -/
  exact hφ.1 (H _ this ▸ this)
  /-
    🎉 no goals
  -/


local notation "Stab" => MulAction.stabilizer (K ≃ₐ[k] K)


lemma mem_stabilizer_mk_iff (φ : K →+* ℂ) (σ : K ≃ₐ[k] K) :
    σ ∈ Stab (mk φ) ↔ σ = 1 ∨ ComplexEmbedding.IsConj φ σ := by
  /-
    k : Type u_1
    inst✝² : Field k
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : Algebra k K
    φ : RingHom K Complex
    σ : AlgEquiv k K K
    ⊢ Iff (Membership.mem (MulAction.stabilizer (AlgEquiv k K K) (NumberField.Infi …
  -/
  simp only [MulAction.mem_stabilizer_iff, smul_mk, mk_eq_iff]
  /-
    k : Type u_1
    inst✝² : Field k
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : Algebra k K
    φ : RingHom K Complex
    σ : AlgEquiv k K K
    ⊢ Iff (Or (Eq (φ.comp ↑σ.symm) φ) (Eq (NumberField.ComplexEmbedding.conjugate  …
  -/
  rw [← ComplexEmbedding.isConj_symm, ComplexEmbedding.conjugate, star_eq_iff_star_eq]
  /-
    k : Type u_1
    inst✝² : Field k
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : Algebra k K
    φ : RingHom K Complex
    σ : AlgEquiv k K K
    ⊢ Iff (Or (Eq (φ.comp ↑σ.symm) φ) (Eq (Star.star φ) (φ.comp ↑σ.symm))) (Or (Eq …
  -/
  refine or_congr ⟨fun H ↦ ?_, fun H ↦ H ▸ rfl⟩ Iff.rfl
  exact congr_arg AlgEquiv.symm
    (AlgEquiv.ext (g := AlgEquiv.refl) fun x ↦ φ.injective (RingHom.congr_fun H x))


lemma IsUnramified.stabilizer_eq_bot (h : IsUnramified k w) : Stab w = ⊥ := by
  /-
    k : Type u_1
    inst✝² : Field k
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : Algebra k K
    w : NumberField.InfinitePlace K
    h : NumberField.InfinitePlace.IsUnramified k w
    ⊢ Eq (MulAction.stabilizer (AlgEquiv k K K) w) Bot.bot
  -/
  rw [eq_bot_iff, ← mk_embedding w, SetLike.le_def]
  /-
    k : Type u_1
    inst✝² : Field k
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : Algebra k K
    w : NumberField.InfinitePlace K
    h : NumberField.InfinitePlace.IsUnramified k w
    ⊢ ∀ ⦃x : AlgEquiv k K K⦄, Membership.mem (MulAction.stabilizer (AlgEquiv k K K …
  -/
  simp only [mem_stabilizer_mk_iff, Subgroup.mem_bot, forall_eq_or_imp, true_and]
  /-
    k : Type u_1
    inst✝² : Field k
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : Algebra k K
    w : NumberField.InfinitePlace K
    h : NumberField.InfinitePlace.IsUnramified k w
    ⊢ ∀ (a : AlgEquiv k K K), NumberField.ComplexEmbedding.IsConj w.embedding a →  …
  -/
  exact fun σ hσ ↦ hσ.isUnramified_mk_iff.mp ((mk_embedding w).symm ▸ h)
  /-
    🎉 no goals
  -/


lemma _root_.NumberField.ComplexEmbedding.IsConj.coe_stabilzer_mk
    {φ : K →+* ℂ} (h : ComplexEmbedding.IsConj φ σ) :
    (Stab (mk φ) : Set (K ≃ₐ[k] K)) = {1, σ} := by
  /-
    k : Type u_1
    inst✝² : Field k
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : Algebra k K
    σ : AlgEquiv k K K
    φ : RingHom K Complex
    h : NumberField.ComplexEmbedding.IsConj φ σ
    ⊢ Eq (↑(MulAction.stabilizer (AlgEquiv k K K) (NumberField.InfinitePlace.mk φ) …
  -/
  ext
  rw [SetLike.mem_coe, mem_stabilizer_mk_iff, Set.mem_insert_iff, Set.mem_singleton_iff,
    ← h.ext_iff, eq_comm (a := σ)]


lemma nat_card_stabilizer_eq_one_or_two :
    Nat.card (Stab w) = 1 ∨ Nat.card (Stab w) = 2 := by
  /-
    k : Type u_1
    inst✝² : Field k
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : Algebra k K
    w : NumberField.InfinitePlace K
    ⊢ Or (Eq (Nat.card (Subtype fun x => Membership.mem (MulAction.stabilizer (Alg …
  -/
  rw [← SetLike.coe_sort_coe, ← mk_embedding w]
  /-
    k : Type u_1
    inst✝² : Field k
    K : Type u_2
    inst✝¹ : Field K
    inst✝ : Algebra k K
    w : NumberField.InfinitePlace K
    ⊢ Or (Eq (Nat.card ↑↑(MulAction.stabilizer (AlgEquiv k K K) (NumberField.Infin …
  -/
  by_cases h : ∃ σ, ComplexEmbedding.IsConj (k := k) (embedding w) σ
    /-
      case pos
      k : Type u_1
      inst✝² : Field k
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : Algebra k K
      w : NumberField.InfinitePlace K
      h : Exists fun σ => NumberField.ComplexEmbedding.IsConj w.embedding σ
      ⊢ Or (Eq (Nat.card ↑↑(MulAction.stabilizer (AlgEquiv k K K) (NumberField.Infin …
    -/
  · obtain ⟨σ, hσ⟩ := h
    /-
      case pos.intro
      k : Type u_1
      inst✝² : Field k
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : Algebra k K
      w : NumberField.InfinitePlace K
      σ : AlgEquiv k K K
      hσ : NumberField.ComplexEmbedding.IsConj w.embedding σ
      ⊢ Or (Eq (Nat.card ↑↑(MulAction.stabilizer (AlgEquiv k K K) (NumberField.Infin …
    -/
    simp only [hσ.coe_stabilzer_mk, Nat.card_eq_fintype_card, card_ofFinset, Set.toFinset_singleton]
    /-
      case pos.intro
      k : Type u_1
      inst✝² : Field k
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : Algebra k K
      w : NumberField.InfinitePlace K
      σ : AlgEquiv k K K
      hσ : NumberField.ComplexEmbedding.IsConj w.embedding σ
      ⊢ Or (Eq (Insert.insert 1 (Singleton.singleton σ)).card 1) (Eq (Insert.insert  …
    -/
    by_cases 1 = σ
      /-
        case pos
        k : Type u_1
        inst✝² : Field k
        K : Type u_2
        inst✝¹ : Field K
        inst✝ : Algebra k K
        w : NumberField.InfinitePlace K
        σ : AlgEquiv k K K
        hσ : NumberField.ComplexEmbedding.IsConj w.embedding σ
        h✝ : Eq 1 σ
        ⊢ Or (Eq (Insert.insert 1 (Singleton.singleton σ)).card 1) (Eq (Insert.insert  …
      -/
    · left; simp [*]
            /-
              🎉 no goals
            -/
      /-
        case neg
        k : Type u_1
        inst✝² : Field k
        K : Type u_2
        inst✝¹ : Field K
        inst✝ : Algebra k K
        w : NumberField.InfinitePlace K
        σ : AlgEquiv k K K
        hσ : NumberField.ComplexEmbedding.IsConj w.embedding σ
        h✝ : Not (Eq 1 σ)
        ⊢ Or (Eq (Insert.insert 1 (Singleton.singleton σ)).card 1) (Eq (Insert.insert  …
      -/
    · right; simp [*]
             /-
               🎉 no goals
             -/
    /-
      case neg
      k : Type u_1
      inst✝² : Field k
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : Algebra k K
      w : NumberField.InfinitePlace K
      h : Not (Exists fun σ => NumberField.ComplexEmbedding.IsConj w.embedding σ)
      ⊢ Or (Eq (Nat.card ↑↑(MulAction.stabilizer (AlgEquiv k K K) (NumberField.Infin …
    -/
  · push_neg at h
    /-
      case neg
      k : Type u_1
      inst✝² : Field k
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : Algebra k K
      w : NumberField.InfinitePlace K
      h : ∀ (σ : AlgEquiv k K K), Not (NumberField.ComplexEmbedding.IsConj w.embeddi …
      ⊢ Or (Eq (Nat.card ↑↑(MulAction.stabilizer (AlgEquiv k K K) (NumberField.Infin …
    -/
    left
    /-
      case neg.h
      k : Type u_1
      inst✝² : Field k
      K : Type u_2
      inst✝¹ : Field K
      inst✝ : Algebra k K
      w : NumberField.InfinitePlace K
      h : ∀ (σ : AlgEquiv k K K), Not (NumberField.ComplexEmbedding.IsConj w.embeddi …
      ⊢ Eq (Nat.card ↑↑(MulAction.stabilizer (AlgEquiv k K K) (NumberField.InfiniteP …
    -/
    trans Nat.card ({1} : Set (K ≃ₐ[k] K))
      /-
        k : Type u_1
        inst✝² : Field k
        K : Type u_2
        inst✝¹ : Field K
        inst✝ : Algebra k K
        w : NumberField.InfinitePlace K
        h : ∀ (σ : AlgEquiv k K K), Not (NumberField.ComplexEmbedding.IsConj w.embeddi …
        ⊢ Eq (Nat.card ↑↑(MulAction.stabilizer (AlgEquiv k K K) (NumberField.InfiniteP …
      -/
    · congr with x
      simp only [SetLike.mem_coe, mem_stabilizer_mk_iff, Set.mem_singleton_iff, or_iff_left_iff_imp,
        h x, IsEmpty.forall_iff]
      /-
        k : Type u_1
        inst✝² : Field k
        K : Type u_2
        inst✝¹ : Field K
        inst✝ : Algebra k K
        w : NumberField.InfinitePlace K
        h : ∀ (σ : AlgEquiv k K K), Not (NumberField.ComplexEmbedding.IsConj w.embeddi …
        ⊢ Eq (Nat.card ↑(Singleton.singleton 1)) 1
      -/
    · simp
      /-
        🎉 no goals
      -/


lemma isUnramified_iff_stabilizer_eq_bot [IsGalois k K] : IsUnramified k w ↔ Stab w = ⊥ := by
  /-
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    w : NumberField.InfinitePlace K
    inst✝ : IsGalois k K
    ⊢ Iff (NumberField.InfinitePlace.IsUnramified k w) (Eq (MulAction.stabilizer ( …
  -/
  rw [← mk_embedding w, isUnramified_mk_iff_forall_isConj]
  simp only [eq_bot_iff, SetLike.le_def, mem_stabilizer_mk_iff,
    Subgroup.mem_bot, forall_eq_or_imp, true_and]


lemma isUnramified_iff_card_stabilizer_eq_one [IsGalois k K] :
    IsUnramified k w ↔ Nat.card (Stab w) = 1 := by
  /-
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    w : NumberField.InfinitePlace K
    inst✝ : IsGalois k K
    ⊢ Iff (NumberField.InfinitePlace.IsUnramified k w) (Eq (Nat.card (Subtype fun  …
  -/
  rw [isUnramified_iff_stabilizer_eq_bot, Subgroup.card_eq_one]
  /-
    🎉 no goals
  -/


lemma not_isUnramified_iff_card_stabilizer_eq_two [IsGalois k K] :
    ¬ IsUnramified k w ↔ Nat.card (Stab w) = 2 := by
  /-
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    w : NumberField.InfinitePlace K
    inst✝ : IsGalois k K
    ⊢ Iff (Not (NumberField.InfinitePlace.IsUnramified k w)) (Eq (Nat.card (Subtyp …
  -/
  rw [isUnramified_iff_card_stabilizer_eq_one]
  /-
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    w : NumberField.InfinitePlace K
    inst✝ : IsGalois k K
    ⊢ Iff (Not (Eq (Nat.card (Subtype fun x => Membership.mem (MulAction.stabilize …
  -/
                                                                       /-
                                                                         🎉 no goals
                                                                       -/
  obtain (e|e) := nat_card_stabilizer_eq_one_or_two k w <;> rw [e] <;> decide
                                                                       /-
                                                                         🎉 no goals
                                                                       -/


lemma card_stabilizer [IsGalois k K] :
    Nat.card (Stab w) = if IsUnramified k w then 1 else 2 := by
  /-
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    w : NumberField.InfinitePlace K
    inst✝ : IsGalois k K
    ⊢ Eq (Nat.card (Subtype fun x => Membership.mem (MulAction.stabilizer (AlgEqui …
  -/
  split
    /-
      case isTrue
      k : Type u_1
      inst✝³ : Field k
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra k K
      w : NumberField.InfinitePlace K
      inst✝ : IsGalois k K
      h✝ : NumberField.InfinitePlace.IsUnramified k w
      ⊢ Eq (Nat.card (Subtype fun x => Membership.mem (MulAction.stabilizer (AlgEqui …
    -/
  · rwa [← isUnramified_iff_card_stabilizer_eq_one]
    /-
      🎉 no goals
    -/
    /-
      case isFalse
      k : Type u_1
      inst✝³ : Field k
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra k K
      w : NumberField.InfinitePlace K
      inst✝ : IsGalois k K
      h✝ : Not (NumberField.InfinitePlace.IsUnramified k w)
      ⊢ Eq (Nat.card (Subtype fun x => Membership.mem (MulAction.stabilizer (AlgEqui …
    -/
  · rwa [← not_isUnramified_iff_card_stabilizer_eq_two]
    /-
      🎉 no goals
    -/


lemma even_nat_card_aut_of_not_isUnramified [IsGalois k K] (hw : ¬ IsUnramified k w) :
    Even (Nat.card <| K ≃ₐ[k] K) := by
  /-
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    w : NumberField.InfinitePlace K
    inst✝ : IsGalois k K
    hw : Not (NumberField.InfinitePlace.IsUnramified k w)
    ⊢ Even (Nat.card (AlgEquiv k K K))
  -/
  by_cases H : Finite (K ≃ₐ[k] K)
    /-
      case pos
      k : Type u_1
      inst✝³ : Field k
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra k K
      w : NumberField.InfinitePlace K
      inst✝ : IsGalois k K
      hw : Not (NumberField.InfinitePlace.IsUnramified k w)
      H : Finite (AlgEquiv k K K)
      ⊢ Even (Nat.card (AlgEquiv k K K))
    -/
  · cases nonempty_fintype (K ≃ₐ[k] K)
    /-
      case pos.intro
      k : Type u_1
      inst✝³ : Field k
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra k K
      w : NumberField.InfinitePlace K
      inst✝ : IsGalois k K
      hw : Not (NumberField.InfinitePlace.IsUnramified k w)
      H : Finite (AlgEquiv k K K)
      val✝ : Fintype (AlgEquiv k K K)
      ⊢ Even (Nat.card (AlgEquiv k K K))
    -/
    rw [even_iff_two_dvd, ← not_isUnramified_iff_card_stabilizer_eq_two.mp hw]
    /-
      case pos.intro
      k : Type u_1
      inst✝³ : Field k
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra k K
      w : NumberField.InfinitePlace K
      inst✝ : IsGalois k K
      hw : Not (NumberField.InfinitePlace.IsUnramified k w)
      H : Finite (AlgEquiv k K K)
      val✝ : Fintype (AlgEquiv k K K)
      ⊢ Dvd.dvd (Nat.card (Subtype fun x => Membership.mem (MulAction.stabilizer (Al …
    -/
    exact Subgroup.card_subgroup_dvd_card (Stab w)
    /-
      🎉 no goals
    -/
    /-
      case neg
      k : Type u_1
      inst✝³ : Field k
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra k K
      w : NumberField.InfinitePlace K
      inst✝ : IsGalois k K
      hw : Not (NumberField.InfinitePlace.IsUnramified k w)
      H : Not (Finite (AlgEquiv k K K))
      ⊢ Even (Nat.card (AlgEquiv k K K))
    -/
  · convert even_zero
    /-
      case h.e'_3
      k : Type u_1
      inst✝³ : Field k
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra k K
      w : NumberField.InfinitePlace K
      inst✝ : IsGalois k K
      hw : Not (NumberField.InfinitePlace.IsUnramified k w)
      H : Not (Finite (AlgEquiv k K K))
      ⊢ Eq (Nat.card (AlgEquiv k K K)) 0
    -/
    by_contra e
    /-
      case h.e'_3
      k : Type u_1
      inst✝³ : Field k
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra k K
      w : NumberField.InfinitePlace K
      inst✝ : IsGalois k K
      hw : Not (NumberField.InfinitePlace.IsUnramified k w)
      H : Not (Finite (AlgEquiv k K K))
      e : Not (Eq (Nat.card (AlgEquiv k K K)) 0)
      ⊢ False
    -/
    exact H (Nat.finite_of_card_ne_zero e)
    /-
      🎉 no goals
    -/


lemma even_card_aut_of_not_isUnramified [IsGalois k K] [FiniteDimensional k K]
    (hw : ¬ IsUnramified k w) :
    Even (Fintype.card <| K ≃ₐ[k] K) :=
  Nat.card_eq_fintype_card (α := K ≃ₐ[k] K) ▸ even_nat_card_aut_of_not_isUnramified hw


lemma even_finrank_of_not_isUnramified [IsGalois k K]
    (hw : ¬ IsUnramified k w) : Even (finrank k K) := by
  /-
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    w : NumberField.InfinitePlace K
    inst✝ : IsGalois k K
    hw : Not (NumberField.InfinitePlace.IsUnramified k w)
    ⊢ Even (Module.finrank k K)
  -/
  by_cases FiniteDimensional k K
    /-
      case pos
      k : Type u_1
      inst✝³ : Field k
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra k K
      w : NumberField.InfinitePlace K
      inst✝ : IsGalois k K
      hw : Not (NumberField.InfinitePlace.IsUnramified k w)
      h✝ : FiniteDimensional k K
      ⊢ Even (Module.finrank k K)
    -/
  · exact IsGalois.card_aut_eq_finrank k K ▸ even_card_aut_of_not_isUnramified hw
    /-
      🎉 no goals
    -/
    /-
      case neg
      k : Type u_1
      inst✝³ : Field k
      K : Type u_2
      inst✝² : Field K
      inst✝¹ : Algebra k K
      w : NumberField.InfinitePlace K
      inst✝ : IsGalois k K
      hw : Not (NumberField.InfinitePlace.IsUnramified k w)
      h✝ : Not (FiniteDimensional k K)
      ⊢ Even (Module.finrank k K)
    -/
  · exact finrank_of_not_finite ‹_› ▸ even_zero
    /-
      🎉 no goals
    -/


lemma isUnramified_smul_iff :
    IsUnramified k (σ • w) ↔ IsUnramified k w := by
  rw [isUnramified_iff, isUnramified_iff, isReal_smul_iff, comap_smul,
    ← AlgEquiv.toAlgHom_toRingHom, AlgHom.comp_algebraMap]


/-- A infinite place of the base field is unramified in a field extension if every
infinite place over it is unramified. -/
def IsUnramifiedIn (w : InfinitePlace k) : Prop :=
  ∀ v, comap v (algebraMap k K) = w → IsUnramified k v


lemma isUnramifiedIn_comap [IsGalois k K] {w : InfinitePlace K} :
    (w.comap (algebraMap k K)).IsUnramifiedIn K ↔ w.IsUnramified k := by
  /-
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    w : NumberField.InfinitePlace K
    ⊢ Iff (NumberField.InfinitePlace.IsUnramifiedIn K (w.comap (algebraMap k K)))  …
  -/
  refine ⟨fun H ↦ H _ rfl, fun H v hv ↦ ?_⟩
  /-
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    w : NumberField.InfinitePlace K
    H : NumberField.InfinitePlace.IsUnramified k w
    v : NumberField.InfinitePlace K
    hv : Eq (v.comap (algebraMap k K)) (w.comap (algebraMap k K))
    ⊢ NumberField.InfinitePlace.IsUnramified k v
  -/
  obtain ⟨σ, rfl⟩ := exists_smul_eq_of_comap_eq hv
  /-
    case intro
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    v : NumberField.InfinitePlace K
    σ : AlgEquiv k K K
    H : NumberField.InfinitePlace.IsUnramified k (HSMul.hSMul σ v)
    hv : Eq (v.comap (algebraMap k K)) ((HSMul.hSMul σ v).comap (algebraMap k K))
    ⊢ NumberField.InfinitePlace.IsUnramified k v
  -/
  rwa [isUnramified_smul_iff] at H
  /-
    🎉 no goals
  -/


lemma even_card_aut_of_not_isUnramifiedIn [IsGalois k K] [FiniteDimensional k K]
    {w : InfinitePlace k} (hw : ¬ w.IsUnramifiedIn K) :
    Even (Fintype.card <| K ≃ₐ[k] K) := by
  /-
    k : Type u_1
    inst✝⁴ : Field k
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra k K
    inst✝¹ : IsGalois k K
    inst✝ : FiniteDimensional k K
    w : NumberField.InfinitePlace k
    hw : Not (NumberField.InfinitePlace.IsUnramifiedIn K w)
    ⊢ Even (Fintype.card (AlgEquiv k K K))
  -/
  obtain ⟨v, rfl⟩ := comap_surjective (K := K) w
  /-
    case intro
    k : Type u_1
    inst✝⁴ : Field k
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra k K
    inst✝¹ : IsGalois k K
    inst✝ : FiniteDimensional k K
    v : NumberField.InfinitePlace K
    hw : Not (NumberField.InfinitePlace.IsUnramifiedIn K ((fun x => x.comap (algeb …
    ⊢ Even (Fintype.card (AlgEquiv k K K))
  -/
  rw [isUnramifiedIn_comap] at hw
  /-
    case intro
    k : Type u_1
    inst✝⁴ : Field k
    K : Type u_2
    inst✝³ : Field K
    inst✝² : Algebra k K
    inst✝¹ : IsGalois k K
    inst✝ : FiniteDimensional k K
    v : NumberField.InfinitePlace K
    hw : Not (NumberField.InfinitePlace.IsUnramified k v)
    ⊢ Even (Fintype.card (AlgEquiv k K K))
  -/
  exact even_card_aut_of_not_isUnramified hw
  /-
    🎉 no goals
  -/


lemma even_finrank_of_not_isUnramifiedIn
    [IsGalois k K] {w : InfinitePlace k} (hw : ¬ w.IsUnramifiedIn K) :
    Even (finrank k K) := by
  /-
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    w : NumberField.InfinitePlace k
    hw : Not (NumberField.InfinitePlace.IsUnramifiedIn K w)
    ⊢ Even (Module.finrank k K)
  -/
  obtain ⟨v, rfl⟩ := comap_surjective (K := K) w
  /-
    case intro
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    v : NumberField.InfinitePlace K
    hw : Not (NumberField.InfinitePlace.IsUnramifiedIn K ((fun x => x.comap (algeb …
    ⊢ Even (Module.finrank k K)
  -/
  rw [isUnramifiedIn_comap] at hw
  /-
    case intro
    k : Type u_1
    inst✝³ : Field k
    K : Type u_2
    inst✝² : Field K
    inst✝¹ : Algebra k K
    inst✝ : IsGalois k K
    v : NumberField.InfinitePlace K
    hw : Not (NumberField.InfinitePlace.IsUnramified k v)
    ⊢ Even (Module.finrank k K)
  -/
  exact even_finrank_of_not_isUnramified hw
  /-
    🎉 no goals
  -/


open Finset in
lemma card_isUnramified [NumberField k] [IsGalois k K] :
    #{w : InfinitePlace K | w.IsUnramified k} =
      #{w : InfinitePlace k | w.IsUnramifiedIn K} * finrank k K := by
  /-
    k : Type u_1
    inst✝⁵ : Field k
    K : Type u_2
    inst✝⁴ : Field K
    inst✝³ : Algebra k K
    inst✝² : NumberField K
    inst✝¹ : NumberField k
    inst✝ : IsGalois k K
    ⊢ Eq (Finset.filter (fun w => NumberField.InfinitePlace.IsUnramified k w) Fins …
  -/
  letI := Module.Finite.of_restrictScalars_finite ℚ k K
  rw [← IsGalois.card_aut_eq_finrank,
    Finset.card_eq_sum_card_fiberwise (f := (comap · (algebraMap k K)))
    (t := {w : InfinitePlace k | w.IsUnramifiedIn K}), ← smul_eq_mul, ← sum_const]
    /-
      k : Type u_1
      inst✝⁵ : Field k
      K : Type u_2
      inst✝⁴ : Field K
      inst✝³ : Algebra k K
      inst✝² : NumberField K
      inst✝¹ : NumberField k
      inst✝ : IsGalois k K
      this : Module.Finite k K := Module.Finite.of_restrictScalars_finite Rat k K
      ⊢ Eq ((Finset.filter (fun w => NumberField.InfinitePlace.IsUnramifiedIn K w) F …
    -/
  · refine sum_congr rfl (fun w hw ↦ ?_)
    /-
      k : Type u_1
      inst✝⁵ : Field k
      K : Type u_2
      inst✝⁴ : Field K
      inst✝³ : Algebra k K
      inst✝² : NumberField K
      inst✝¹ : NumberField k
      inst✝ : IsGalois k K
      this : Module.Finite k K := Module.Finite.of_restrictScalars_finite Rat k K
      w : NumberField.InfinitePlace k
      hw : Membership.mem (Finset.filter (fun w => NumberField.InfinitePlace.IsUnram …
      ⊢ Eq (Finset.filter (fun a => Eq (a.comap (algebraMap k K)) w) (Finset.filter  …
    -/
    obtain ⟨w, rfl⟩ := comap_surjective (K := K) w
    /-
      case intro
      k : Type u_1
      inst✝⁵ : Field k
      K : Type u_2
      inst✝⁴ : Field K
      inst✝³ : Algebra k K
      inst✝² : NumberField K
      inst✝¹ : NumberField k
      inst✝ : IsGalois k K
      this : Module.Finite k K := Module.Finite.of_restrictScalars_finite Rat k K
      w : NumberField.InfinitePlace K
      hw : Membership.mem (Finset.filter (fun w => NumberField.InfinitePlace.IsUnram …
      ⊢ Eq (Finset.filter (fun a => Eq (a.comap (algebraMap k K)) ((fun x => x.comap …
    -/
    simp only [mem_univ, forall_true_left, mem_filter, true_and] at hw
    /-
      case intro
      k : Type u_1
      inst✝⁵ : Field k
      K : Type u_2
      inst✝⁴ : Field K
      inst✝³ : Algebra k K
      inst✝² : NumberField K
      inst✝¹ : NumberField k
      inst✝ : IsGalois k K
      this : Module.Finite k K := Module.Finite.of_restrictScalars_finite Rat k K
      w : NumberField.InfinitePlace K
      hw : NumberField.InfinitePlace.IsUnramifiedIn K (w.comap (algebraMap k K))
      ⊢ Eq (Finset.filter (fun a => Eq (a.comap (algebraMap k K)) ((fun x => x.comap …
    -/
    trans #(MulAction.orbit (K ≃ₐ[k] K) w).toFinset
      /-
        k : Type u_1
        inst✝⁵ : Field k
        K : Type u_2
        inst✝⁴ : Field K
        inst✝³ : Algebra k K
        inst✝² : NumberField K
        inst✝¹ : NumberField k
        inst✝ : IsGalois k K
        this : Module.Finite k K := Module.Finite.of_restrictScalars_finite Rat k K
        w : NumberField.InfinitePlace K
        hw : NumberField.InfinitePlace.IsUnramifiedIn K (w.comap (algebraMap k K))
        ⊢ Eq (Finset.filter (fun a => Eq (a.comap (algebraMap k K)) ((fun x => x.comap …
      -/
    · congr; ext w'
      simp only [mem_univ, forall_true_left, filter_congr_decidable, mem_filter, true_and,
        Set.mem_toFinset, mem_orbit_iff, @eq_comm _ (comap w' _), and_iff_right_iff_imp]
      /-
        case e_s.h
        k : Type u_1
        inst✝⁵ : Field k
        K : Type u_2
        inst✝⁴ : Field K
        inst✝³ : Algebra k K
        inst✝² : NumberField K
        inst✝¹ : NumberField k
        inst✝ : IsGalois k K
        this : Module.Finite k K := Module.Finite.of_restrictScalars_finite Rat k K
        w : NumberField.InfinitePlace K
        hw : NumberField.InfinitePlace.IsUnramifiedIn K (w.comap (algebraMap k K))
        w' : NumberField.InfinitePlace K
        ⊢ Eq (w.comap (algebraMap k K)) (w'.comap (algebraMap k K)) → NumberField.Infi …
      -/
      intro e; rwa [← isUnramifiedIn_comap, ← e]
               /-
                 🎉 no goals
               -/
    · rw [← MulAction.card_orbit_mul_card_stabilizer_eq_card_group _ w,
        ← Nat.card_eq_fintype_card (α := Stab w), card_stabilizer, if_pos,
        mul_one, Set.toFinset_card]
      /-
        case hc
        k : Type u_1
        inst✝⁵ : Field k
        K : Type u_2
        inst✝⁴ : Field K
        inst✝³ : Algebra k K
        inst✝² : NumberField K
        inst✝¹ : NumberField k
        inst✝ : IsGalois k K
        this : Module.Finite k K := Module.Finite.of_restrictScalars_finite Rat k K
        w : NumberField.InfinitePlace K
        hw : NumberField.InfinitePlace.IsUnramifiedIn K (w.comap (algebraMap k K))
        ⊢ NumberField.InfinitePlace.IsUnramified k w
      -/
      rwa [← isUnramifiedIn_comap]
      /-
        🎉 no goals
      -/
    /-
      k : Type u_1
      inst✝⁵ : Field k
      K : Type u_2
      inst✝⁴ : Field K
      inst✝³ : Algebra k K
      inst✝² : NumberField K
      inst✝¹ : NumberField k
      inst✝ : IsGalois k K
      this : Module.Finite k K := Module.Finite.of_restrictScalars_finite Rat k K
      ⊢ ∀ (x : NumberField.InfinitePlace K), Membership.mem (Finset.filter (fun w => …
    -/
  · simp [isUnramifiedIn_comap]
    /-
      🎉 no goals
    -/


open Finset in
lemma card_isUnramified_compl [NumberField k] [IsGalois k K] :
    #({w : InfinitePlace K | w.IsUnramified k} : Finset _)ᶜ =
      #({w : InfinitePlace k | w.IsUnramifiedIn K} : Finset _)ᶜ * (finrank k K / 2) := by
  /-
    k : Type u_1
    inst✝⁵ : Field k
    K : Type u_2
    inst✝⁴ : Field K
    inst✝³ : Algebra k K
    inst✝² : NumberField K
    inst✝¹ : NumberField k
    inst✝ : IsGalois k K
    ⊢ Eq (HasCompl.compl (Finset.filter (fun w => NumberField.InfinitePlace.IsUnra …
  -/
  letI := Module.Finite.of_restrictScalars_finite ℚ k K
  rw [← IsGalois.card_aut_eq_finrank,
    Finset.card_eq_sum_card_fiberwise (f := (comap · (algebraMap k K)))
    (t := ({w : InfinitePlace k | w.IsUnramifiedIn K}: Finset _)ᶜ), ← smul_eq_mul, ← sum_const]
    /-
      k : Type u_1
      inst✝⁵ : Field k
      K : Type u_2
      inst✝⁴ : Field K
      inst✝³ : Algebra k K
      inst✝² : NumberField K
      inst✝¹ : NumberField k
      inst✝ : IsGalois k K
      this : Module.Finite k K := Module.Finite.of_restrictScalars_finite Rat k K
      ⊢ Eq ((HasCompl.compl (Finset.filter (fun w => NumberField.InfinitePlace.IsUnr …
    -/
  · refine sum_congr rfl (fun w hw ↦ ?_)
    /-
      k : Type u_1
      inst✝⁵ : Field k
      K : Type u_2
      inst✝⁴ : Field K
      inst✝³ : Algebra k K
      inst✝² : NumberField K
      inst✝¹ : NumberField k
      inst✝ : IsGalois k K
      this : Module.Finite k K := Module.Finite.of_restrictScalars_finite Rat k K
      w : NumberField.InfinitePlace k
      hw : Membership.mem (HasCompl.compl (Finset.filter (fun w => NumberField.Infin …
      ⊢ Eq (Finset.filter (fun a => Eq (a.comap (algebraMap k K)) w) (HasCompl.compl …
    -/
    obtain ⟨w, rfl⟩ := comap_surjective (K := K) w
    /-
      case intro
      k : Type u_1
      inst✝⁵ : Field k
      K : Type u_2
      inst✝⁴ : Field K
      inst✝³ : Algebra k K
      inst✝² : NumberField K
      inst✝¹ : NumberField k
      inst✝ : IsGalois k K
      this : Module.Finite k K := Module.Finite.of_restrictScalars_finite Rat k K
      w : NumberField.InfinitePlace K
      hw : Membership.mem (HasCompl.compl (Finset.filter (fun w => NumberField.Infin …
      ⊢ Eq (Finset.filter (fun a => Eq (a.comap (algebraMap k K)) ((fun x => x.comap …
    -/
    simp only [mem_univ, forall_true_left, compl_filter, not_not, mem_filter, true_and] at hw
    /-
      case intro
      k : Type u_1
      inst✝⁵ : Field k
      K : Type u_2
      inst✝⁴ : Field K
      inst✝³ : Algebra k K
      inst✝² : NumberField K
      inst✝¹ : NumberField k
      inst✝ : IsGalois k K
      this : Module.Finite k K := Module.Finite.of_restrictScalars_finite Rat k K
      w : NumberField.InfinitePlace K
      hw : Not (NumberField.InfinitePlace.IsUnramifiedIn K (w.comap (algebraMap k K)))
      ⊢ Eq (Finset.filter (fun a => Eq (a.comap (algebraMap k K)) ((fun x => x.comap …
    -/
    trans Finset.card (MulAction.orbit (K ≃ₐ[k] K) w).toFinset
      /-
        k : Type u_1
        inst✝⁵ : Field k
        K : Type u_2
        inst✝⁴ : Field K
        inst✝³ : Algebra k K
        inst✝² : NumberField K
        inst✝¹ : NumberField k
        inst✝ : IsGalois k K
        this : Module.Finite k K := Module.Finite.of_restrictScalars_finite Rat k K
        w : NumberField.InfinitePlace K
        hw : Not (NumberField.InfinitePlace.IsUnramifiedIn K (w.comap (algebraMap k K)))
        ⊢ Eq (Finset.filter (fun a => Eq (a.comap (algebraMap k K)) ((fun x => x.comap …
      -/
    · congr; ext w'
      simp only [compl_filter, filter_congr_decidable, mem_filter, mem_univ, true_and,
        @eq_comm _ (comap w' _), Set.mem_toFinset, mem_orbit_iff, and_iff_right_iff_imp]
      /-
        case e_s.h
        k : Type u_1
        inst✝⁵ : Field k
        K : Type u_2
        inst✝⁴ : Field K
        inst✝³ : Algebra k K
        inst✝² : NumberField K
        inst✝¹ : NumberField k
        inst✝ : IsGalois k K
        this : Module.Finite k K := Module.Finite.of_restrictScalars_finite Rat k K
        w : NumberField.InfinitePlace K
        hw : Not (NumberField.InfinitePlace.IsUnramifiedIn K (w.comap (algebraMap k K)))
        w' : NumberField.InfinitePlace K
        ⊢ Eq (w.comap (algebraMap k K)) (w'.comap (algebraMap k K)) → Not (NumberField …
      -/
      intro e; rwa [← isUnramifiedIn_comap, ← e]
               /-
                 🎉 no goals
               -/
    · rw [← MulAction.card_orbit_mul_card_stabilizer_eq_card_group _ w,
        ← Nat.card_eq_fintype_card (α := Stab w), InfinitePlace.card_stabilizer, if_neg,
        Nat.mul_div_cancel _ zero_lt_two, Set.toFinset_card]
      /-
        case hnc
        k : Type u_1
        inst✝⁵ : Field k
        K : Type u_2
        inst✝⁴ : Field K
        inst✝³ : Algebra k K
        inst✝² : NumberField K
        inst✝¹ : NumberField k
        inst✝ : IsGalois k K
        this : Module.Finite k K := Module.Finite.of_restrictScalars_finite Rat k K
        w : NumberField.InfinitePlace K
        hw : Not (NumberField.InfinitePlace.IsUnramifiedIn K (w.comap (algebraMap k K)))
        ⊢ Not (NumberField.InfinitePlace.IsUnramified k w)
      -/
      rwa [← isUnramifiedIn_comap]
      /-
        🎉 no goals
      -/
    /-
      k : Type u_1
      inst✝⁵ : Field k
      K : Type u_2
      inst✝⁴ : Field K
      inst✝³ : Algebra k K
      inst✝² : NumberField K
      inst✝¹ : NumberField k
      inst✝ : IsGalois k K
      this : Module.Finite k K := Module.Finite.of_restrictScalars_finite Rat k K
      ⊢ ∀ (x : NumberField.InfinitePlace K), Membership.mem (HasCompl.compl (Finset. …
    -/
  · simp [isUnramifiedIn_comap]
    /-
      🎉 no goals
    -/


lemma card_eq_card_isUnramifiedIn [NumberField k] [IsGalois k K] :
    Fintype.card (InfinitePlace K) =
      #{w : InfinitePlace k | w.IsUnramifiedIn K} * finrank k K +
      #({w : InfinitePlace k | w.IsUnramifiedIn K} : Finset _)ᶜ * (finrank k K / 2) := by
  /-
    k : Type u_1
    inst✝⁵ : Field k
    K : Type u_2
    inst✝⁴ : Field K
    inst✝³ : Algebra k K
    inst✝² : NumberField K
    inst✝¹ : NumberField k
    inst✝ : IsGalois k K
    ⊢ Eq (Fintype.card (NumberField.InfinitePlace K)) (HAdd.hAdd (HMul.hMul (Finse …
  -/
  rw [← card_isUnramified, ← card_isUnramified_compl, Finset.card_add_card_compl]
  /-
    🎉 no goals
  -/


/-- A field extension is unramified at infinite places if every infinite place is unramified. -/
class IsUnramifiedAtInfinitePlaces : Prop where
  isUnramified : ∀ w : InfinitePlace K, w.IsUnramified k


instance IsUnramifiedAtInfinitePlaces.id : IsUnramifiedAtInfinitePlaces K K where
  isUnramified w := w.isUnramified_self


lemma IsUnramifiedAtInfinitePlaces.trans
    [h₁ : IsUnramifiedAtInfinitePlaces k K] [h₂ : IsUnramifiedAtInfinitePlaces K F] :
    IsUnramifiedAtInfinitePlaces k F where
  isUnramified w :=
    Eq.trans (IsScalarTower.algebraMap_eq k K F ▸ h₁.1 (w.comap (algebraMap _ _))) (h₂.1 w)


lemma IsUnramifiedAtInfinitePlaces.top [h : IsUnramifiedAtInfinitePlaces k F] :
    IsUnramifiedAtInfinitePlaces K F where
  isUnramified w := (h.1 w).of_restrictScalars K


lemma IsUnramifiedAtInfinitePlaces.bot [h₁ : IsUnramifiedAtInfinitePlaces k F]
    [Algebra.IsAlgebraic K F] :
    IsUnramifiedAtInfinitePlaces k K where
  isUnramified w := by
    /-
      k : Type u_1
      inst✝⁷ : Field k
      K : Type u_2
      inst✝⁶ : Field K
      F : Type u_3
      inst✝⁵ : Field F
      inst✝⁴ : Algebra k K
      inst✝³ : Algebra k F
      inst✝² : Algebra K F
      inst✝¹ : IsScalarTower k K F
      h₁ : IsUnramifiedAtInfinitePlaces k F
      inst✝ : Algebra.IsAlgebraic K F
      w : NumberField.InfinitePlace K
      ⊢ NumberField.InfinitePlace.IsUnramified k w
    -/
    obtain ⟨w, rfl⟩ := InfinitePlace.comap_surjective (K := F) w
    /-
      case intro
      k : Type u_1
      inst✝⁷ : Field k
      K : Type u_2
      inst✝⁶ : Field K
      F : Type u_3
      inst✝⁵ : Field F
      inst✝⁴ : Algebra k K
      inst✝³ : Algebra k F
      inst✝² : Algebra K F
      inst✝¹ : IsScalarTower k K F
      h₁ : IsUnramifiedAtInfinitePlaces k F
      inst✝ : Algebra.IsAlgebraic K F
      w : NumberField.InfinitePlace F
      ⊢ NumberField.InfinitePlace.IsUnramified k ((fun x => x.comap (algebraMap K F) …
    -/
    exact (h₁.1 w).comap K
    /-
      🎉 no goals
    -/


lemma NumberField.InfinitePlace.isUnramified [IsUnramifiedAtInfinitePlaces k K]
    (w : InfinitePlace K) : IsUnramified k w := IsUnramifiedAtInfinitePlaces.isUnramified w


lemma NumberField.InfinitePlace.isUnramifiedIn [IsUnramifiedAtInfinitePlaces k K]
    (w : InfinitePlace k) : IsUnramifiedIn K w := fun v _ ↦ v.isUnramified k


lemma IsUnramifiedAtInfinitePlaces_of_odd_card_aut [IsGalois k K] [FiniteDimensional k K]
    (h : Odd (Fintype.card <| K ≃ₐ[k] K)) : IsUnramifiedAtInfinitePlaces k K :=
  ⟨fun _ ↦ not_not.mp (Nat.not_even_iff_odd.2 h ∘ InfinitePlace.even_card_aut_of_not_isUnramified)⟩


lemma IsUnramifiedAtInfinitePlaces_of_odd_finrank [IsGalois k K]
    (h : Odd (Module.finrank k K)) : IsUnramifiedAtInfinitePlaces k K :=
  ⟨fun _ ↦ not_not.mp (Nat.not_even_iff_odd.2 h ∘ InfinitePlace.even_finrank_of_not_isUnramified)⟩


open Module in
lemma IsUnramifiedAtInfinitePlaces.card_infinitePlace [NumberField k] [NumberField K]
    [IsGalois k K] [IsUnramifiedAtInfinitePlaces k K] :
    Fintype.card (InfinitePlace K) = Fintype.card (InfinitePlace k) * finrank k K := by
  rw [InfinitePlace.card_eq_card_isUnramifiedIn (k := k) (K := K), Finset.filter_true_of_mem,
    Finset.card_univ, Finset.card_eq_zero.mpr, zero_mul, add_zero]
    /-
      k : Type u_1
      inst✝⁶ : Field k
      K : Type u_2
      inst✝⁵ : Field K
      inst✝⁴ : Algebra k K
      inst✝³ : NumberField k
      inst✝² : NumberField K
      inst✝¹ : IsGalois k K
      inst✝ : IsUnramifiedAtInfinitePlaces k K
      ⊢ Eq (HasCompl.compl Finset.univ) EmptyCollection.emptyCollection
    -/
  · exact Finset.compl_univ
    /-
      🎉 no goals
    -/
  /-
    k : Type u_1
    inst✝⁶ : Field k
    K : Type u_2
    inst✝⁵ : Field K
    inst✝⁴ : Algebra k K
    inst✝³ : NumberField k
    inst✝² : NumberField K
    inst✝¹ : IsGalois k K
    inst✝ : IsUnramifiedAtInfinitePlaces k K
    ⊢ ∀ (x : NumberField.InfinitePlace k), Membership.mem Finset.univ x → NumberFi …
  -/
  simp only [Finset.mem_univ, forall_true_left, Finset.filter_eq_empty_iff]
  /-
    k : Type u_1
    inst✝⁶ : Field k
    K : Type u_2
    inst✝⁵ : Field K
    inst✝⁴ : Algebra k K
    inst✝³ : NumberField k
    inst✝² : NumberField K
    inst✝¹ : IsGalois k K
    inst✝ : IsUnramifiedAtInfinitePlaces k K
    ⊢ ∀ (x : NumberField.InfinitePlace k), NumberField.InfinitePlace.IsUnramifiedI …
  -/
  exact InfinitePlace.isUnramifiedIn K
  /-
    🎉 no goals
  -/


theorem nrRealPlaces_eq_zero_of_two_lt (hk : 2 < k) (hζ : IsPrimitiveRoot ζ k) :
    NumberField.InfinitePlace.nrRealPlaces K = 0 := by
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ζ : K
    k : Nat
    hk : LT.lt 2 k
    hζ : IsPrimitiveRoot ζ k
    ⊢ Eq (NumberField.InfinitePlace.nrRealPlaces K) 0
  -/
  refine (@Fintype.card_eq_zero_iff _ (_)).2 ⟨fun ⟨w, hwreal⟩ ↦ ?_⟩
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ζ : K
    k : Nat
    hk : LT.lt 2 k
    hζ : IsPrimitiveRoot ζ k
    x✝ : Subtype fun w => w.IsReal
    w : NumberField.InfinitePlace K
    hwreal : w.IsReal
    ⊢ False
  -/
  rw [NumberField.InfinitePlace.isReal_iff] at hwreal
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ζ : K
    k : Nat
    hk : LT.lt 2 k
    hζ : IsPrimitiveRoot ζ k
    x✝ : Subtype fun w => w.IsReal
    w : NumberField.InfinitePlace K
    hwreal : NumberField.ComplexEmbedding.IsReal w.embedding
    ⊢ False
  -/
  let f := w.embedding
  /-
    K : Type u_1
    inst✝¹ : Field K
    inst✝ : NumberField K
    ζ : K
    k : Nat
    hk : LT.lt 2 k
    hζ : IsPrimitiveRoot ζ k
    x✝ : Subtype fun w => w.IsReal
    w : NumberField.InfinitePlace K
    hwreal : NumberField.ComplexEmbedding.IsReal w.embedding
    f : RingHom K Complex := w.embedding
    ⊢ False
  -/
  have hζ' : IsPrimitiveRoot (f ζ) k := hζ.map_of_injective f.injective
  have him : (f ζ).im = 0 := by
    rw [← Complex.conj_eq_iff_im, ← NumberField.ComplexEmbedding.conjugate_coe_eq]
    congr
  have hre : (f ζ).re = 1 ∨ (f ζ).re = -1 := by
    rw [← Complex.abs_re_eq_abs] at him
    have := Complex.norm_eq_one_of_pow_eq_one hζ'.pow_eq_one (by omega)
    rwa [Complex.norm_eq_abs, ← him, ← abs_one, abs_eq_abs] at this
  cases hre with
  | inl hone =>
    exact hζ'.ne_one (by omega) <| Complex.ext (by simp [hone]) (by simp [him])
  | inr hnegone =>
    replace hζ' := hζ'.eq_orderOf
    simp only [show f ζ = -1 from Complex.ext (by simp [hnegone]) (by simp [him]),
      orderOf_neg_one, ringChar.eq_zero, OfNat.zero_ne_ofNat, ↓reduceIte] at hζ'
    omega


/-- The infinite place of `ℚ`, coming from the canonical map `ℚ → ℂ`. -/
noncomputable def infinitePlace : InfinitePlace ℚ := .mk (Rat.castHom _)


@[simp]
lemma infinitePlace_apply (v : InfinitePlace ℚ) (x : ℚ) : v x = |x| := by
  /-
    v : NumberField.InfinitePlace Rat
    x : Rat
    ⊢ Eq (v x) ↑(abs x)
  -/
  rw [NumberField.InfinitePlace.coe_apply]
  /-
    v : NumberField.InfinitePlace Rat
    x : Rat
    ⊢ Eq (↑v x) ↑(abs x)
  -/
  obtain ⟨_, _, rfl⟩ := v
  /-
    case mk.intro
    x : Rat
    w✝ : RingHom Rat Complex
    ⊢ Eq (↑⟨NumberField.place w✝, ⋯⟩ x) ↑(abs x)
  -/
  simp
  /-
    🎉 no goals
  -/


instance : Subsingleton (InfinitePlace ℚ) where
                  /-
                    a b : NumberField.InfinitePlace Rat
                    ⊢ Eq a b
                  -/
  allEq a b := by ext; simp
                       /-
                         🎉 no goals
                       -/


