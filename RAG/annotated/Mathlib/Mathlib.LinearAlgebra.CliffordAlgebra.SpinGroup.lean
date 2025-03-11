/-- `lipschitzGroup` is the subgroup closure of all the invertible elements in the form of `ι Q m`
where `ι` is the canonical linear map `M →ₗ[R] CliffordAlgebra Q`. -/
def lipschitzGroup (Q : QuadraticForm R M) : Subgroup (CliffordAlgebra Q)ˣ :=
  Subgroup.closure ((↑) ⁻¹' Set.range (ι Q) : Set (CliffordAlgebra Q)ˣ)


/-- The conjugation action by elements of the Lipschitz group keeps vectors as vectors. -/
theorem conjAct_smul_ι_mem_range_ι {x : (CliffordAlgebra Q)ˣ} (hx : x ∈ lipschitzGroup Q)
    [Invertible (2 : R)] (m : M) :
    ConjAct.toConjAct x • ι Q m ∈ LinearMap.range (ι Q) := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    Q : QuadraticForm R M
    x : Units (CliffordAlgebra Q)
    hx : Membership.mem (lipschitzGroup Q) x
    inst✝ : Invertible 2
    m : M
    ⊢ Membership.mem (LinearMap.range (CliffordAlgebra.ι Q)) (HSMul.hSMul (ConjAct …
  -/
  unfold lipschitzGroup at hx
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    Q : QuadraticForm R M
    x : Units (CliffordAlgebra Q)
    hx : Membership.mem (Subgroup.closure (Set.preimage Units.val (Set.range ⇑(Cli …
    inst✝ : Invertible 2
    m : M
    ⊢ Membership.mem (LinearMap.range (CliffordAlgebra.ι Q)) (HSMul.hSMul (ConjAct …
  -/
  rw [ConjAct.units_smul_def, ConjAct.ofConjAct_toConjAct]
  induction hx using Subgroup.closure_induction'' generalizing m with
  | mem x hx =>
    obtain ⟨a, ha⟩ := hx
    letI := x.invertible
    letI : Invertible (ι Q a) := by rwa [ha]
    letI : Invertible (Q a) := invertibleOfInvertibleι Q a
    simp_rw [← invOf_units x, ← ha, ι_mul_ι_mul_invOf_ι, LinearMap.mem_range_self]
  | inv_mem x hx =>
    obtain ⟨a, ha⟩ := hx
    letI := x.invertible
    letI : Invertible (ι Q a) := by rwa [ha]
    letI : Invertible (Q a) := invertibleOfInvertibleι Q a
    letI := invertibleNeg (ι Q a)
    letI := Invertible.map involute (ι Q a)
    simp_rw [← invOf_units x, inv_inv, ← ha, invOf_ι_mul_ι_mul_ι, LinearMap.mem_range_self]
  | one => simp_rw [inv_one, Units.val_one, one_mul, mul_one, LinearMap.mem_range_self]
  | mul y z _ _ hy hz =>
    simp_rw [mul_inv_rev, Units.val_mul]
    suffices ↑y * (↑z * ι Q m * ↑z⁻¹) * ↑y⁻¹ ∈ _ by
      simpa only [mul_assoc] using this
    obtain ⟨z', hz'⟩ := hz m
    obtain ⟨y', hy'⟩ := hy z'
    simp_rw [← hz', ← hy', LinearMap.mem_range_self]


/-- This is another version of `lipschitzGroup.conjAct_smul_ι_mem_range_ι` which uses `involute`. -/
theorem involute_act_ι_mem_range_ι [Invertible (2 : R)]
    {x : (CliffordAlgebra Q)ˣ} (hx : x ∈ lipschitzGroup Q) (b : M) :
      involute (Q := Q) ↑x * ι Q b * ↑x⁻¹ ∈ LinearMap.range (ι Q) := by
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    Q : QuadraticForm R M
    inst✝ : Invertible 2
    x : Units (CliffordAlgebra Q)
    hx : Membership.mem (lipschitzGroup Q) x
    b : M
    ⊢ Membership.mem (LinearMap.range (CliffordAlgebra.ι Q)) (HMul.hMul (HMul.hMul …
  -/
  unfold lipschitzGroup at hx
  induction hx using Subgroup.closure_induction'' generalizing b with
  | mem x hx =>
    obtain ⟨a, ha⟩ := hx
    letI := x.invertible
    letI : Invertible (ι Q a) := by rwa [ha]
    letI : Invertible (Q a) := invertibleOfInvertibleι Q a
    simp_rw [← invOf_units x, ← ha, involute_ι, neg_mul, ι_mul_ι_mul_invOf_ι Q a b, ← map_neg,
      LinearMap.mem_range_self]
  | inv_mem x hx =>
    obtain ⟨a, ha⟩ := hx
    letI := x.invertible
    letI : Invertible (ι Q a) := by rwa [ha]
    letI : Invertible (Q a) := invertibleOfInvertibleι Q a
    letI := invertibleNeg (ι Q a)
    letI := Invertible.map involute (ι Q a)
    simp_rw [← invOf_units x, inv_inv, ← ha, map_invOf, involute_ι, invOf_neg, neg_mul,
      invOf_ι_mul_ι_mul_ι, ← map_neg, LinearMap.mem_range_self]
  | one => simp_rw [inv_one, Units.val_one, map_one, one_mul, mul_one, LinearMap.mem_range_self]
  | mul y z _ _ hy hz =>
    simp_rw [mul_inv_rev, Units.val_mul, map_mul]
    suffices involute (Q := Q) ↑y * (involute (Q := Q) ↑z * ι Q b * ↑z⁻¹) * ↑y⁻¹ ∈ _ by
      simpa only [mul_assoc] using this
    obtain ⟨z', hz'⟩ := hz b
    obtain ⟨y', hy'⟩ := hy z'
    simp_rw [← hz', ← hy', LinearMap.mem_range_self]


/-- If x is in `lipschitzGroup Q`, then `(ι Q).range` is closed under twisted conjugation.
The reverse statement presumably is true only in finite dimensions.-/
theorem conjAct_smul_range_ι {x : (CliffordAlgebra Q)ˣ} (hx : x ∈ lipschitzGroup Q)
    [Invertible (2 : R)] :
    ConjAct.toConjAct x • LinearMap.range (ι Q) = LinearMap.range (ι Q) := by
  suffices ∀ x ∈ lipschitzGroup Q,
      ConjAct.toConjAct x • LinearMap.range (ι Q) ≤ LinearMap.range (ι Q) by
    apply le_antisymm
    · exact this _ hx
    · have := smul_mono_right (ConjAct.toConjAct x) <| this _ (inv_mem hx)
      refine Eq.trans_le ?_ this
      simp only [map_inv, smul_inv_smul]
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    Q : QuadraticForm R M
    x : Units (CliffordAlgebra Q)
    hx : Membership.mem (lipschitzGroup Q) x
    inst✝ : Invertible 2
    ⊢ ∀ (x : Units (CliffordAlgebra Q)), Membership.mem (lipschitzGroup Q) x → LE. …
  -/
  intro x hx
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    Q : QuadraticForm R M
    x✝ : Units (CliffordAlgebra Q)
    hx✝ : Membership.mem (lipschitzGroup Q) x✝
    inst✝ : Invertible 2
    x : Units (CliffordAlgebra Q)
    hx : Membership.mem (lipschitzGroup Q) x
    ⊢ LE.le (HSMul.hSMul (ConjAct.toConjAct x) (LinearMap.range (CliffordAlgebra.ι …
  -/
  erw [Submodule.map_le_iff_le_comap]
  /-
    R : Type u_1
    inst✝³ : CommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    Q : QuadraticForm R M
    x✝ : Units (CliffordAlgebra Q)
    hx✝ : Membership.mem (lipschitzGroup Q) x✝
    inst✝ : Invertible 2
    x : Units (CliffordAlgebra Q)
    hx : Membership.mem (lipschitzGroup Q) x
    ⊢ LE.le (LinearMap.range (CliffordAlgebra.ι Q)) (Submodule.comap (DistribMulAc …
  -/
  rintro _ ⟨m, rfl⟩
  /-
    case intro
    R : Type u_1
    inst✝³ : CommRing R
    M : Type u_2
    inst✝² : AddCommGroup M
    inst✝¹ : Module R M
    Q : QuadraticForm R M
    x✝ : Units (CliffordAlgebra Q)
    hx✝ : Membership.mem (lipschitzGroup Q) x✝
    inst✝ : Invertible 2
    x : Units (CliffordAlgebra Q)
    hx : Membership.mem (lipschitzGroup Q) x
    m : M
    ⊢ Membership.mem (Submodule.comap (DistribMulAction.toLinearMap R (CliffordAlg …
  -/
  exact conjAct_smul_ι_mem_range_ι hx _
  /-
    🎉 no goals
  -/


theorem coe_mem_iff_mem {x : (CliffordAlgebra Q)ˣ} :
    ↑x ∈ (lipschitzGroup Q).toSubmonoid.map (Units.coeHom <| CliffordAlgebra Q) ↔
    x ∈ lipschitzGroup Q := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : Units (CliffordAlgebra Q)
    ⊢ Iff (Membership.mem (Submonoid.map (Units.coeHom (CliffordAlgebra Q)) (lipsc …
  -/
  simp only [Submonoid.mem_map, Subgroup.mem_toSubmonoid, Units.coeHom_apply, exists_prop]
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : Units (CliffordAlgebra Q)
    ⊢ Iff (Exists fun x_1 => And (Membership.mem (lipschitzGroup Q) x_1) (Eq ↑x_1  …
  -/
  norm_cast
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : Units (CliffordAlgebra Q)
    ⊢ Iff (Exists fun x_1 => And (Membership.mem (lipschitzGroup Q) x_1) (Eq x_1 x …
  -/
  exact exists_eq_right
  /-
    🎉 no goals
  -/


/-- `pinGroup Q` is defined as the infimum of `lipschitzGroup Q` and `unitary (CliffordAlgebra Q)`.
See `mem_iff`. -/
def pinGroup (Q : QuadraticForm R M) : Submonoid (CliffordAlgebra Q) :=
  (lipschitzGroup Q).toSubmonoid.map (Units.coeHom <| CliffordAlgebra Q) ⊓ unitary _


/-- An element is in `pinGroup Q` if and only if it is in `lipschitzGroup Q` and `unitary`. -/
theorem mem_iff {x : CliffordAlgebra Q} :
    x ∈ pinGroup Q ↔
      x ∈ (lipschitzGroup Q).toSubmonoid.map (Units.coeHom <| CliffordAlgebra Q) ∧
        x ∈ unitary (CliffordAlgebra Q) :=
  Iff.rfl


theorem mem_lipschitzGroup {x : CliffordAlgebra Q} (hx : x ∈ pinGroup Q) :
    x ∈ (lipschitzGroup Q).toSubmonoid.map (Units.coeHom <| CliffordAlgebra Q) :=
  hx.1


theorem mem_unitary {x : CliffordAlgebra Q} (hx : x ∈ pinGroup Q) :
    x ∈ unitary (CliffordAlgebra Q) :=
  hx.2


theorem units_mem_iff {x : (CliffordAlgebra Q)ˣ} :
    ↑x ∈ pinGroup Q ↔ x ∈ lipschitzGroup Q ∧ ↑x ∈ unitary (CliffordAlgebra Q) := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : Units (CliffordAlgebra Q)
    ⊢ Iff (Membership.mem (pinGroup Q) ↑x) (And (Membership.mem (lipschitzGroup Q) …
  -/
  rw [mem_iff, lipschitzGroup.coe_mem_iff_mem]
  /-
    🎉 no goals
  -/


theorem units_mem_lipschitzGroup {x : (CliffordAlgebra Q)ˣ} (hx : ↑x ∈ pinGroup Q) :
    x ∈ lipschitzGroup Q :=
  (units_mem_iff.1 hx).1


/-- The conjugation action by elements of the spin group keeps vectors as vectors. -/
theorem conjAct_smul_ι_mem_range_ι {x : (CliffordAlgebra Q)ˣ} (hx : ↑x ∈ pinGroup Q)
    [Invertible (2 : R)] (y : M) : ConjAct.toConjAct x • ι Q y ∈ LinearMap.range (ι Q) :=
  lipschitzGroup.conjAct_smul_ι_mem_range_ι (units_mem_lipschitzGroup hx) y


/-- This is another version of `conjAct_smul_ι_mem_range_ι` which uses `involute`. -/
theorem involute_act_ι_mem_range_ι {x : (CliffordAlgebra Q)ˣ} (hx : ↑x ∈ pinGroup Q)
    [Invertible (2 : R)] (y : M) : involute (Q := Q) ↑x * ι Q y * ↑x⁻¹ ∈ LinearMap.range (ι Q) :=
  lipschitzGroup.involute_act_ι_mem_range_ι (units_mem_lipschitzGroup hx) y


/-- If x is in `pinGroup Q`, then `(ι Q).range` is closed under twisted conjugation. The reverse
statement presumably being true only in finite dimensions.-/
theorem conjAct_smul_range_ι {x : (CliffordAlgebra Q)ˣ} (hx : ↑x ∈ pinGroup Q)
    [Invertible (2 : R)] : ConjAct.toConjAct x • LinearMap.range (ι Q) = LinearMap.range (ι Q) :=
  lipschitzGroup.conjAct_smul_range_ι (units_mem_lipschitzGroup hx)


@[simp]
theorem star_mul_self_of_mem {x : CliffordAlgebra Q} (hx : x ∈ pinGroup Q) : star x * x = 1 :=
  hx.2.1


@[simp]
theorem mul_star_self_of_mem {x : CliffordAlgebra Q} (hx : x ∈ pinGroup Q) : x * star x = 1 :=
  hx.2.2


/-- See `star_mem_iff` for both directions. -/
theorem star_mem {x : CliffordAlgebra Q} (hx : x ∈ pinGroup Q) : star x ∈ pinGroup Q := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : CliffordAlgebra Q
    hx : Membership.mem (pinGroup Q) x
    ⊢ Membership.mem (pinGroup Q) (Star.star x)
  -/
  rw [mem_iff] at hx ⊢
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : CliffordAlgebra Q
    hx : And (Membership.mem (Submonoid.map (Units.coeHom (CliffordAlgebra Q)) (li …
    ⊢ And (Membership.mem (Submonoid.map (Units.coeHom (CliffordAlgebra Q)) (lipsc …
  -/
  refine ⟨?_, unitary.star_mem hx.2⟩
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : CliffordAlgebra Q
    hx : And (Membership.mem (Submonoid.map (Units.coeHom (CliffordAlgebra Q)) (li …
    ⊢ Membership.mem (Submonoid.map (Units.coeHom (CliffordAlgebra Q)) (lipschitzG …
  -/
  rcases hx with ⟨⟨y, hy₁, hy₂⟩, _hx₂, hx₃⟩
  /-
    case intro.intro.intro.intro
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : CliffordAlgebra Q
    y : Units (CliffordAlgebra Q)
    hy₁ : Membership.mem (↑(lipschitzGroup Q).toSubmonoid) y
    hy₂ : Eq ((Units.coeHom (CliffordAlgebra Q)) y) x
    _hx₂ : Eq (HMul.hMul (Star.star x) x) 1
    hx₃ : Eq (HMul.hMul x (Star.star x)) 1
    ⊢ Membership.mem (Submonoid.map (Units.coeHom (CliffordAlgebra Q)) (lipschitzG …
  -/
  simp only [Subgroup.coe_toSubmonoid, SetLike.mem_coe] at hy₁
  /-
    case intro.intro.intro.intro
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : CliffordAlgebra Q
    y : Units (CliffordAlgebra Q)
    hy₂ : Eq ((Units.coeHom (CliffordAlgebra Q)) y) x
    _hx₂ : Eq (HMul.hMul (Star.star x) x) 1
    hx₃ : Eq (HMul.hMul x (Star.star x)) 1
    hy₁ : Membership.mem (lipschitzGroup Q) y
    ⊢ Membership.mem (Submonoid.map (Units.coeHom (CliffordAlgebra Q)) (lipschitzG …
  -/
  simp only [Units.coeHom_apply] at hy₂
  /-
    case intro.intro.intro.intro
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : CliffordAlgebra Q
    y : Units (CliffordAlgebra Q)
    hy₂ : Eq (↑y) x
    _hx₂ : Eq (HMul.hMul (Star.star x) x) 1
    hx₃ : Eq (HMul.hMul x (Star.star x)) 1
    hy₁ : Membership.mem (lipschitzGroup Q) y
    ⊢ Membership.mem (Submonoid.map (Units.coeHom (CliffordAlgebra Q)) (lipschitzG …
  -/
  simp only [Submonoid.mem_map, Subgroup.mem_toSubmonoid, Units.coeHom_apply, exists_prop]
  /-
    case intro.intro.intro.intro
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : CliffordAlgebra Q
    y : Units (CliffordAlgebra Q)
    hy₂ : Eq (↑y) x
    _hx₂ : Eq (HMul.hMul (Star.star x) x) 1
    hx₃ : Eq (HMul.hMul x (Star.star x)) 1
    hy₁ : Membership.mem (lipschitzGroup Q) y
    ⊢ Exists fun x_1 => And (Membership.mem (lipschitzGroup Q) x_1) (Eq (↑x_1) (St …
  -/
  refine ⟨star y, ?_, by simp only [hy₂, Units.coe_star]⟩
  /-
    case intro.intro.intro.intro
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : CliffordAlgebra Q
    y : Units (CliffordAlgebra Q)
    hy₂ : Eq (↑y) x
    _hx₂ : Eq (HMul.hMul (Star.star x) x) 1
    hx₃ : Eq (HMul.hMul x (Star.star x)) 1
    hy₁ : Membership.mem (lipschitzGroup Q) y
    ⊢ Membership.mem (lipschitzGroup Q) (Star.star y)
  -/
  rw [← hy₂] at hx₃
  have hy₃ : y * star y = 1 := by
    rw [← Units.eq_iff]
    simp only [hx₃, Units.val_mul, Units.coe_star, Units.val_one]
  /-
    case intro.intro.intro.intro
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : CliffordAlgebra Q
    y : Units (CliffordAlgebra Q)
    hy₂ : Eq (↑y) x
    _hx₂ : Eq (HMul.hMul (Star.star x) x) 1
    hx₃ : Eq (HMul.hMul (↑y) (Star.star ↑y)) 1
    hy₁ : Membership.mem (lipschitzGroup Q) y
    hy₃ : Eq (HMul.hMul y (Star.star y)) 1
    ⊢ Membership.mem (lipschitzGroup Q) (Star.star y)
  -/
  apply_fun fun x => y⁻¹ * x at hy₃
  /-
    case intro.intro.intro.intro
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : CliffordAlgebra Q
    y : Units (CliffordAlgebra Q)
    hy₂ : Eq (↑y) x
    _hx₂ : Eq (HMul.hMul (Star.star x) x) 1
    hx₃ : Eq (HMul.hMul (↑y) (Star.star ↑y)) 1
    hy₁ : Membership.mem (lipschitzGroup Q) y
    hy₃ : Eq (HMul.hMul (Inv.inv y) (HMul.hMul y (Star.star y))) (HMul.hMul (Inv.i …
    ⊢ Membership.mem (lipschitzGroup Q) (Star.star y)
  -/
  simp only [inv_mul_cancel_left, mul_one] at hy₃
  /-
    case intro.intro.intro.intro
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : CliffordAlgebra Q
    y : Units (CliffordAlgebra Q)
    hy₂ : Eq (↑y) x
    _hx₂ : Eq (HMul.hMul (Star.star x) x) 1
    hx₃ : Eq (HMul.hMul (↑y) (Star.star ↑y)) 1
    hy₁ : Membership.mem (lipschitzGroup Q) y
    hy₃ : Eq (Star.star y) (Inv.inv y)
    ⊢ Membership.mem (lipschitzGroup Q) (Star.star y)
  -/
  simp only [hy₃, hy₁, inv_mem_iff]
  /-
    🎉 no goals
  -/


/-- An element is in `pinGroup Q` if and only if `star x` is in `pinGroup Q`.
See `star_mem` for only one direction. -/
@[simp]
theorem star_mem_iff {x : CliffordAlgebra Q} : star x ∈ pinGroup Q ↔ x ∈ pinGroup Q := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : CliffordAlgebra Q
    ⊢ Iff (Membership.mem (pinGroup Q) (Star.star x)) (Membership.mem (pinGroup Q) …
  -/
  refine ⟨?_, star_mem⟩
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : CliffordAlgebra Q
    ⊢ Membership.mem (pinGroup Q) (Star.star x) → Membership.mem (pinGroup Q) x
  -/
  intro hx
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : CliffordAlgebra Q
    hx : Membership.mem (pinGroup Q) (Star.star x)
    ⊢ Membership.mem (pinGroup Q) x
  -/
  convert star_mem hx
  /-
    case h.e'_5
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : CliffordAlgebra Q
    hx : Membership.mem (pinGroup Q) (Star.star x)
    ⊢ Eq x (Star.star (Star.star x))
  -/
  exact (star_star x).symm
  /-
    🎉 no goals
  -/


instance : Star (pinGroup Q) where
  star x := ⟨star x, star_mem x.prop⟩


@[simp, norm_cast]
theorem coe_star {x : pinGroup Q} : ↑(star x) = (star x : CliffordAlgebra Q) :=
  rfl


theorem coe_star_mul_self (x : pinGroup Q) : (star x : CliffordAlgebra Q) * x = 1 :=
  star_mul_self_of_mem x.prop


theorem coe_mul_star_self (x : pinGroup Q) : (x : CliffordAlgebra Q) * star x = 1 :=
  mul_star_self_of_mem x.prop


@[simp]
theorem star_mul_self (x : pinGroup Q) : star x * x = 1 :=
  Subtype.ext <| coe_star_mul_self x


@[simp]
theorem mul_star_self (x : pinGroup Q) : x * star x = 1 :=
  Subtype.ext <| coe_mul_star_self x


/-- `pinGroup Q` forms a group where the inverse is `star`. -/
instance : Group (pinGroup Q) where
  inv := star
  inv_mul_cancel := star_mul_self


instance : StarMul (pinGroup Q) where
  star_involutive _ := Subtype.ext <| star_involutive _
  star_mul _ _ := Subtype.ext <| star_mul _ _


instance : Inhabited (pinGroup Q) :=
  ⟨1⟩


theorem star_eq_inv (x : pinGroup Q) : star x = x⁻¹ :=
  rfl


theorem star_eq_inv' : (star : pinGroup Q → pinGroup Q) = Inv.inv :=
  rfl


/-- The elements in `pinGroup Q` embed into (CliffordAlgebra Q)ˣ. -/
@[simps]
def toUnits : pinGroup Q →* (CliffordAlgebra Q)ˣ where
  toFun x := ⟨x, ↑x⁻¹, coe_mul_star_self x, coe_star_mul_self x⟩
  map_one' := Units.ext rfl
  map_mul' _x _y := Units.ext rfl


theorem toUnits_injective : Function.Injective (toUnits : pinGroup Q → (CliffordAlgebra Q)ˣ) :=
  fun _x _y h => Subtype.ext <| Units.ext_iff.mp h


/-- `spinGroup Q` is defined as the infimum of `pinGroup Q` and `CliffordAlgebra.even Q`.
See `mem_iff`. -/
def spinGroup (Q : QuadraticForm R M) : Submonoid (CliffordAlgebra Q) :=
  pinGroup Q ⊓ (CliffordAlgebra.even Q).toSubring.toSubmonoid


/-- An element is in `spinGroup Q` if and only if it is in `pinGroup Q` and `even Q`. -/
theorem mem_iff {x : CliffordAlgebra Q} : x ∈ spinGroup Q ↔ x ∈ pinGroup Q ∧ x ∈ even Q :=
  Iff.rfl


theorem mem_pin {x : CliffordAlgebra Q} (hx : x ∈ spinGroup Q) : x ∈ pinGroup Q :=
  hx.1


theorem mem_even {x : CliffordAlgebra Q} (hx : x ∈ spinGroup Q) : x ∈ even Q :=
  hx.2


theorem units_mem_lipschitzGroup {x : (CliffordAlgebra Q)ˣ} (hx : ↑x ∈ spinGroup Q) :
    x ∈ lipschitzGroup Q :=
  pinGroup.units_mem_lipschitzGroup (mem_pin hx)


/-- If x is in `spinGroup Q`, then `involute x` is equal to x.-/
theorem involute_eq {x : CliffordAlgebra Q} (hx : x ∈ spinGroup Q) : involute x = x :=
  involute_eq_of_mem_even (mem_even hx)


theorem units_involute_act_eq_conjAct {x : (CliffordAlgebra Q)ˣ} (hx : ↑x ∈ spinGroup Q) (y : M) :
    involute (Q := Q) ↑x * ι Q y * ↑x⁻¹ = ConjAct.toConjAct x • (ι Q y) := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : Units (CliffordAlgebra Q)
    hx : Membership.mem (spinGroup Q) ↑x
    y : M
    ⊢ Eq (HMul.hMul (HMul.hMul (CliffordAlgebra.involute ↑x) ((CliffordAlgebra.ι Q …
  -/
  rw [involute_eq hx, @ConjAct.units_smul_def, @ConjAct.ofConjAct_toConjAct]
  /-
    🎉 no goals
  -/


/-- The conjugation action by elements of the spin group keeps vectors as vectors. -/
theorem conjAct_smul_ι_mem_range_ι {x : (CliffordAlgebra Q)ˣ} (hx : ↑x ∈ spinGroup Q)
    [Invertible (2 : R)] (y : M) : ConjAct.toConjAct x • ι Q y ∈ LinearMap.range (ι Q) :=
  lipschitzGroup.conjAct_smul_ι_mem_range_ι (units_mem_lipschitzGroup hx) y

/- This is another version of `conjAct_smul_ι_mem_range_ι` which uses `involute`.-/

theorem involute_act_ι_mem_range_ι {x : (CliffordAlgebra Q)ˣ} (hx : ↑x ∈ spinGroup Q)
    [Invertible (2 : R)] (y : M) : involute (Q := Q) ↑x * ι Q y * ↑x⁻¹ ∈ LinearMap.range (ι Q) :=
  lipschitzGroup.involute_act_ι_mem_range_ι (units_mem_lipschitzGroup hx) y

/- If x is in `spinGroup Q`, then `(ι Q).range` is closed under twisted conjugation. The reverse
statement presumably being true only in finite dimensions.-/

theorem conjAct_smul_range_ι {x : (CliffordAlgebra Q)ˣ} (hx : ↑x ∈ spinGroup Q)
    [Invertible (2 : R)] : ConjAct.toConjAct x • LinearMap.range (ι Q) = LinearMap.range (ι Q) :=
  lipschitzGroup.conjAct_smul_range_ι (units_mem_lipschitzGroup hx)


@[simp]
theorem star_mul_self_of_mem {x : CliffordAlgebra Q} (hx : x ∈ spinGroup Q) : star x * x = 1 :=
  hx.1.2.1


@[simp]
theorem mul_star_self_of_mem {x : CliffordAlgebra Q} (hx : x ∈ spinGroup Q) : x * star x = 1 :=
  hx.1.2.2


/-- See `star_mem_iff` for both directions. -/
theorem star_mem {x : CliffordAlgebra Q} (hx : x ∈ spinGroup Q) : star x ∈ spinGroup Q := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : CliffordAlgebra Q
    hx : Membership.mem (spinGroup Q) x
    ⊢ Membership.mem (spinGroup Q) (Star.star x)
  -/
  rw [mem_iff] at hx ⊢
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : CliffordAlgebra Q
    hx : And (Membership.mem (pinGroup Q) x) (Membership.mem (CliffordAlgebra.even …
    ⊢ And (Membership.mem (pinGroup Q) (Star.star x)) (Membership.mem (CliffordAlg …
  -/
  cases' hx with hx₁ hx₂
  /-
    case intro
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : CliffordAlgebra Q
    hx₁ : Membership.mem (pinGroup Q) x
    hx₂ : Membership.mem (CliffordAlgebra.even Q) x
    ⊢ And (Membership.mem (pinGroup Q) (Star.star x)) (Membership.mem (CliffordAlg …
  -/
  refine ⟨pinGroup.star_mem hx₁, ?_⟩
  /-
    case intro
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : CliffordAlgebra Q
    hx₁ : Membership.mem (pinGroup Q) x
    hx₂ : Membership.mem (CliffordAlgebra.even Q) x
    ⊢ Membership.mem (CliffordAlgebra.even Q) (Star.star x)
  -/
  dsimp only [CliffordAlgebra.even] at hx₂ ⊢
  /-
    case intro
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : CliffordAlgebra Q
    hx₁ : Membership.mem (pinGroup Q) x
    hx₂ : Membership.mem ((CliffordAlgebra.evenOdd Q 0).toSubalgebra ⋯ ⋯) x
    ⊢ Membership.mem ((CliffordAlgebra.evenOdd Q 0).toSubalgebra ⋯ ⋯) (Star.star x)
  -/
  simp only [Submodule.mem_toSubalgebra] at hx₂ ⊢
  /-
    case intro
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : CliffordAlgebra Q
    hx₁ : Membership.mem (pinGroup Q) x
    hx₂ : Membership.mem (CliffordAlgebra.evenOdd Q 0) x
    ⊢ Membership.mem (CliffordAlgebra.evenOdd Q 0) (Star.star x)
  -/
  simp only [star_def, reverse_mem_evenOdd_iff, involute_mem_evenOdd_iff, hx₂]
  /-
    🎉 no goals
  -/


/-- An element is in `spinGroup Q` if and only if `star x` is in `spinGroup Q`.
See `star_mem` for only one direction.
-/
@[simp]
theorem star_mem_iff {x : CliffordAlgebra Q} : star x ∈ spinGroup Q ↔ x ∈ spinGroup Q := by
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : CliffordAlgebra Q
    ⊢ Iff (Membership.mem (spinGroup Q) (Star.star x)) (Membership.mem (spinGroup  …
  -/
  refine ⟨?_, star_mem⟩
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : CliffordAlgebra Q
    ⊢ Membership.mem (spinGroup Q) (Star.star x) → Membership.mem (spinGroup Q) x
  -/
  intro hx
  /-
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : CliffordAlgebra Q
    hx : Membership.mem (spinGroup Q) (Star.star x)
    ⊢ Membership.mem (spinGroup Q) x
  -/
  convert star_mem hx
  /-
    case h.e'_5
    R : Type u_1
    inst✝² : CommRing R
    M : Type u_2
    inst✝¹ : AddCommGroup M
    inst✝ : Module R M
    Q : QuadraticForm R M
    x : CliffordAlgebra Q
    hx : Membership.mem (spinGroup Q) (Star.star x)
    ⊢ Eq x (Star.star (Star.star x))
  -/
  exact (star_star x).symm
  /-
    🎉 no goals
  -/


instance : Star (spinGroup Q) where
  star x := ⟨star x, star_mem x.prop⟩


@[simp, norm_cast]
theorem coe_star {x : spinGroup Q} : ↑(star x) = (star x : CliffordAlgebra Q) :=
  rfl


theorem coe_star_mul_self (x : spinGroup Q) : (star x : CliffordAlgebra Q) * x = 1 :=
  star_mul_self_of_mem x.prop


theorem coe_mul_star_self (x : spinGroup Q) : (x : CliffordAlgebra Q) * star x = 1 :=
  mul_star_self_of_mem x.prop


@[simp]
theorem star_mul_self (x : spinGroup Q) : star x * x = 1 :=
  Subtype.ext <| coe_star_mul_self x


@[simp]
theorem mul_star_self (x : spinGroup Q) : x * star x = 1 :=
  Subtype.ext <| coe_mul_star_self x


/-- `spinGroup Q` forms a group where the inverse is `star`. -/
instance : Group (spinGroup Q) where
  inv := star
  inv_mul_cancel := star_mul_self


instance : StarMul (spinGroup Q) where
  star_involutive _ := Subtype.ext <| star_involutive _
  star_mul _ _ := Subtype.ext <| star_mul _ _


instance : Inhabited (spinGroup Q) :=
  ⟨1⟩


theorem star_eq_inv (x : spinGroup Q) : star x = x⁻¹ :=
  rfl


theorem star_eq_inv' : (star : spinGroup Q → spinGroup Q) = Inv.inv :=
  rfl


/-- The elements in `spinGroup Q` embed into (CliffordAlgebra Q)ˣ. -/
@[simps]
def toUnits : spinGroup Q →* (CliffordAlgebra Q)ˣ where
  toFun x := ⟨x, ↑x⁻¹, coe_mul_star_self x, coe_star_mul_self x⟩
  map_one' := Units.ext rfl
  map_mul' _x _y := Units.ext rfl


theorem toUnits_injective : Function.Injective (toUnits : spinGroup Q → (CliffordAlgebra Q)ˣ) :=
  fun _x _y h => Subtype.ext <| Units.ext_iff.mp h


