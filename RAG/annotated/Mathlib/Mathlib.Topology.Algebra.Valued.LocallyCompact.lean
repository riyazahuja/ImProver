@[simp]
lemma NormedField.v_eq_valuation (x : K) : Valued.v x = NormedField.valuation x := rfl


/-- An element is in the valuation ring if the norm is bounded by 1. This is a variant of
`Valuation.mem_integer_iff`, phrased using norms instead of the valuation. -/
lemma mem_iff {x : K} : x ∈ 𝒪[K] ↔ ‖x‖ ≤ 1 := by
  /-
    K : Type u_1
    inst✝¹ : NontriviallyNormedField K
    inst✝ : IsUltrametricDist K
    x : K
    ⊢ Iff (Membership.mem (Valued.integer K) x) (LE.le (Norm.norm x) 1)
  -/
  simp [Valuation.mem_integer_iff, ← NNReal.coe_le_coe]
  /-
    🎉 no goals
  -/


lemma norm_le_one (x : 𝒪[K]) : ‖x‖ ≤ 1 := mem_iff.mp x.prop


@[simp]
lemma norm_coe_unit (u : 𝒪[K]ˣ) : ‖((u : 𝒪[K]) : K)‖ = 1 := by
  simpa [← NNReal.coe_inj] using
    (Valuation.integer.integers (NormedField.valuation (K := K))).valuation_unit u


lemma norm_unit (u : 𝒪[K]ˣ) : ‖(u : 𝒪[K])‖ = 1 := by
  /-
    K : Type u_1
    inst✝¹ : NontriviallyNormedField K
    inst✝ : IsUltrametricDist K
    u : Units (Subtype fun x => Membership.mem (Valued.integer K) x)
    ⊢ Eq (Norm.norm ↑u) 1
  -/
  simp
  /-
    🎉 no goals
  -/


lemma isUnit_iff_norm_eq_one {u : 𝒪[K]} : IsUnit u ↔ ‖u‖ = 1 := by
  simpa [← NNReal.coe_inj] using
    (Valuation.integer.integers (NormedField.valuation (K := K))).isUnit_iff_valuation_eq_one


lemma norm_irreducible_lt_one {ϖ : 𝒪[K]} (h : Irreducible ϖ) : ‖ϖ‖ < 1 :=
  lt_of_le_of_ne (norm_le_one ϖ) (mt isUnit_iff_norm_eq_one.mpr h.not_unit)


lemma norm_irreducible_pos {ϖ : 𝒪[K]} (h : Irreducible ϖ) : 0 < ‖ϖ‖ :=
                                            /-
                                              K : Type u_1
                                              inst✝¹ : NontriviallyNormedField K
                                              inst✝ : IsUltrametricDist K
                                              ϖ : Subtype fun x => Membership.mem (Valued.integer K) x
                                              h : Irreducible ϖ
                                              ⊢ Ne 0 (Norm.norm ϖ)
                                            -/
  lt_of_le_of_ne (_root_.norm_nonneg ϖ) (by simp [eq_comm, h.ne_zero])
                                            /-
                                              🎉 no goals
                                            -/


lemma coe_span_singleton_eq_closedBall (x : 𝒪[K]) :
    (Ideal.span {x} : Set 𝒪[K]) = Metric.closedBall 0 ‖x‖ := by
  /-
    K : Type u_1
    inst✝¹ : NontriviallyNormedField K
    inst✝ : IsUltrametricDist K
    x : Subtype fun x => Membership.mem (Valued.integer K) x
    ⊢ Eq (↑(Ideal.span (Singleton.singleton x))) (Metric.closedBall 0 (Norm.norm x))
  -/
  rcases eq_or_ne x 0 with rfl|hx
    /-
      case inl
      K : Type u_1
      inst✝¹ : NontriviallyNormedField K
      inst✝ : IsUltrametricDist K
      ⊢ Eq (↑(Ideal.span (Singleton.singleton 0))) (Metric.closedBall 0 (Norm.norm 0))
    -/
  · simp [Set.singleton_zero, Ideal.span_zero]
    /-
      🎉 no goals
    -/
  /-
    case inr
    K : Type u_1
    inst✝¹ : NontriviallyNormedField K
    inst✝ : IsUltrametricDist K
    x : Subtype fun x => Membership.mem (Valued.integer K) x
    hx : Ne x 0
    ⊢ Eq (↑(Ideal.span (Singleton.singleton x))) (Metric.closedBall 0 (Norm.norm x))
  -/
  ext y
  simp only [SetLike.mem_coe, Ideal.mem_span_singleton', AddSubgroupClass.coe_norm,
    Metric.mem_closedBall, dist_zero_right]
  /-
    case inr.h
    K : Type u_1
    inst✝¹ : NontriviallyNormedField K
    inst✝ : IsUltrametricDist K
    x : Subtype fun x => Membership.mem (Valued.integer K) x
    hx : Ne x 0
    y : Subtype fun x => Membership.mem (Valued.integer K) x
    ⊢ Iff (Exists fun a => Eq (HMul.hMul a x) y) (LE.le (Norm.norm ↑y) (Norm.norm  …
  -/
  constructor
    /-
      case inr.h.mp
      K : Type u_1
      inst✝¹ : NontriviallyNormedField K
      inst✝ : IsUltrametricDist K
      x : Subtype fun x => Membership.mem (Valued.integer K) x
      hx : Ne x 0
      y : Subtype fun x => Membership.mem (Valued.integer K) x
      ⊢ (Exists fun a => Eq (HMul.hMul a x) y) → LE.le (Norm.norm ↑y) (Norm.norm ↑x)
    -/
  · rintro ⟨z, rfl⟩
    /-
      case inr.h.mp.intro
      K : Type u_1
      inst✝¹ : NontriviallyNormedField K
      inst✝ : IsUltrametricDist K
      x : Subtype fun x => Membership.mem (Valued.integer K) x
      hx : Ne x 0
      z : Subtype fun x => Membership.mem (Valued.integer K) x
      ⊢ LE.le (Norm.norm ↑(HMul.hMul z x)) (Norm.norm ↑x)
    -/
    simpa using mul_le_mul_of_nonneg_right (norm_le_one z) (_root_.norm_nonneg x)
    /-
      🎉 no goals
    -/
    /-
      case inr.h.mpr
      K : Type u_1
      inst✝¹ : NontriviallyNormedField K
      inst✝ : IsUltrametricDist K
      x : Subtype fun x => Membership.mem (Valued.integer K) x
      hx : Ne x 0
      y : Subtype fun x => Membership.mem (Valued.integer K) x
      ⊢ LE.le (Norm.norm ↑y) (Norm.norm ↑x) → Exists fun a => Eq (HMul.hMul a x) y
    -/
  · intro h
    /-
      case inr.h.mpr
      K : Type u_1
      inst✝¹ : NontriviallyNormedField K
      inst✝ : IsUltrametricDist K
      x : Subtype fun x => Membership.mem (Valued.integer K) x
      hx : Ne x 0
      y : Subtype fun x => Membership.mem (Valued.integer K) x
      h : LE.le (Norm.norm ↑y) (Norm.norm ↑x)
      ⊢ Exists fun a => Eq (HMul.hMul a x) y
    -/
    refine ⟨⟨y / x, ?_⟩, ?_⟩
      /-
        case inr.h.mpr.refine_1
        K : Type u_1
        inst✝¹ : NontriviallyNormedField K
        inst✝ : IsUltrametricDist K
        x : Subtype fun x => Membership.mem (Valued.integer K) x
        hx : Ne x 0
        y : Subtype fun x => Membership.mem (Valued.integer K) x
        h : LE.le (Norm.norm ↑y) (Norm.norm ↑x)
        ⊢ Membership.mem (Valued.integer K) (HDiv.hDiv ↑y ↑x)
      -/
    · simpa [mem_iff] using div_le_one_of_le₀ h (_root_.norm_nonneg _)
      /-
        🎉 no goals
      -/
      /-
        case inr.h.mpr.refine_2
        K : Type u_1
        inst✝¹ : NontriviallyNormedField K
        inst✝ : IsUltrametricDist K
        x : Subtype fun x => Membership.mem (Valued.integer K) x
        hx : Ne x 0
        y : Subtype fun x => Membership.mem (Valued.integer K) x
        h : LE.le (Norm.norm ↑y) (Norm.norm ↑x)
        ⊢ Eq (HMul.hMul ⟨HDiv.hDiv ↑y ↑x, ⋯⟩ x) y
      -/
    · simpa only [Subtype.ext_iff] using div_mul_cancel₀ (y : K) (by simpa using hx)
      /-
        🎉 no goals
      -/


lemma _root_.Irreducible.maximalIdeal_eq_closedBall [IsDiscreteValuationRing 𝒪[K]]
    {ϖ : 𝒪[K]} (h : Irreducible ϖ) :
    (𝓂[K] : Set 𝒪[K]) = Metric.closedBall 0 ‖ϖ‖ := by
  /-
    K : Type u_1
    inst✝² : NontriviallyNormedField K
    inst✝¹ : IsUltrametricDist K
    inst✝ : IsDiscreteValuationRing (Subtype fun x => Membership.mem (Valued.integ …
    ϖ : Subtype fun x => Membership.mem (Valued.integer K) x
    h : Irreducible ϖ
    ⊢ Eq (↑(Valued.maximalIdeal K)) (Metric.closedBall 0 (Norm.norm ϖ))
  -/
  rw [← coe_span_singleton_eq_closedBall, ← h.maximalIdeal_eq]
  /-
    🎉 no goals
  -/


lemma _root_.Irreducible.maximalIdeal_pow_eq_closedBall_pow [IsDiscreteValuationRing 𝒪[K]]
    {ϖ : 𝒪[K]} (h : Irreducible ϖ) (n : ℕ) :
    ((𝓂[K] ^ n : Ideal 𝒪[K]) : Set 𝒪[K]) = Metric.closedBall 0 (‖ϖ‖ ^ n) := by
  /-
    K : Type u_1
    inst✝² : NontriviallyNormedField K
    inst✝¹ : IsUltrametricDist K
    inst✝ : IsDiscreteValuationRing (Subtype fun x => Membership.mem (Valued.integ …
    ϖ : Subtype fun x => Membership.mem (Valued.integer K) x
    h : Irreducible ϖ
    n : Nat
    ⊢ Eq (↑(HPow.hPow (Valued.maximalIdeal K) n)) (Metric.closedBall 0 (HPow.hPow  …
  -/
  have : ‖ϖ‖ ^ n = ‖ϖ ^ n‖ := by simp
  /-
    K : Type u_1
    inst✝² : NontriviallyNormedField K
    inst✝¹ : IsUltrametricDist K
    inst✝ : IsDiscreteValuationRing (Subtype fun x => Membership.mem (Valued.integ …
    ϖ : Subtype fun x => Membership.mem (Valued.integer K) x
    h : Irreducible ϖ
    n : Nat
    this : Eq (HPow.hPow (Norm.norm ϖ) n) (Norm.norm (HPow.hPow ϖ n))
    ⊢ Eq (↑(HPow.hPow (Valued.maximalIdeal K) n)) (Metric.closedBall 0 (HPow.hPow  …
  -/
  rw [this, ← coe_span_singleton_eq_closedBall, ← Ideal.span_singleton_pow, ← h.maximalIdeal_eq]
  /-
    🎉 no goals
  -/


lemma finite_quotient_maximalIdeal_pow_of_finite_residueField [IsDiscreteValuationRing 𝒪[K]]
    (h : Finite 𝓀[K]) (n : ℕ) :
    Finite (𝒪[K] ⧸ 𝓂[K] ^ n) := by
  induction n with
  | zero =>
    simp only [pow_zero, Ideal.one_eq_top]
    exact Finite.of_fintype (↥𝒪[K] ⧸ ⊤)
  | succ n ih =>
    have : 𝓂[K] ^ (n + 1) ≤ 𝓂[K] ^ n := Ideal.pow_le_pow_right (by simp)
    replace ih := Finite.of_equiv _ (DoubleQuot.quotQuotEquivQuotOfLE this).symm.toEquiv
    suffices Finite (Ideal.map (Ideal.Quotient.mk (𝓂[K] ^ (n + 1))) (𝓂[K] ^ n)) from
      Finite.of_finite_quot_finite_ideal
        (I := Ideal.map (Ideal.Quotient.mk _) (𝓂[K] ^ n))
    exact @Finite.of_equiv _ _ h
      ((Ideal.quotEquivPowQuotPowSuccEquiv (IsPrincipalIdealRing.principal 𝓂[K])
        (IsDiscreteValuationRing.not_a_field _) n).trans
        (Ideal.powQuotPowSuccEquivMapMkPowSuccPow _ n))


lemma totallyBounded_iff_finite_residueField [IsDiscreteValuationRing 𝒪[K]] :
    TotallyBounded (Set.univ (α := 𝒪[K])) ↔ Finite 𝓀[K] := by
  /-
    K : Type u_2
    inst✝² : NontriviallyNormedField K
    inst✝¹ : IsUltrametricDist K
    inst✝ : IsDiscreteValuationRing (Subtype fun x => Membership.mem (Valued.integ …
    ⊢ Iff (TotallyBounded Set.univ) (Finite (Valued.ResidueField K))
  -/
  constructor
    /-
      case mp
      K : Type u_2
      inst✝² : NontriviallyNormedField K
      inst✝¹ : IsUltrametricDist K
      inst✝ : IsDiscreteValuationRing (Subtype fun x => Membership.mem (Valued.integ …
      ⊢ TotallyBounded Set.univ → Finite (Valued.ResidueField K)
    -/
  · intro H
    /-
      case mp
      K : Type u_2
      inst✝² : NontriviallyNormedField K
      inst✝¹ : IsUltrametricDist K
      inst✝ : IsDiscreteValuationRing (Subtype fun x => Membership.mem (Valued.integ …
      H : TotallyBounded Set.univ
      ⊢ Finite (Valued.ResidueField K)
    -/
    obtain ⟨p, hp⟩ := IsDiscreteValuationRing.exists_irreducible 𝒪[K]
    /-
      case mp.intro
      K : Type u_2
      inst✝² : NontriviallyNormedField K
      inst✝¹ : IsUltrametricDist K
      inst✝ : IsDiscreteValuationRing (Subtype fun x => Membership.mem (Valued.integ …
      H : TotallyBounded Set.univ
      p : Subtype fun x => Membership.mem (Valued.integer K) x
      hp : Irreducible p
      ⊢ Finite (Valued.ResidueField K)
    -/
    have := Metric.finite_approx_of_totallyBounded H ‖p‖ (norm_pos_iff.mpr hp.ne_zero)
    /-
      case mp.intro
      K : Type u_2
      inst✝² : NontriviallyNormedField K
      inst✝¹ : IsUltrametricDist K
      inst✝ : IsDiscreteValuationRing (Subtype fun x => Membership.mem (Valued.integ …
      H : TotallyBounded Set.univ
      p : Subtype fun x => Membership.mem (Valued.integer K) x
      hp : Irreducible p
      this : Exists fun t => And (HasSubset.Subset t Set.univ) (And t.Finite (HasSub …
      ⊢ Finite (Valued.ResidueField K)
    -/
    simp only [Set.subset_univ, Set.univ_subset_iff, true_and] at this
    /-
      case mp.intro
      K : Type u_2
      inst✝² : NontriviallyNormedField K
      inst✝¹ : IsUltrametricDist K
      inst✝ : IsDiscreteValuationRing (Subtype fun x => Membership.mem (Valued.integ …
      H : TotallyBounded Set.univ
      p : Subtype fun x => Membership.mem (Valued.integer K) x
      hp : Irreducible p
      this : Exists fun t => And t.Finite (Eq (Set.iUnion fun y => Set.iUnion fun h  …
      ⊢ Finite (Valued.ResidueField K)
    -/
    obtain ⟨t, ht, ht'⟩ := this
    /-
      case mp.intro.intro.intro
      K : Type u_2
      inst✝² : NontriviallyNormedField K
      inst✝¹ : IsUltrametricDist K
      inst✝ : IsDiscreteValuationRing (Subtype fun x => Membership.mem (Valued.integ …
      H : TotallyBounded Set.univ
      p : Subtype fun x => Membership.mem (Valued.integer K) x
      hp : Irreducible p
      t : Set (Subtype fun x => Membership.mem (Valued.integer K) x)
      ht : t.Finite
      ht' : Eq (Set.iUnion fun y => Set.iUnion fun h => Metric.ball y (Norm.norm p)) …
      ⊢ Finite (Valued.ResidueField K)
    -/
    rw [← Set.finite_univ_iff]
    /-
      case mp.intro.intro.intro
      K : Type u_2
      inst✝² : NontriviallyNormedField K
      inst✝¹ : IsUltrametricDist K
      inst✝ : IsDiscreteValuationRing (Subtype fun x => Membership.mem (Valued.integ …
      H : TotallyBounded Set.univ
      p : Subtype fun x => Membership.mem (Valued.integer K) x
      hp : Irreducible p
      t : Set (Subtype fun x => Membership.mem (Valued.integer K) x)
      ht : t.Finite
      ht' : Eq (Set.iUnion fun y => Set.iUnion fun h => Metric.ball y (Norm.norm p)) …
      ⊢ Set.univ.Finite
    -/
    refine (ht.image (IsLocalRing.residue _)).subset ?_
    /-
      case mp.intro.intro.intro
      K : Type u_2
      inst✝² : NontriviallyNormedField K
      inst✝¹ : IsUltrametricDist K
      inst✝ : IsDiscreteValuationRing (Subtype fun x => Membership.mem (Valued.integ …
      H : TotallyBounded Set.univ
      p : Subtype fun x => Membership.mem (Valued.integer K) x
      hp : Irreducible p
      t : Set (Subtype fun x => Membership.mem (Valued.integer K) x)
      ht : t.Finite
      ht' : Eq (Set.iUnion fun y => Set.iUnion fun h => Metric.ball y (Norm.norm p)) …
      ⊢ HasSubset.Subset Set.univ (Set.image (⇑(IsLocalRing.residue (Subtype fun x = …
    -/
    rintro ⟨x⟩
    /-
      case mp.intro.intro.intro.mk
      K : Type u_2
      inst✝² : NontriviallyNormedField K
      inst✝¹ : IsUltrametricDist K
      inst✝ : IsDiscreteValuationRing (Subtype fun x => Membership.mem (Valued.integ …
      H : TotallyBounded Set.univ
      p : Subtype fun x => Membership.mem (Valued.integer K) x
      hp : Irreducible p
      t : Set (Subtype fun x => Membership.mem (Valued.integer K) x)
      ht : t.Finite
      ht' : Eq (Set.iUnion fun y => Set.iUnion fun h => Metric.ball y (Norm.norm p)) …
      a✝ : IsLocalRing.ResidueField (Subtype fun x => Membership.mem (Valued.integer …
      x : Subtype fun x => Membership.mem (Valued.integer K) x
      ⊢ Membership.mem Set.univ (Quot.mk (⇑(Submodule.quotientRel (IsLocalRing.maxim …
    -/
    replace ht' := ht'.ge (Set.mem_univ x)
    /-
      case mp.intro.intro.intro.mk
      K : Type u_2
      inst✝² : NontriviallyNormedField K
      inst✝¹ : IsUltrametricDist K
      inst✝ : IsDiscreteValuationRing (Subtype fun x => Membership.mem (Valued.integ …
      H : TotallyBounded Set.univ
      p : Subtype fun x => Membership.mem (Valued.integer K) x
      hp : Irreducible p
      t : Set (Subtype fun x => Membership.mem (Valued.integer K) x)
      ht : t.Finite
      a✝ : IsLocalRing.ResidueField (Subtype fun x => Membership.mem (Valued.integer …
      x : Subtype fun x => Membership.mem (Valued.integer K) x
      ht' : Membership.mem (Set.iUnion fun y => Set.iUnion fun h => Metric.ball y (N …
      ⊢ Membership.mem Set.univ (Quot.mk (⇑(Submodule.quotientRel (IsLocalRing.maxim …
    -/
    simp only [Set.mem_iUnion, Metric.mem_ball, exists_prop] at ht'
    /-
      case mp.intro.intro.intro.mk
      K : Type u_2
      inst✝² : NontriviallyNormedField K
      inst✝¹ : IsUltrametricDist K
      inst✝ : IsDiscreteValuationRing (Subtype fun x => Membership.mem (Valued.integ …
      H : TotallyBounded Set.univ
      p : Subtype fun x => Membership.mem (Valued.integer K) x
      hp : Irreducible p
      t : Set (Subtype fun x => Membership.mem (Valued.integer K) x)
      ht : t.Finite
      a✝ : IsLocalRing.ResidueField (Subtype fun x => Membership.mem (Valued.integer …
      x : Subtype fun x => Membership.mem (Valued.integer K) x
      ht' : Exists fun i => And (Membership.mem t i) (LT.lt (Dist.dist x i) (Norm.no …
      ⊢ Membership.mem Set.univ (Quot.mk (⇑(Submodule.quotientRel (IsLocalRing.maxim …
    -/
    obtain ⟨y, hy, hy'⟩ := ht'
    simp only [Submodule.Quotient.quot_mk_eq_mk, Ideal.Quotient.mk_eq_mk, Set.mem_univ,
      IsLocalRing.residue, Set.mem_image, true_implies]
    /-
      case mp.intro.intro.intro.mk.intro.intro
      K : Type u_2
      inst✝² : NontriviallyNormedField K
      inst✝¹ : IsUltrametricDist K
      inst✝ : IsDiscreteValuationRing (Subtype fun x => Membership.mem (Valued.integ …
      H : TotallyBounded Set.univ
      p : Subtype fun x => Membership.mem (Valued.integer K) x
      hp : Irreducible p
      t : Set (Subtype fun x => Membership.mem (Valued.integer K) x)
      ht : t.Finite
      a✝ : IsLocalRing.ResidueField (Subtype fun x => Membership.mem (Valued.integer …
      x y : Subtype fun x => Membership.mem (Valued.integer K) x
      hy : Membership.mem t y
      hy' : LT.lt (Dist.dist x y) (Norm.norm p)
      ⊢ Exists fun x_1 => And (Membership.mem t x_1) (Eq ((Ideal.Quotient.mk (IsLoca …
    -/
    refine ⟨y, hy, ?_⟩
    /-
      case mp.intro.intro.intro.mk.intro.intro
      K : Type u_2
      inst✝² : NontriviallyNormedField K
      inst✝¹ : IsUltrametricDist K
      inst✝ : IsDiscreteValuationRing (Subtype fun x => Membership.mem (Valued.integ …
      H : TotallyBounded Set.univ
      p : Subtype fun x => Membership.mem (Valued.integer K) x
      hp : Irreducible p
      t : Set (Subtype fun x => Membership.mem (Valued.integer K) x)
      ht : t.Finite
      a✝ : IsLocalRing.ResidueField (Subtype fun x => Membership.mem (Valued.integer …
      x y : Subtype fun x => Membership.mem (Valued.integer K) x
      hy : Membership.mem t y
      hy' : LT.lt (Dist.dist x y) (Norm.norm p)
      ⊢ Eq ((Ideal.Quotient.mk (IsLocalRing.maximalIdeal (Subtype fun x => Membershi …
    -/
    convert (Ideal.Quotient.mk_eq_mk_iff_sub_mem (I := 𝓂[K]) y x).mpr _
    -- TODO: make Valued.maximalIdeal abbreviations instead of def
    rw [Valued.maximalIdeal, hp.maximalIdeal_eq, ← SetLike.mem_coe,
      coe_span_singleton_eq_closedBall]
    /-
      case mp.intro.intro.intro.mk.intro.intro
      K : Type u_2
      inst✝² : NontriviallyNormedField K
      inst✝¹ : IsUltrametricDist K
      inst✝ : IsDiscreteValuationRing (Subtype fun x => Membership.mem (Valued.integ …
      H : TotallyBounded Set.univ
      p : Subtype fun x => Membership.mem (Valued.integer K) x
      hp : Irreducible p
      t : Set (Subtype fun x => Membership.mem (Valued.integer K) x)
      ht : t.Finite
      a✝ : IsLocalRing.ResidueField (Subtype fun x => Membership.mem (Valued.integer …
      x y : Subtype fun x => Membership.mem (Valued.integer K) x
      hy : Membership.mem t y
      hy' : LT.lt (Dist.dist x y) (Norm.norm p)
      ⊢ Membership.mem (Metric.closedBall 0 (Norm.norm p)) (HSub.hSub y x)
    -/
    rw [dist_comm] at hy'
    /-
      case mp.intro.intro.intro.mk.intro.intro
      K : Type u_2
      inst✝² : NontriviallyNormedField K
      inst✝¹ : IsUltrametricDist K
      inst✝ : IsDiscreteValuationRing (Subtype fun x => Membership.mem (Valued.integ …
      H : TotallyBounded Set.univ
      p : Subtype fun x => Membership.mem (Valued.integer K) x
      hp : Irreducible p
      t : Set (Subtype fun x => Membership.mem (Valued.integer K) x)
      ht : t.Finite
      a✝ : IsLocalRing.ResidueField (Subtype fun x => Membership.mem (Valued.integer …
      x y : Subtype fun x => Membership.mem (Valued.integer K) x
      hy : Membership.mem t y
      hy' : LT.lt (Dist.dist y x) (Norm.norm p)
      ⊢ Membership.mem (Metric.closedBall 0 (Norm.norm p)) (HSub.hSub y x)
    -/
    simpa [dist_eq_norm] using hy'.le
    /-
      🎉 no goals
    -/
    /-
      case mpr
      K : Type u_2
      inst✝² : NontriviallyNormedField K
      inst✝¹ : IsUltrametricDist K
      inst✝ : IsDiscreteValuationRing (Subtype fun x => Membership.mem (Valued.integ …
      ⊢ Finite (Valued.ResidueField K) → TotallyBounded Set.univ
    -/
  · intro H
    /-
      case mpr
      K : Type u_2
      inst✝² : NontriviallyNormedField K
      inst✝¹ : IsUltrametricDist K
      inst✝ : IsDiscreteValuationRing (Subtype fun x => Membership.mem (Valued.integ …
      H : Finite (Valued.ResidueField K)
      ⊢ TotallyBounded Set.univ
    -/
    rw [Metric.totallyBounded_iff]
    /-
      case mpr
      K : Type u_2
      inst✝² : NontriviallyNormedField K
      inst✝¹ : IsUltrametricDist K
      inst✝ : IsDiscreteValuationRing (Subtype fun x => Membership.mem (Valued.integ …
      H : Finite (Valued.ResidueField K)
      ⊢ ∀ (ε : Real), GT.gt ε 0 → Exists fun t => And t.Finite (HasSubset.Subset Set …
    -/
    intro ε εpos
    /-
      case mpr
      K : Type u_2
      inst✝² : NontriviallyNormedField K
      inst✝¹ : IsUltrametricDist K
      inst✝ : IsDiscreteValuationRing (Subtype fun x => Membership.mem (Valued.integ …
      H : Finite (Valued.ResidueField K)
      ε : Real
      εpos : GT.gt ε 0
      ⊢ Exists fun t => And t.Finite (HasSubset.Subset Set.univ (Set.iUnion fun y => …
    -/
    obtain ⟨p, hp⟩ := IsDiscreteValuationRing.exists_irreducible 𝒪[K]
    /-
      case mpr.intro
      K : Type u_2
      inst✝² : NontriviallyNormedField K
      inst✝¹ : IsUltrametricDist K
      inst✝ : IsDiscreteValuationRing (Subtype fun x => Membership.mem (Valued.integ …
      H : Finite (Valued.ResidueField K)
      ε : Real
      εpos : GT.gt ε 0
      p : Subtype fun x => Membership.mem (Valued.integer K) x
      hp : Irreducible p
      ⊢ Exists fun t => And t.Finite (HasSubset.Subset Set.univ (Set.iUnion fun y => …
    -/
    have hp' := norm_irreducible_lt_one hp
    /-
      case mpr.intro
      K : Type u_2
      inst✝² : NontriviallyNormedField K
      inst✝¹ : IsUltrametricDist K
      inst✝ : IsDiscreteValuationRing (Subtype fun x => Membership.mem (Valued.integ …
      H : Finite (Valued.ResidueField K)
      ε : Real
      εpos : GT.gt ε 0
      p : Subtype fun x => Membership.mem (Valued.integer K) x
      hp : Irreducible p
      hp' : LT.lt (Norm.norm p) 1
      ⊢ Exists fun t => And t.Finite (HasSubset.Subset Set.univ (Set.iUnion fun y => …
    -/
    obtain ⟨n, hn⟩ : ∃ n : ℕ, ‖p‖ ^ n < ε := exists_pow_lt_of_lt_one εpos hp'
    /-
      case mpr.intro.intro
      K : Type u_2
      inst✝² : NontriviallyNormedField K
      inst✝¹ : IsUltrametricDist K
      inst✝ : IsDiscreteValuationRing (Subtype fun x => Membership.mem (Valued.integ …
      H : Finite (Valued.ResidueField K)
      ε : Real
      εpos : GT.gt ε 0
      p : Subtype fun x => Membership.mem (Valued.integer K) x
      hp : Irreducible p
      hp' : LT.lt (Norm.norm p) 1
      n : Nat
      hn : LT.lt (HPow.hPow (Norm.norm p) n) ε
      ⊢ Exists fun t => And t.Finite (HasSubset.Subset Set.univ (Set.iUnion fun y => …
    -/
    have hF := finite_quotient_maximalIdeal_pow_of_finite_residueField H n
    /-
      case mpr.intro.intro
      K : Type u_2
      inst✝² : NontriviallyNormedField K
      inst✝¹ : IsUltrametricDist K
      inst✝ : IsDiscreteValuationRing (Subtype fun x => Membership.mem (Valued.integ …
      H : Finite (Valued.ResidueField K)
      ε : Real
      εpos : GT.gt ε 0
      p : Subtype fun x => Membership.mem (Valued.integer K) x
      hp : Irreducible p
      hp' : LT.lt (Norm.norm p) 1
      n : Nat
      hn : LT.lt (HPow.hPow (Norm.norm p) n) ε
      hF : Finite (HasQuotient.Quotient (Subtype fun x => Membership.mem (Valued.int …
      ⊢ Exists fun t => And t.Finite (HasSubset.Subset Set.univ (Set.iUnion fun y => …
    -/
    refine ⟨Quotient.out '' (Set.univ (α := 𝒪[K] ⧸ (𝓂[K] ^ n))), Set.toFinite _, ?_⟩
    simp only [Ideal.univ_eq_iUnion_image_add (𝓂[K] ^ n), hp.maximalIdeal_pow_eq_closedBall_pow,
      AddSubgroupClass.coe_norm, Set.image_add_left, preimage_add_closedBall, sub_neg_eq_add,
      zero_add, Set.image_univ, Set.mem_range, Set.iUnion_exists, Set.iUnion_iUnion_eq',
      Set.iUnion_subset_iff, Metric.vadd_closedBall, vadd_eq_add, add_zero]
    /-
      case mpr.intro.intro
      K : Type u_2
      inst✝² : NontriviallyNormedField K
      inst✝¹ : IsUltrametricDist K
      inst✝ : IsDiscreteValuationRing (Subtype fun x => Membership.mem (Valued.integ …
      H : Finite (Valued.ResidueField K)
      ε : Real
      εpos : GT.gt ε 0
      p : Subtype fun x => Membership.mem (Valued.integer K) x
      hp : Irreducible p
      hp' : LT.lt (Norm.norm p) 1
      n : Nat
      hn : LT.lt (HPow.hPow (Norm.norm p) n) ε
      hF : Finite (HasQuotient.Quotient (Subtype fun x => Membership.mem (Valued.int …
      ⊢ ∀ (i : HasQuotient.Quotient (Subtype fun x => Membership.mem (Valued.integer …
    -/
    intro
    /-
      case mpr.intro.intro
      K : Type u_2
      inst✝² : NontriviallyNormedField K
      inst✝¹ : IsUltrametricDist K
      inst✝ : IsDiscreteValuationRing (Subtype fun x => Membership.mem (Valued.integ …
      H : Finite (Valued.ResidueField K)
      ε : Real
      εpos : GT.gt ε 0
      p : Subtype fun x => Membership.mem (Valued.integer K) x
      hp : Irreducible p
      hp' : LT.lt (Norm.norm p) 1
      n : Nat
      hn : LT.lt (HPow.hPow (Norm.norm p) n) ε
      hF : Finite (HasQuotient.Quotient (Subtype fun x => Membership.mem (Valued.int …
      i✝ : HasQuotient.Quotient (Subtype fun x => Membership.mem (Valued.integer K)  …
      ⊢ HasSubset.Subset (Metric.closedBall (Quotient.out i✝) (HPow.hPow (Norm.norm  …
    -/
    exact (Metric.closedBall_subset_ball hn).trans (Set.subset_iUnion_of_subset _ le_rfl)
    /-
      🎉 no goals
    -/


